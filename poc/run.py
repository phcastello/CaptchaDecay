"""Explicit CLI; importing or omitting --live never contacts Correios."""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
from pathlib import Path
import sys
import time

from .recognizer import DEFAULT_CHECKPOINT, Recognizer
from .results import Artifacts, outcome, validate_code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Uma consulta Correios com CRNN D congelada.")
    parser.add_argument("--live", action="store_true", help="Autoriza acesso ao site público.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true", help="Com --live: inspeciona sem preencher/submeter.")
    mode.add_argument("--smoke-image", type=Path, help="Inferência local em PNG sintético; sem rede.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--tracking-code", help="UM código; prefira entrada interativa ou CAPTCHADECAY_TRACKING_CODE.")
    parser.add_argument("--max-attempts", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=30000)
    args = parser.parse_args(argv)
    if args.live and args.smoke_image:
        parser.error("--smoke-image é exclusivamente local; não use --live.")
    if not 1 <= args.timeout_ms <= 120000 or args.threads < 1:
        parser.error("Timeout deve estar entre 1 e 120000 ms; threads deve ser positivo.")
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    if not args.live and not args.smoke_image:
        parser.print_help()
        print("\nSem --live: nenhum acesso ao site.")
        return 0
    started = time.perf_counter()
    artifacts = None
    try:
        if args.smoke_image:
            resolved = args.smoke_image.resolve()
            if not resolved.is_relative_to((Path.cwd() / "dataset/synthetic").resolve()):
                raise ValueError("Smoke test exige um PNG local dentro de dataset/synthetic.")
            artifacts = Artifacts(mode="local_model_smoke")
            artifacts.environment()
            recognizer = Recognizer(args.checkpoint, args.device, args.threads)
            prediction = recognizer.recognize(resolved.read_bytes())
            recognizer.verify_preservation()
            artifacts.result.update(recognizer=recognizer.metadata, inference=prediction,
                                    model_preservation_verified=True)
            result = artifacts.finish(outcome("site_changed_or_unrecognized", "local_inference_only"),
                                      time.perf_counter() - started)
            print(json.dumps({"mode": result["mode"], "text": prediction["text"],
                              "inference_seconds": prediction["inference_seconds"],
                              "artifacts": str(artifacts.directory)}, ensure_ascii=False))
            return 0
        code = None
        if not args.inspect:
            code = args.tracking_code or os.environ.get("CAPTCHADECAY_TRACKING_CODE")
            if not code:
                if not sys.stdin.isatty():
                    raise ValueError("Código ausente: use terminal interativo, variável de ambiente ou --tracking-code.")
                code = getpass.getpass("Código de rastreamento (entrada oculta): ")
            validate_code(code)
        artifacts = Artifacts(mode="public_inspection" if args.inspect else "live",
                              code=code, max_attempts=args.max_attempts, headed=args.headed)
        # A missing/incompatible model fails before browser launch/navigation.
        recognizer = None if args.inspect else Recognizer(args.checkpoint, args.device, args.threads)
        from .correios import execute
        result = asyncio.run(execute(recognizer, code, artifacts, headed=args.headed,
            max_attempts=args.max_attempts, timeout_ms=args.timeout_ms, inspect=args.inspect))
        if result["evidence"]["rule"] == "headed_requires_display":
            print("Modo --headed exige DISPLAY ou WAYLAND_DISPLAY; não há display disponível.", file=sys.stderr)
        print(json.dumps({"status": result["status"], "success": result["success"],
                          "reason": result["evidence"]["rule"], "submissions": result["submissions"],
                          "artifacts": str(artifacts.directory)},
                         ensure_ascii=False))
        return 0 if result["success"] or (args.inspect and "inspection" in result) else 1
    except (ValueError, RuntimeError, OSError, EOFError, KeyboardInterrupt) as error:
        # No exception rendering: browser/OS errors may include sensitive input.
        if artifacts:
            artifacts.finish(outcome("inference_error", "preflight_failed_before_navigation"),
                             time.perf_counter() - started)
        safe_prefixes = ("Checkpoint D ausente:", "SHA-256 incompatível:", "Época incompatível:",
                         "Classes do checkpoint", "Informe UM código:", "Código ausente:",
                         "Smoke test exige", "CUDA was explicitly requested")
        detail = str(error)
        if not detail.startswith(safe_prefixes):
            detail = "Pré-verificação local falhou; confira checkpoint D, dispositivo, PNG e dependências."
        print(f"ERRO: {detail}", file=sys.stderr)
        if artifacts:
            print(f"Evidências: {artifacts.directory}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
