"""Conservative classification and allowlisted, privacy-preserving artifacts."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import uuid
from importlib.metadata import version

from ml.train import atomic_write, versions

STATUSES = frozenset({
    "tracking_success", "captcha_rejected", "object_not_found",
    "challenge_expired_or_changed", "challenge_not_present", "rate_limited",
    "access_blocked", "network_error", "site_changed_or_unrecognized", "inference_error",
})


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_code(code):
    if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{2}[0-9]{9}[A-Z]{2}", code):
        raise ValueError("Informe UM código: duas letras maiúsculas, nove dígitos e duas letras.")
    return code


def mask_code(code):
    return code[:2] + "*********" + code[-2:]


def outcome(status, rule, **evidence):
    if status not in STATUSES:
        raise ValueError("Unknown result status")
    return {"status": status, "success": status == "tracking_success",
            "evidence": {"rule": rule, **evidence}}


def classify_response(http_status, payload, *, correlated, tracking_code,
                      captcha_error="Captcha inválido", not_found_messages=()):
    """No raw service message, event/address or identifier leaves this function.

    The captcha literal and success field names were observed in public JS.
    not_found_messages is an adapter contract; production has no assumed literal.
    An explicit Portuguese negative existence statement can also be recognized
    semantically; this does not assert an unobserved server error string.
    Unknown messages stay unknown, including purported session errors.
    """
    if http_status == 429:
        return outcome("rate_limited", "http_429", http_status=429)
    if http_status in (401, 403):
        return outcome("access_blocked", "http_access_denied", http_status=http_status)
    if not correlated:
        return outcome("site_changed_or_unrecognized", "response_not_correlated")
    if http_status != 200:
        return outcome("network_error", "http_error", http_status=http_status)
    if not isinstance(payload, dict):
        return outcome("site_changed_or_unrecognized", "unknown_response_schema")
    message = payload.get("mensagem")
    explicit_not_found = isinstance(message, str) and bool(re.search(
        r"\bobjeto\s+(?:(?:[A-Z]{2}\d{9}[A-Z]{2})\s+)?n[ãa]o\s+(?:foi\s+)?(?:encontrado|localizado)\b",
        message, re.IGNORECASE))
    if payload.get("erro") or (payload.get("codObjeto") == tracking_code and explicit_not_found):
        message_hash = hashlib.sha256(str(message).encode()).hexdigest()
        if message == captcha_error:
            return outcome("captcha_rejected", "explicit_captcha_error", message_sha256=message_hash)
        if isinstance(message, str) and (message in not_found_messages or explicit_not_found):
            return outcome("object_not_found", "explicit_object_not_found", message_sha256=message_hash,
                           captcha_accepted=None)
        return outcome("site_changed_or_unrecognized", "unrecognized_service_error",
                       message_sha256=message_hash)
    if payload.get("codObjeto") != tracking_code:
        return outcome("site_changed_or_unrecognized", "object_mismatch_or_missing")
    events = payload.get("eventos")
    if (not isinstance(events, list) or not events or any(
        not isinstance(e, dict) or not isinstance(e.get("descricao"), str) or not e["descricao"].strip()
        or not isinstance(e.get("dtHrCriado"), dict)
        or not isinstance(e["dtHrCriado"].get("date"), str)
        or not re.match(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", e["dtHrCriado"]["date"])
        for e in events
    )):
        return outcome("site_changed_or_unrecognized", "no_valid_tracking_events")
    return outcome("tracking_success", "current_response_matching_object_with_events",
                   object_matches=True, event_count=len(events), correlated=True)


class Artifacts:
    """A new private directory per run; callers only supply curated metadata."""

    def __init__(self, root="outputs/poc", *, mode, code=None, max_attempts=1, headed=False):
        self.monotonic_started = time.perf_counter()
        self.started = utc_now()
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
        self.directory = Path(root) / self.run_id
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(self.directory, 0o700)
        self.attempts = []
        self.result = {
            "run_id": self.run_id, "started_at": self.started, "mode": mode,
            "tracking_code_masked": mask_code(code) if code else None,
            "max_attempts": max_attempts, "browser_mode": "headed" if headed else "headless",
            "submissions": 0, "attempt_count": 0, "queries_started": 0,
            "queries_completed": 0, "browser_closed": False,
            "screenshot": {"saved": False, "reason": "Raw screenshots omitted: arbitrary service personal data cannot be safely masked."},
        }
        self.log("run_created")

    def environment(self):
        def git(*args):
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        try:
            revision, dirty = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
            patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"])
            code_hash = hashlib.sha256(patch)
            for path in sorted(Path("poc").glob("*.py")):
                code_hash.update(path.name.encode())
                code_hash.update(path.read_bytes())
            identity = code_hash.hexdigest()
        except (OSError, subprocess.SubprocessError):
            revision, dirty, identity = None, None, None
        self.result["environment"] = {**versions(), "platform": platform.platform(),
            "playwright": version("playwright"), "git_revision": revision, "git_dirty": dirty,
            "working_code_sha256": identity}

    def log(self, event, **safe_fields):
        with (self.directory / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": utc_now(), "event": event, **safe_fields}, ensure_ascii=False) + "\n")

    def save_captcha(self, number, png):
        path = self.directory / f"captcha-{number:02d}.png"
        with path.open("xb") as stream:
            stream.write(png)
        return path.name

    def finish(self, final, total_seconds):
        self.result.update(final)
        self.result.update(finished_at=utc_now(), attempt_count=len(self.attempts),
                           total_seconds=total_seconds, attempts=self.attempts,
                           queries_completed=int(final["success"]))
        atomic_write(self.directory / "result.json", lambda s: json.dump(
            self.result, s, indent=2, ensure_ascii=False, allow_nan=False))
        fields = ("attempt", "started_at", "status", "image_sha256", "prediction",
                  "submitted_prediction", "submissions", "capture_seconds", "inference_seconds",
                  "submission_seconds", "total_seconds")
        def write_csv(stream):
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.attempts)
        atomic_write(self.directory / "attempts.csv", write_csv)
        self.log("run_finished", status=final["status"], submissions=self.result["submissions"])
        return self.result
