"""Coleta imagens do CAPTCHA dos Correios para um dataset 85/10/5."""

import argparse
import hashlib
import http.cookiejar
import logging
import math
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener
from uuid import uuid4

URL = "https://rastreamento.correios.com.br/app/index.php#"
SPLITS = {"train": 85, "validation": 10, "test": 5}
LOGGER = logging.getLogger(__name__)


class _CaptchaParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.src = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "img" and attrs.get("id") == "captcha_image":
            self.src = attrs.get("src")


def _split_counts(count):
    """Arredonda pelos maiores restos, mantendo a soma igual a count."""
    result = {name: count * weight // 100 for name, weight in SPLITS.items()}
    order = sorted(SPLITS, key=lambda name: count * SPLITS[name] % 100, reverse=True)
    for name in order[:count - sum(result.values())]:
        result[name] += 1
    return result


def _retry_after_seconds(value):
    """Interpreta Retry-After em segundos ou como data HTTP."""
    if not value:
        return 0.0
    value = value.strip()
    try:
        if value.isascii() and value.isdigit():
            seconds = float(value)
            return seconds if math.isfinite(seconds) else 0.0
        deadline = parsedate_to_datetime(value)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _fetch(session, request, *, timeout, retries, backoff):
    """Repete GETs em falhas temporárias, respeitando o prazo do servidor."""
    for attempt in range(retries + 1):
        retry_after = 0.0
        rate_limited = False
        try:
            with session.open(request, timeout=timeout) as response:
                return (
                    response.geturl(),
                    response.headers.get_content_type(),
                    response.headers.get_content_charset() or "utf-8",
                    response.read(),
                )
        except HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504}:
                error.close()
                raise
            reason = f"HTTP {error.code} ({error.reason})"
            rate_limited = error.code == 429
            retry_after = _retry_after_seconds(error.headers.get("Retry-After"))
            error.close()
        except (URLError, TimeoutError, ConnectionError, IncompleteRead, RemoteDisconnected) as error:
            reason = str(error)

        if attempt == retries:
            raise RuntimeError(
                f"{reason}: falha após {retries + 1} tentativa(s). "
                "As imagens já salvas foram preservadas. Aguarde antes de tentar novamente."
            )
        wait = max(backoff * 2 ** attempt, retry_after, 60.0 if rate_limited else 0.0)
        LOGGER.warning(
            "%s. Aguardando %.1f s antes da nova tentativa %s/%s.",
            reason, wait, attempt + 1, retries,
        )
        time.sleep(wait)


def collect_captchas(
    count: int,
    output_dir: str | Path = "dataset",
    *,
    delay: float = 5.0,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 60.0,
    seed: int = 42,
    url: str = URL,
) -> dict[str, int]:
    """Baixa count imagens novas e retorna as quantidades salvas por subconjunto.

    A proporção se aplica ao lote desta chamada. Arquivos existentes são
    preservados e seus hashes são usados para evitar duplicatas entre pastas.
    Em caso de erro, as imagens já salvas permanecem no disco e uma exceção
    é propagada; um lote interrompido pode ter proporções incompletas.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count deve ser um inteiro positivo.")
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("delay deve ser um número finito maior ou igual a zero.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout deve ser um número finito maior que zero.")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError("retries deve ser um inteiro maior ou igual a zero.")
    if not math.isfinite(backoff) or backoff <= 0:
        raise ValueError("backoff deve ser um número finito maior que zero.")

    root = Path(output_dir)
    seen = set()
    for split in SPLITS:
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        for path in folder.glob("*.png"):
            seen.add(hashlib.sha256(path.read_bytes()).hexdigest())

    counts = _split_counts(count)
    destinations = [name for name, total in counts.items() for _ in range(total)]
    random.Random(seed).shuffle(destinations)
    saved = dict.fromkeys(SPLITS, 0)

    session = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
    session.addheaders = [("User-Agent", "VulnCaptchaDataset/1.0")]
    page_url, _, charset, page_data = _fetch(
        session, url, timeout=timeout, retries=retries, backoff=backoff
    )
    html = page_data.decode(charset)
    parser = _CaptchaParser()
    parser.feed(html)
    if not parser.src:
        raise RuntimeError("Não foi encontrado um img#captcha_image com atributo src.")
    image_url = urljoin(page_url, parser.src)

    # Limita tentativas para não permanecer num loop caso o servidor repita imagens.
    for attempt in range(count * 3):
        if attempt:
            time.sleep(delay)
        parts = urlsplit(image_url)
        query = parse_qsl(parts.query, keep_blank_values=True)
        query.append(("_dataset", uuid4().hex))
        fresh_url = urlunsplit(parts._replace(query=urlencode(query)))
        request = Request(fresh_url, headers={"Referer": page_url, "Cache-Control": "no-cache"})
        _, content_type, _, data = _fetch(
            session, request, timeout=timeout, retries=retries, backoff=backoff
        )
        if content_type != "image/png" or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise RuntimeError(f"Resposta inesperada do CAPTCHA: {content_type}; esperado PNG.")

        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            LOGGER.info("Imagem repetida ignorada.")
            continue
        split = destinations[sum(saved.values())]
        path = root / split / f"{digest}.png"
        with path.open("xb") as output:
            output.write(data)
        seen.add(digest)
        saved[split] += 1
        LOGGER.info("%s/%s: %s", sum(saved.values()), count, path)
        if sum(saved.values()) == count:
            return saved

    raise RuntimeError(f"Muitas imagens repetidas: foram salvas {sum(saved.values())}/{count}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Imagens novas a coletar (padrão: 100).")
    parser.add_argument("--output", type=Path, default=Path("dataset"), help="Pasta do dataset.")
    parser.add_argument("--delay", type=float, default=5.0, help="Intervalo em segundos (padrão: 5).")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout de cada requisição.")
    parser.add_argument("--retries", type=int, default=3, help="Novas tentativas por requisição (padrão: 3).")
    parser.add_argument("--backoff", type=float, default=60.0, help="Espera inicial entre novas tentativas (padrão: 60 s).")
    parser.add_argument("--seed", type=int, default=42, help="Semente para embaralhar a divisão.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        result = collect_captchas(
            args.count, args.output, delay=args.delay, timeout=args.timeout,
            retries=args.retries, backoff=args.backoff, seed=args.seed
        )
    except (ValueError, RuntimeError, OSError, URLError) as error:
        parser.exit(1, f"Erro: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nColeta interrompida; imagens já salvas foram preservadas.\n")
    print("Concluído: " + ", ".join(f"{name}={total}" for name, total in result.items()))


if __name__ == "__main__":
    main()
