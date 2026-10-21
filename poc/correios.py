"""Observed Correios UI contract and sequential browser execution.

2026-10-21: selectors and normal fetch URL observed without a tracking query,
in /app/index.php, /app/index.js and /core/js/rastroUnico.js. The JSON success
schema is inferred from that renderer, not yet guaranteed by a live response.
Only tests inject a different contract. Production CLI has no URL override.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import io
import os
import sys
import time
from urllib.parse import parse_qs, urlsplit

from PIL import Image
from playwright.async_api import async_playwright, Error as BrowserError, TimeoutError as BrowserTimeout

from .results import Artifacts, classify_response, outcome, utc_now, validate_code


@dataclass(frozen=True)
class Interface:
    url: str = "https://rastreamento.correios.com.br/app/index.php"
    image_path: str = "/core/securimage/securimage_show.php"
    result_path: str = "/app/resultado.php"
    image: str = "img#captcha_image"
    tracking: str = "#objeto"
    captcha: str = "#captcha"
    submit: str = "#b-pesquisar"
    refresh: str = "#captcha_refresh_btn"
    not_found_messages: tuple[str, ...] = ()  # No real server literal observed yet.


CORREIOS = Interface()

# Observe the displayed image; never read or override the site's validation JS.
# A capture-phase guard only prevents submitting an already changed challenge.
MONITOR = """({image, submit}) => {
  const state = window.__captchaDecay = {revision: 0, pin: null, blocked: false};
  const isImage = e => e && e.matches && e.matches(image);
  new MutationObserver(records => {
    for (const r of records) {
      if ((r.type === 'attributes' && isImage(r.target)) ||
          (r.type === 'childList' && [...r.addedNodes, ...r.removedNodes].some(n =>
            isImage(n) || (n.querySelector && n.querySelector(image))))) state.revision++;
    }
  }).observe(document, {subtree:true,childList:true,attributes:true,attributeFilter:['src','srcset','id']});
  document.addEventListener('load', e => { if (isImage(e.target)) state.revision++; }, true);
  document.addEventListener('click', e => {
    if (!e.target.closest || !e.target.closest(submit) || !state.pin) return;
    const el = document.querySelector(image), p = state.pin;
    if (!el || !el.complete || !el.naturalWidth || el.currentSrc !== p.src || state.revision !== p.revision) {
      state.blocked = true; e.preventDefault(); e.stopImmediatePropagation();
    }
  }, true);
}"""


class FlowError(Exception):
    def __init__(self, result):
        super().__init__(result["status"])
        self.result = result


class BrowserFlow:
    def __init__(self, page, interface, timeout_ms):
        self.page, self.interface, self.timeout_ms = page, interface, timeout_ms
        self.origin = urlsplit(interface.url).netloc
        self.images = []
        self.fatal = None
        self.fatal_event = asyncio.Event()
        self.armed = None
        self.response_future = None
        self.current = None
        self.submissions = 0
        self.last_submission = None
        self.blocked_submission = False
        page.on("request", self._on_request)
        page.on("response", self._on_response)

    def _same_origin(self, url):
        parts = urlsplit(url)
        base = urlsplit(self.interface.url)
        return parts.scheme == base.scheme and parts.netloc == self.origin

    def _on_request(self, request):
        if (self._same_origin(request.url) and urlsplit(request.url).path == self.interface.image_path
                and request.resource_type == "image"):
            self.images.append({"request": request, "response": None, "at": utc_now()})

    def _on_response(self, response):
        if not self._same_origin(response.url):
            return
        if response.status in (401, 403, 429):
            # Any denial from the tracking origin stops this session immediately.
            self.fatal = outcome("rate_limited" if response.status == 429 else "access_blocked",
                                 "http_429" if response.status == 429 else "http_access_denied",
                                 http_status=response.status)
            retry_after = response.headers.get("retry-after", "")
            if retry_after:
                # Retain a parsed time, never arbitrary header contents.
                from captcha_dataset import _retry_after_seconds
                self.fatal["evidence"]["retry_after_seconds"] = _retry_after_seconds(retry_after)
            self.fatal_event.set()
        for image in reversed(self.images):
            if image["request"] == response.request:
                image["response"] = response
                image["received_at"] = utc_now()
                break
        if (self.current and response.request == self.current
                and self.response_future and not self.response_future.done()):
            self.response_future.set_result(response)

    async def bounded(self, awaitable):
        operation = asyncio.ensure_future(awaitable)
        denied = asyncio.create_task(self.fatal_event.wait())
        try:
            done, _ = await asyncio.wait({operation, denied}, timeout=self.timeout_ms / 1000,
                                         return_when=asyncio.FIRST_COMPLETED)
            if self.fatal:
                raise FlowError(self.fatal)
            if operation not in done:
                raise asyncio.TimeoutError()
            return await operation
        finally:
            for task in (operation, denied):
                if not task.done():
                    task.cancel()
            await asyncio.gather(operation, denied, return_exceptions=True)

    async def install(self):
        import json
        await self.page.add_init_script("(" + MONITOR + ")(" + json.dumps(
            {"image": self.interface.image, "submit": self.interface.submit}) + ")")
        # The page still makes its normal request. This gate only prevents a
        # stale/mismatched/duplicate outgoing submission; it creates no request.
        await self.page.route("**/*", self._route)

    async def _route(self, route):
        request = route.request
        if self.fatal:
            await route.abort("aborted")
            return
        if not (self._same_origin(request.url) and urlsplit(request.url).path == self.interface.result_path):
            await route.fallback()
            return
        query = parse_qs(urlsplit(request.url).query, keep_blank_values=True)
        pin = self.armed
        if (self.fatal or not pin or self.current is not None
                or query.get("objeto") != [pin["code"]] or query.get("captcha") != [pin["prediction"]]
                or not await self.unchanged(pin["snapshot"], pin["image_count"])):
            self.blocked_submission = True
            await route.abort("aborted")
            return
        self.current = request
        self.submissions += 1
        self.last_submission = utc_now()
        await route.continue_()

    async def open(self):
        response = await self.bounded(self.page.goto(self.interface.url, wait_until="load",
                                                      timeout=self.timeout_ms))
        if response is None or response.status != 200:
            raise FlowError(outcome("network_error", "initial_document_error"))
        # Explicit denial can also be delivered as HTTP 200. Do not infer an OCR error.
        text = (await self.bounded(self.page.locator("body").inner_text())).strip().casefold()
        if text.startswith(("access denied", "request rejected", "acesso negado", "acesso bloqueado")):
            raise FlowError(outcome("access_blocked", "explicit_denial_document"))

    async def snapshot(self):
        return await self.page.locator(self.interface.image).evaluate("""el => ({
            src:el.currentSrc, revision:window.__captchaDecay.revision,
            complete:el.complete, width:el.naturalWidth, height:el.naturalHeight
        })""")

    async def capture(self):
        image = self.page.locator(self.interface.image)
        if await image.count() != 1 or not await image.is_visible():
            raise FlowError(outcome("challenge_not_present", "displayed_captcha_absent"))
        for selector in (self.interface.tracking, self.interface.captcha, self.interface.submit):
            if await self.page.locator(selector).count() != 1:
                raise FlowError(outcome("site_changed_or_unrecognized", "required_control_missing"))
        await self.bounded(self.page.wait_for_function(
            "sel => {const i=document.querySelector(sel); return i && i.complete && i.naturalWidth>0}",
            arg=self.interface.image, timeout=self.timeout_ms))
        snapshot = await self.snapshot()
        matches = [i for i in self.images if i["request"].url == snapshot["src"]]
        if not matches or not matches[-1]["response"]:
            raise FlowError(outcome("site_changed_or_unrecognized", "displayed_image_response_unavailable"))
        captured = matches[-1]
        count = len(self.images)
        response = captured["response"]
        if response.status != 200:
            raise FlowError(outcome("network_error", "captcha_http_error", http_status=response.status))
        png = await self.bounded(response.body())  # Bytes already received, never another GET.
        try:
            with Image.open(io.BytesIO(png)) as decoded:
                if (decoded.format != "PNG" or decoded.size != (215, 80)
                        or getattr(decoded, "n_frames", 1) != 1):
                    raise ValueError("Expected original 215x80 PNG")
                decoded.verify()
            if (snapshot["width"], snapshot["height"]) != (215, 80):
                raise ValueError("Displayed natural dimensions differ")
        except (OSError, ValueError, SyntaxError):
            raise FlowError(outcome("inference_error", "unsupported_captcha_format_or_dimensions")) from None
        if not await self.unchanged(snapshot, count):
            raise FlowError(outcome("challenge_expired_or_changed", "image_changed_during_capture"))
        return png, snapshot, count, utc_now()

    async def unchanged(self, snapshot, count):
        if self.fatal or len(self.images) != count:
            return False
        try:
            current = await self.snapshot()
            return current == snapshot and current["complete"]
        except BrowserError:
            return False

    async def submit(self, code, prediction, snapshot, count):
        await self.page.locator(self.interface.tracking).fill(code)
        await self.page.locator(self.interface.captcha).fill(prediction)
        if (await self.page.locator(self.interface.tracking).input_value() != code or
                await self.page.locator(self.interface.captcha).input_value() != prediction):
            raise FlowError(outcome("site_changed_or_unrecognized", "field_value_changed"))
        button = self.page.locator(self.interface.submit)
        await self.bounded(button.click(trial=True, timeout=self.timeout_ms))
        if not await self.unchanged(snapshot, count):
            raise FlowError(outcome("challenge_expired_or_changed", "image_changed_before_submission"))
        await self.page.evaluate("p => {window.__captchaDecay.pin=p; window.__captchaDecay.blocked=false}",
                                 {"src": snapshot["src"], "revision": snapshot["revision"]})
        self.armed = {"code": code, "prediction": prediction, "snapshot": snapshot, "image_count": count}
        self.current = None
        self.blocked_submission = False
        self.response_future = asyncio.get_running_loop().create_future()
        try:
            # Response and request observers were registered before the click.
            await self.bounded(button.click(timeout=self.timeout_ms))
            if await self.page.evaluate("window.__captchaDecay.blocked"):
                raise FlowError(outcome("challenge_expired_or_changed", "click_guard_detected_change"))
            if self.blocked_submission:
                raise FlowError(outcome("challenge_expired_or_changed", "submission_gate_rejected"))
            response = await self.bounded(self.response_future)
            raw = await self.bounded(response.body())
            import json
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                payload = None
            result = classify_response(response.status, payload, correlated=(response.request == self.current),
                tracking_code=code, not_found_messages=self.interface.not_found_messages)
            result["evidence"].update(response_sha256=hashlib.sha256(raw).hexdigest(),
                                      http_status=response.status, response_received_at=utc_now())
            return result
        finally:
            self.armed = None
            await self.page.evaluate("() => {window.__captchaDecay.pin=null}")

    async def next_challenge(self, snapshot, count, delay):
        await self.bounded(asyncio.sleep(delay))
        if await self.unchanged(snapshot, count):
            await self.bounded(self.page.locator(self.interface.refresh).click(timeout=self.timeout_ms))
        # A normal frontend refresh after rejection is already a new challenge.
        await self.bounded(self.page.wait_for_function(
            "r => window.__captchaDecay.revision !== r", arg=snapshot["revision"], timeout=self.timeout_ms))


async def execute(recognizer, code, artifacts: Artifacts, *, headed=False, max_attempts=1,
                  timeout_ms=30000, interface=CORREIOS, inspect=False, retry_delay=5,
                  context_setup=None):
    """Test infrastructure alone injects interface/context_setup; no CLI target override.

    Orchestration tests use a simulated recognizer and mark mode=local_fixture.
    """
    started = artifacts.monotonic_started
    final = outcome("site_changed_or_unrecognized", "execution_not_completed")
    browser = context = flow = None
    phase = "browser_start"
    attempt = None
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts deve estar entre 1 e 3.")
    if timeout_ms <= 0 or timeout_ms > 120000:
        raise ValueError("timeout_ms deve estar entre 1 e 120000.")
    if not inspect:
        validate_code(code)
    try:
        artifacts.environment()
        if recognizer:
            artifacts.result["recognizer"] = recognizer.metadata
            artifacts.log("model_loaded", checkpoint_sha256=recognizer.metadata.get("checkpoint_sha256"),
                          checkpoint_epoch=recognizer.metadata.get("checkpoint_epoch"),
                          model_load_seconds=recognizer.metadata.get("model_load_seconds"))
        if (headed and sys.platform.startswith("linux")
                and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
            raise FlowError(outcome("site_changed_or_unrecognized", "headed_requires_display"))
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(headless=not headed, timeout=timeout_ms)
                artifacts.result["chromium"] = browser.version
                context = await browser.new_context(device_scale_factor=1)
                context.set_default_timeout(timeout_ms)
                if context_setup:
                    await context_setup(context)
                page = await context.new_page()
                flow = BrowserFlow(page, interface, timeout_ms)
                await flow.install()
                phase = "navigation"
                if not inspect:
                    artifacts.result["queries_started"] = 1
                navigation_started = time.perf_counter()
                try:
                    await flow.open()
                finally:
                    artifacts.result["navigation_seconds"] = time.perf_counter() - navigation_started
                if inspect:
                    phase = "inspection"
                    png, _, _, captured_at = await flow.capture()
                    artifacts.result["inspection"] = {
                        "selectors_present": True, "image_sha256": hashlib.sha256(png).hexdigest(),
                        "captured_at": captured_at, "dimensions": [215, 80],
                        "submission_performed": False, "result_contract_verified": False}
                    final = outcome("site_changed_or_unrecognized", "inspection_only_no_tracking_query")
                else:
                    for number in range(1, max_attempts + 1):
                        attempt_started = time.perf_counter()
                        attempt = {"attempt": number, "started_at": utc_now(), "submissions": 0}
                        artifacts.attempts.append(attempt)
                        phase = "capture"
                        captured_started = time.perf_counter()
                        try:
                            png, snapshot, count, captured_at = await flow.capture()
                        finally:
                            attempt["capture_seconds"] = time.perf_counter() - captured_started
                        attempt.update(capture_source="browser_response_body", captured_at=captured_at,
                            image_path=interface.image_path,
                            image_sha256=hashlib.sha256(png).hexdigest(), image_revision=snapshot["revision"],
                            image_responses_seen=count, captcha_file=artifacts.save_captcha(number, png))
                        phase = "inference"
                        inference_started = time.perf_counter()
                        try:
                            inferred = await asyncio.to_thread(recognizer.recognize, png)
                        finally:
                            attempt["inference_seconds"] = time.perf_counter() - inference_started
                        prediction = inferred["text"]
                        if not isinstance(prediction, str) or not prediction:
                            raise FlowError(outcome("inference_error", "empty_prediction"))
                        if inferred["image_sha256"] != attempt["image_sha256"]:
                            raise FlowError(outcome("inference_error", "recognizer_image_mismatch"))
                        attempt.update(prediction=prediction, inference_seconds=inferred["inference_seconds"])
                        phase = "submission"
                        submission_started = time.perf_counter()
                        before = flow.submissions
                        try:
                            final = await flow.submit(code, prediction, snapshot, count)
                        finally:
                            attempt.update(submission_seconds=time.perf_counter() - submission_started,
                                           submissions=flow.submissions - before)
                            if attempt["submissions"]:
                                attempt.update(submitted_prediction=prediction, submitted_at=flow.last_submission)
                        attempt.update(final, total_seconds=time.perf_counter() - attempt_started)
                        artifacts.log("attempt_finished", attempt=number, status=final["status"],
                                      submissions=attempt["submissions"])
                        if final["status"] != "captcha_rejected" or number == max_attempts:
                            break
                        phase = "normal_refresh_after_explicit_rejection"
                        await flow.next_challenge(snapshot, count, retry_delay)
            finally:
                try:
                    if context:
                        await asyncio.wait_for(context.close(), timeout=timeout_ms / 1000)
                finally:
                    if browser:
                        await asyncio.wait_for(browser.close(), timeout=timeout_ms / 1000)
                    artifacts.result["browser_closed"] = browser is None or not browser.is_connected()
    except FlowError as error:
        final = error.result
    except (BrowserTimeout, asyncio.TimeoutError):
        final = outcome("network_error", "finite_timeout_no_automatic_resubmission", phase=phase,
                        submission_state="unknown" if flow and flow.submissions else "not_submitted")
    except asyncio.CancelledError:
        final = outcome("network_error", "execution_interrupted_no_resubmission", phase=phase)
    except (BrowserError, OSError, ValueError, RuntimeError, KeyError):
        # Exception strings can contain URLs, form values, addresses or tokens.
        final = outcome("inference_error" if phase == "inference" else
                        "site_changed_or_unrecognized", "handled_error", phase=phase)
    finally:
        if flow and flow.fatal:
            final = flow.fatal
            if attempt:
                attempt.update(final)
        if recognizer:
            try:
                recognizer.verify_preservation()
                artifacts.result["model_preservation_verified"] = True
            except (RuntimeError, OSError):
                final = outcome("inference_error", "model_preservation_failed")
        if attempt and "status" not in attempt:
            attempt.update(final, total_seconds=time.perf_counter() - attempt_started)
            artifacts.log("attempt_finished", attempt=attempt["attempt"], status=final["status"],
                          submissions=attempt["submissions"])
        if attempt and flow:
            attempt["image_requests_seen_at_end"] = len(flow.images)
            attempt["challenge_change_detected"] = final["status"] == "challenge_expired_or_changed"
        artifacts.result["submissions"] = flow.submissions if flow else 0
        final_result = artifacts.finish(final, time.perf_counter() - started)
    return final_result
