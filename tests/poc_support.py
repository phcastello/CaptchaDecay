"""Loopback-only browser fixture; OCR is simulated, answer fixed independently."""
import hashlib
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
import uuid

from PIL import Image

from poc.correios import Interface

FIXTURE_CODE = 'ZZ000000001ZZ'  # Fictional, never used outside the loopback server.
FIXED_ANSWER = 'ab2l'


class SimulatedRecognizer:
    metadata = {'checkpoint': 'SIMULATED', 'device': 'cpu', 'model_load_seconds': 0}
    def __init__(self, prediction=FIXED_ANSWER):
        self.prediction = prediction
        self.images = []
    def recognize(self, png):
        self.images.append(png)
        return {'text': self.prediction, 'image_sha256': hashlib.sha256(png).hexdigest(),
                'inference_seconds': .001}
    def verify_preservation(self):
        pass


class Fixture:
    def __init__(self, scenario='success'):
        self.scenario = scenario
        self.sessions = {}
        self.submissions = []
        self.image_gets = 0
        self.paths = []
        self.release = threading.Event()
        fixture = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def reply(self, status, data, mime='text/html', headers=None):
                self.send_response(status)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            def do_GET(self):
                path = urlsplit(self.path).path
                fixture.paths.append(path)  # Intentionally no query logging, even in fixtures.
                cookie = SimpleCookie(self.headers.get('Cookie', ''))
                sid = cookie['fixture_session'].value if 'fixture_session' in cookie else ''
                if path == '/app/index.php':
                    if fixture.scenario == 'initial_blocked':
                        self.reply(403, b'Access denied')
                        return
                    if fixture.scenario == 'initial_rate':
                        self.reply(429, b'Too many requests', headers={'Retry-After':'120'})
                        return
                    sid = uuid.uuid4().hex
                    fixture.sessions[sid] = {'images': 0}
                    html = Path('tests/fixtures/poc/form.html').read_text().replace('__SCENARIO__', fixture.scenario)
                    if fixture.scenario == 'missing_selector':
                        html = html.replace('id="b-pesquisar"', 'id="new-button"')
                    if fixture.scenario == 'no_captcha':
                        html = html.replace('id="captcha_image"', 'id="absent-image"')
                    if fixture.scenario == 'external':
                        html += '<img src="https://external.invalid/never-access">'
                    self.reply(200, html.encode(), headers={'Set-Cookie':f'fixture_session={sid}; HttpOnly; SameSite=Lax; Path=/'})
                elif path == '/core/securimage/securimage_show.php':
                    if sid not in fixture.sessions:
                        self.reply(403, b'Wrong session')
                        return
                    fixture.image_gets += 1
                    fixture.sessions[sid]['images'] += 1
                    color = 40 + fixture.sessions[sid]['images'] * 30
                    size = (214, 80) if fixture.scenario == 'wrong_dimensions' else (215, 80)
                    buffer = io.BytesIO()
                    Image.new('RGB', size, (color, 50, 80)).save(buffer, format='PNG')
                    headers = {}
                    if fixture.scenario == 'wrong_session':
                        # Presented bytes belong to A; a different session B has no challenge.
                        other = uuid.uuid4().hex
                        fixture.sessions[other] = {'images': 0}
                        headers['Set-Cookie'] = f'fixture_session={other}; HttpOnly; Path=/'
                    self.reply(200, buffer.getvalue(), 'image/png', headers)
                elif path == '/app/resultado.php':
                    query = parse_qs(urlsplit(self.path).query)
                    fixture.submissions.append({'sid': sid, 'query':query,
                        'image_gets':fixture.image_gets, 'session_images':fixture.sessions.get(sid, {}).get('images', 0)})
                    status = 200
                    if fixture.scenario == 'timeout':
                        fixture.release.wait(timeout=10)
                    if fixture.scenario == 'rate':
                        self.reply(429, b'Limited', headers={'Retry-After':'90'})
                        return
                    if fixture.scenario == 'blocked':
                        self.reply(403, b'Access denied')
                        return
                    if fixture.sessions.get(sid, {}).get('images', 0) == 0:
                        body = {'erro':True, 'mensagem':'Fixture session is invalid'}
                    elif query.get('captcha') != [FIXED_ANSWER] or fixture.scenario == 'rejected' or (
                        fixture.scenario == 'retry_success' and len(fixture.submissions) == 1):
                        body = {'erro':True, 'mensagem':'Captcha inválido'}
                    elif fixture.scenario == 'not_found':
                        body = {'erro':True, 'mensagem':'FIXTURE: objeto não encontrado'}
                    elif fixture.scenario in ('error_200', 'stale'):
                        body = {'erro':True, 'mensagem':'Unrecognized fixture error private@example.invalid'}
                    else:
                        body = {'codObjeto':query.get('objeto', [''])[0], 'eventos':[
                            {'descricao':'Fixture event', 'dtHrCriado':{'date':'2026-10-01 10:20:00'},
                             'endereco':'PRIVATE ADDRESS', 'token':'PRIVATE TOKEN'}], 'destinatario':'PRIVATE PERSON'}
                        if fixture.scenario == 'other_object': body['codObjeto'] = 'ZZ000000002ZZ'
                        if fixture.scenario == 'no_events': body['eventos'] = []
                    self.reply(status, json.dumps(body).encode(), 'application/json')
                else:
                    self.reply(404, b'Not found')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.interface = Interface(url=f'http://127.0.0.1:{self.server.server_port}/app/index.php',
                                   not_found_messages=('FIXTURE: objeto não encontrado',))
        self.external_blocked = []
        self.context = None
        self.browser = None

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        assert not self.thread.is_alive()

    async def setup_context(self, context):
        self.context, self.browser = context, context.browser
        async def only_loopback(route):
            url = urlsplit(route.request.url)
            if url.scheme == 'http' and url.hostname == '127.0.0.1' and url.port == self.server.server_port:
                await route.continue_()
            else:
                self.external_blocked.append(url.hostname)
                await route.abort('blockedbyclient')
        await context.route('**/*', only_loopback)
