"""BEAT Audit Live: a local, offline web app that runs ``beat_audit`` on a dropped CSV.

Start from the repository root:

    .venv\\Scripts\\python demo/live/server.py

It listens on 127.0.0.1 only and prints the address. Uploaded files are treated as untrusted:
size cap, CSV only, the tool's column schema, then the tool's own validation in a separate
worker process (``worker.py``). Errors reach the page as Korean messages, never a traceback.

Stored runs for the stage fallback are produced by the same worker:

    .venv\\Scripts\\python demo/live/server.py --store demo/live/samples/synthetic/*.csv
"""

import argparse
import csv
import glob
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATIC = HERE / "static"
STORED = HERE / "stored"
SAMPLES = HERE / "samples"
WORKER = HERE / "worker.py"

PORT = 8737
MAX_BYTES = 32 * 1024 * 1024
MAX_ROWS = 200_000
MAX_DIMENSION = 1024
MIN_USERS = 6
RUN_TIMEOUT = 180
BASE_COLUMNS = {"user", "day", "segment", "order", "target", "C", "outputs"}
FEATURE_COLUMN = re.compile(r"outputs_(0|[1-9][0-9]*)")
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/live.js": "live.js",
                "/live.css": "live.css"}


class InputError(ValueError):
    """An upload the server refuses, with a Korean message for the page."""

    def __init__(self, message, status=HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = status


def validate_upload(filename, data):
    """Check an untrusted upload before the tool sees it; return a short description."""
    if not filename.lower().endswith(".csv"):
        raise InputError("CSV 파일(.csv)만 받습니다.", HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
    if len(data) > MAX_BYTES:
        raise InputError(f"파일이 너무 큽니다. 최대 {MAX_BYTES // (1024 * 1024)} MB까지 받습니다.",
                         HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
    if not data.strip():
        raise InputError("빈 파일입니다.")
    if b"\x00" in data:
        raise InputError("텍스트 CSV가 아닙니다. 이진 파일은 받지 않습니다.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError("UTF-8 텍스트로 읽을 수 없습니다. CSV를 UTF-8로 저장해 주세요.") from None
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = [name.strip() for name in next(reader)]
    except (StopIteration, csv.Error):
        raise InputError("CSV 머리글(첫 줄)을 읽을 수 없습니다.") from None
    if len(set(header)) != len(header):
        raise InputError("머리글에 같은 열 이름이 두 번 있습니다.")
    features = sorted(int(FEATURE_COLUMN.fullmatch(h).group(1)) for h in header
                      if FEATURE_COLUMN.fullmatch(h))
    unknown = [h for h in header if h not in BASE_COLUMNS and not FEATURE_COLUMN.fullmatch(h)]
    if unknown:
        shown = ", ".join(unknown[:3]) + (" …" if len(unknown) > 3 else "")
        raise InputError(f"알 수 없는 열이 있습니다: {shown}. 허용되는 열은 user, day, segment, "
                         "order, target, C와 outputs(명령) 또는 outputs_0, outputs_1, …(특징)입니다.")
    for required in ("user", "day"):
        if required not in header:
            raise InputError(f"'{required}' 열이 없습니다. 모든 행에 user(사용자)와 day(기록일)가 "
                             "필요합니다.")
    if "outputs" in header and features:
        raise InputError("outputs와 outputs_0, … 을 함께 쓸 수 없습니다. 명령이면 outputs 하나, "
                         "특징이면 outputs_0부터 번호를 붙인 열만 둡니다.")
    if "outputs" not in header and not features:
        raise InputError("출력 열이 없습니다. 명령은 outputs, 특징은 outputs_0, outputs_1, … 열에 "
                         "적어 주세요.")
    if features and features != list(range(len(features))):
        raise InputError("특징 열 번호가 비어 있습니다. outputs_0부터 빠짐없이 이어져야 합니다.")
    if len(features) > MAX_DIMENSION:
        raise InputError(f"특징이 {len(features)}차원입니다. 최대 {MAX_DIMENSION}차원까지 받습니다.")
    user_at, day_at = header.index("user"), header.index("day")
    users, days, rows = set(), set(), 0
    try:
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            rows += 1
            if rows > MAX_ROWS:
                raise InputError(f"행이 너무 많습니다. 최대 {MAX_ROWS:,}행까지 받습니다.")
            if len(row) != len(header):
                raise InputError(f"{line}번째 줄의 칸 수({len(row)})가 머리글({len(header)})과 "
                                 "다릅니다.")
            users.add(row[user_at].strip())
            days.add(row[day_at].strip())
    except csv.Error:
        raise InputError("CSV 형식이 깨져 있습니다(따옴표나 줄바꿈을 확인해 주세요).") from None
    if rows == 0:
        raise InputError("머리글만 있고 데이터 행이 없습니다.")
    users.discard("")
    if len(users) < MIN_USERS:
        raise InputError(f"사용자가 {len(users)}명뿐입니다. 공격자를 학습할 사용자와 평가할 사용자를 "
                         f"나눠야 하므로 {MIN_USERS}명 이상이 필요합니다.")
    days.discard("")
    if len(days) < 2:
        raise InputError("기록일이 하루뿐입니다. 같은 날과 다른 날을 비교하려면 기록일이 둘 이상 "
                         "필요합니다.")
    return {"rows": rows, "users": len(users), "days": len(days),
            "kind": "features" if features else "commands"}


def sample_manifest():
    """Prepared sample files by SHA-256 (content, never file name)."""
    entries = {}
    for path in sorted(SAMPLES.glob("*/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest["files"]:
            entries[entry["sha256"]] = {**entry, "set": manifest["set"]}
    return entries


PREPARED_ROLES = ("A", "B", "C")
PREPARED_SETS = ("yang2025", "synthetic")  # real cohort first, tracked synthetic otherwise
SET_LABELS = {"yang2025": "실제 코호트 표본", "synthetic": "합성 표본"}


def _set_files(name):
    """The A, B and C entries of one sample set's manifest, each with its on-disk path."""
    manifest = SAMPLES / name / "manifest.json"
    if not manifest.is_file():
        return {}
    entries = {}
    for entry in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
        # The manifest is ours, but its file name must still be a bare name in that folder.
        if entry.get("role") in PREPARED_ROLES and Path(entry["file"]).name == entry["file"]:
            entries[entry["role"]] = {**entry, "path": SAMPLES / name / entry["file"]}
    return entries


def prepared_set():
    """The active prepared set: real-cohort samples when all three are present, else synthetic.

    Returns (set name or None, {role: entry}, {set name: [missing roles]}).
    """
    missing = {}
    for name in PREPARED_SETS:
        files = _set_files(name)
        missing[name] = [r for r in PREPARED_ROLES if r not in files or not files[r]["path"].is_file()]
        if not missing[name]:
            return name, files, missing
    return None, {}, missing


def prepared_info():
    """What the page needs for its readiness line and run buttons (no paths)."""
    name, files, missing = prepared_set()
    return {"set": name, "label": SET_LABELS.get(name),
            "files": [{"role": r, "title": files[r]["title"], "file": files[r]["file"]}
                      for r in PREPARED_ROLES if r in files],
            "missing": missing}


def stored_run(sha256):
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        return None
    path = STORED / f"{sha256}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


class Worker:
    """One audit process, started ahead of time so the libraries are already imported."""

    def __init__(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        self.process = subprocess.Popen(
            [sys.executable, str(WORKER)], cwd=ROOT, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8")
        self.ready = threading.Event()
        self.events = []
        self.finished = threading.Event()
        self.listeners = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "ready":
                self.ready.set()
                continue
            self.events.append(event)
            if event.get("type") in ("result", "error"):
                break
        self.ready.set()
        self.finished.set()

    def start(self, path, filename, seconds_per_output):
        job = {"path": str(path), "filename": filename, "seconds_per_output": seconds_per_output}
        self.process.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def stop(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()


def run_once(path, filename, seconds_per_output=None, timeout=RUN_TIMEOUT):
    """Run one file to completion in a fresh worker (used by --store and the tests)."""
    worker = Worker()
    try:
        worker.ready.wait(timeout)
        worker.start(path, filename, seconds_per_output)
        if not worker.finished.wait(timeout):
            raise TimeoutError(f"{filename}: no result within {timeout} s")
    finally:
        worker.stop()
    final = worker.events[-1] if worker.events else {"type": "error", "message": "worker ended"}
    return worker.events, final


class Run:
    def __init__(self, run_id, filename, sha256, path, worker):
        self.id, self.filename, self.sha256, self.path = run_id, filename, sha256, path
        self.worker = worker
        self.started = time.monotonic()
        self.cancelled = False

    def state(self, since=0):
        events = self.worker.events
        final = events[-1] if events and events[-1].get("type") in ("result", "error") else None
        status = "running"
        if self.cancelled:
            status = "cancelled"
        elif final:
            status = "done" if final["type"] == "result" else "error"
        elif self.worker.process.poll() is not None:
            status = "error"
            final = {"type": "error", "message": "감사 프로세스가 예기치 않게 끝났습니다."}
        elif time.monotonic() - self.started > RUN_TIMEOUT:
            self.worker.stop()
            status = "error"
            final = {"type": "error", "message": f"{RUN_TIMEOUT}초 안에 끝나지 않아 중단했습니다."}
        body = {"id": self.id, "status": status, "events": events[since:],
                "next": len(events), "elapsed": time.monotonic() - self.started}
        if status == "done":
            body["summary"] = final["summary"]
        elif status == "error":
            body["error"] = {"message": final["message"], "detail": final.get("detail")}
        return body


class App:
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.lock = threading.Lock()
        self.runs = {}
        self.active = None
        self.spare = Worker()

    def submit(self, filename, data, seconds_per_output):
        summary = validate_upload(filename, data)
        with self.lock:
            if self.active and self.active.state()["status"] == "running":
                raise InputError("앞의 파일을 분석하는 중입니다. 끝난 뒤 다시 놓아 주세요.",
                                 HTTPStatus.CONFLICT)
            run_id = uuid.uuid4().hex[:12]
            path = self.workdir / f"{run_id}.csv"
            path.write_bytes(data)
            sha256 = hashlib.sha256(data).hexdigest()
            worker, self.spare = self.spare, Worker()
            worker.start(path, filename, seconds_per_output)
            run = Run(run_id, filename, sha256, path, worker)
            self.runs[run_id] = self.active = run
        return {"id": run_id, "sha256": sha256, "filename": filename, "precheck": summary,
                "sample": sample_manifest().get(sha256), "stored": stored_run(sha256) is not None}

    def submit_prepared(self, role, seconds_per_output):
        """Run a prepared sample exactly as an upload of that file: read it now, same path."""
        if role not in PREPARED_ROLES:
            raise InputError("준비된 표본은 A, B, C만 있습니다.", HTTPStatus.NOT_FOUND)
        _, files, _ = prepared_set()
        if role not in files:
            raise InputError(f"준비된 표본 {role} 파일이 없습니다.", HTTPStatus.NOT_FOUND)
        entry = files[role]
        return self.submit(entry["file"], entry["path"].read_bytes(), seconds_per_output)

    def tool_ready(self):
        return self.spare.ready.is_set() and self.spare.process.poll() is None

    def cancel(self, run_id):
        run = self.runs.get(run_id)
        if run is None:
            return False
        run.cancelled = True
        run.worker.stop()
        return True

    def close(self):
        self.spare.stop()
        for run in self.runs.values():
            run.worker.stop()


def parse_seconds(value):
    if value in (None, ""):
        return None
    try:
        seconds = float(value)
    except ValueError:
        raise InputError("출력 하나의 시간(초)은 숫자로 적어 주세요.") from None
    if not 0 < seconds <= 3600:
        raise InputError("출력 하나의 시간(초)은 0보다 크고 3600 이하로 적어 주세요.")
    return seconds


class Handler(BaseHTTPRequestHandler):
    server_version = "BEATAuditLive"
    sys_version = ""
    app = None

    def log_message(self, format, *args):  # quiet console for the stage
        pass

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._send(status, {"error": {"message": message}})

    def _local(self):
        """Refuse requests whose Host is not this loopback server (DNS rebinding)."""
        host = (self.headers.get("Host") or "").lower()
        port = self.server.server_address[1]
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._error(HTTPStatus.FORBIDDEN, "이 서버는 이 컴퓨터(127.0.0.1)에서만 엽니다.")
            return False
        origin = self.headers.get("Origin")
        if self.command == "POST" and origin and urlsplit(origin).netloc.lower() != host:
            self._error(HTTPStatus.FORBIDDEN, "다른 사이트의 요청은 받지 않습니다.")
            return False
        return True

    def do_GET(self):
        if not self._local():
            return
        url = urlsplit(self.path)
        if url.path in STATIC_FILES:
            path = STATIC / STATIC_FILES[url.path]
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if kind.startswith("text/") or kind.endswith("javascript"):
                kind += "; charset=utf-8"
            return self._send(HTTPStatus.OK, path.read_bytes(), kind)
        if url.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if url.path == "/api/info":
            return self._send(HTTPStatus.OK, {
                "max_bytes": MAX_BYTES, "min_users": MIN_USERS,
                "stored": sorted(p.stem for p in STORED.glob("*.json")),
                "prepared": prepared_info(), "tool_ready": self.app.tool_ready()})
        match = re.fullmatch(r"/api/runs/([0-9a-f]{12})", url.path)
        if match and match.group(1) in self.app.runs:
            since = parse_qs(url.query).get("since", ["0"])[0]
            since = int(since) if since.isdigit() else 0
            return self._send(HTTPStatus.OK, self.app.runs[match.group(1)].state(since))
        match = re.fullmatch(r"/api/stored/([0-9a-f]{64})", url.path)
        if match:
            stored = stored_run(match.group(1))
            if stored:
                return self._send(HTTPStatus.OK, stored)
            return self._error(HTTPStatus.NOT_FOUND, "이 파일의 저장된 실행이 없습니다.")
        self._error(HTTPStatus.NOT_FOUND, "없는 주소입니다.")

    def do_POST(self):
        if not self._local():
            return
        url = urlsplit(self.path)
        match = re.fullmatch(r"/api/runs/([0-9a-f]{12})/cancel", url.path)
        if match:
            if self.app.cancel(match.group(1)):
                return self._send(HTTPStatus.OK, {"cancelled": True})
            return self._error(HTTPStatus.NOT_FOUND, "없는 실행입니다.")
        if url.path.startswith("/api/samples/"):
            # The client names a prepared file by role only; no body, name or path is read.
            role = url.path[len("/api/samples/"):]
            self.close_connection = True  # any body the client sent is left unread
            try:
                seconds = parse_seconds(self.headers.get("X-Seconds-Per-Output"))
                body = self.app.submit_prepared(role, seconds)
            except InputError as exc:
                return self._error(exc.status, str(exc))
            return self._send(HTTPStatus.ACCEPTED, body)
        if url.path != "/api/runs":
            return self._error(HTTPStatus.NOT_FOUND, "없는 주소입니다.")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._error(HTTPStatus.LENGTH_REQUIRED, "파일 크기를 알 수 없습니다.")
        if length > MAX_BYTES:
            self.close_connection = True
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               f"파일이 너무 큽니다. 최대 {MAX_BYTES // (1024 * 1024)} MB까지 받습니다.")
        data = self.rfile.read(length) if length > 0 else b""
        filename = unquote(self.headers.get("X-File-Name", "")).replace("\\", "/").split("/")[-1]
        try:
            seconds = parse_seconds(self.headers.get("X-Seconds-Per-Output"))
            body = self.app.submit(filename[:200], data, seconds)
        except InputError as exc:
            return self._error(exc.status, str(exc))
        self._send(HTTPStatus.ACCEPTED, body)


class Server(ThreadingHTTPServer):
    # Address reuse would let Windows bind a port another program already serves.
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address):
        # A browser that closes the connection mid-request is expected on stage; one line,
        # not a traceback. Any other error keeps the standard report.
        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError,
                                          BrokenPipeError)):
            print(f"연결이 끊김: {client_address[0]}:{client_address[1]} ({sys.exc_info()[0].__name__})",
                  file=sys.stderr, flush=True)
            return
        super().handle_error(request, client_address)


def make_server(port=PORT, workdir=None):
    server = Server(("127.0.0.1", port), Handler)
    app = App(workdir)
    server.RequestHandlerClass = type("BoundHandler", (Handler,), {"app": app})
    return server, app


def store(patterns):
    """Write stored runs for the fallback through the same worker as live runs."""
    STORED.mkdir(exist_ok=True)
    paths = [Path(p) for pattern in patterns for p in sorted(glob.glob(pattern))]
    if not paths:
        raise SystemExit("--store: no file matches")
    for path in paths:
        data = path.read_bytes()
        validate_upload(path.name, data)
        events, final = run_once(path.resolve(), path.name)
        if final["type"] != "result":
            raise SystemExit(f"{path}: {final.get('message')} {final.get('detail') or ''}")
        sha256 = hashlib.sha256(data).hexdigest()
        record = {"stored": True,
                  "stored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "sha256": sha256, "summary": final["summary"],
                  "points": [e for e in events if e.get("phase") == "point"]}
        (STORED / f"{sha256}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"stored {path.name}: {final['summary']['elapsed_seconds']:.1f} s -> "
              f"{(STORED / f'{sha256}.json').relative_to(ROOT)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="BEAT Audit Live (local, offline)")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--store", nargs="+", metavar="CSV",
                        help="write stored runs of these files for the stage fallback, then exit")
    args = parser.parse_args(argv)
    if args.store:
        return store(args.store)
    workdir = tempfile.mkdtemp(prefix="beat-audit-live-")
    try:
        server, app = make_server(args.port, workdir)
    except OSError:
        shutil.rmtree(workdir, ignore_errors=True)
        raise SystemExit(f"포트 {args.port}을(를) 다른 프로그램이 쓰고 있습니다. "
                         f"--port 8740 처럼 다른 포트를 지정해 주세요.") from None
    print(f"BEAT Audit Live: http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C로 종료)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
