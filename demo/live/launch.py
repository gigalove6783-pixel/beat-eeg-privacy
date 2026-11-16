"""Stage launcher for BEAT Audit Live; ``start_demo.bat`` runs it with the project's ``.venv``.

It checks the prepared samples, starts the server of ``server.py`` in this process on the first
free port from 8737, waits until the page and the audit tool answer, and opens the page in
Chrome as a full-screen app window (the default browser if Chrome is absent). Closing the
console window stops the server and its workers. A second launch while one is running opens the
running page instead of starting another server.
"""

import ctypes
import http.client
import json
import msvcrt
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
import winreg
from pathlib import Path

import server as live

PORTS = range(live.PORT, live.PORT + 20)
LOCK = Path(tempfile.gettempdir()) / "beat-audit-live.lock"
TOOL_TIMEOUT = 120
ROLE_TITLE = {"A": "A 특징 벡터", "B": "B 명령", "C": "C 대조군"}


def say(text=""):
    print(text, flush=True)


def take_lock():
    """An exclusive lock on one byte, released by Windows when this process ends."""
    handle = open(LOCK, "a+")
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


def port_taken(port):
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        return True
    finally:
        probe.close()
    return False


def beat_info(port):
    """The /api/info of a BEAT Audit Live server on this port, or None."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        connection.request("GET", "/api/info")
        response = connection.getresponse()
        body = response.read()
        if response.status != 200 or not response.getheader("Server", "").startswith("BEATAuditLive"):
            return None
        return json.loads(body)
    except (OSError, ValueError):
        return None
    finally:
        connection.close()


def running_server():
    """Port of a BEAT Audit Live server already serving on one of our ports, or None."""
    for port in PORTS:
        if port_taken(port) and beat_info(port) is not None:
            return port
    return None


def find_chrome():
    for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base and (Path(base) / "Google/Chrome/Application/chrome.exe").is_file():
            return Path(base) / "Google/Chrome/Application/chrome.exe"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            with winreg.OpenKey(hive, r"Software\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe") as key:
                path = Path(winreg.QueryValue(key, None))
        except OSError:
            continue
        if path.is_file():
            return path
    return None


def open_page(url):
    chrome = find_chrome()
    if chrome is None:
        say("Chrome을 찾지 못해 기본 브라우저로 엽니다. 전체 화면은 F11을 누르세요.")
        webbrowser.open(url)
        return
    # A profile of its own makes Chrome open a new full-screen app window even when Chrome is
    # already running with the speaker's usual profile.
    profile = Path(os.environ["LOCALAPPDATA"]) / "BEATAuditLive" / "chrome-profile"
    subprocess.Popen([str(chrome), f"--app={url}", "--start-fullscreen", f"--user-data-dir={profile}",
                      "--no-first-run", "--no-default-browser-check"],
                     creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    say("Chrome 전체 화면 창으로 엽니다. 창을 닫으려면 Alt+F4.")


def bring_page_forward():
    """Bring an open BEAT Audit Live page window to the front; False if none is open."""
    user32 = ctypes.windll.user32
    title = "BEAT Audit Live"
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(hwnd, _):
        buffer = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, buffer, 256)
        if user32.IsWindowVisible(hwnd) and buffer.value.startswith(title + " —"):
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(visit, 0)
    if not found:
        return False
    if user32.IsIconic(found[0]):
        user32.ShowWindow(found[0], 9)  # SW_RESTORE
    user32.SetForegroundWindow(found[0])
    return True


def reuse(port):
    url = f"http://127.0.0.1:{port}/"
    say(f"이미 실행 중입니다: {url}")
    if bring_page_forward():
        say("서버를 하나 더 켜지 않고, 열려 있는 페이지를 앞으로 가져왔습니다.")
    else:
        say("서버를 하나 더 켜지 않고, 실행 중인 서버의 페이지를 엽니다.")
        open_page(url)
    say("이 창은 몇 초 뒤 닫힙니다.")
    time.sleep(6)
    return 0


def check_samples():
    name, _, missing = live.prepared_set()
    if name is None:
        say("오류: 준비된 표본 파일이 없습니다.")
        for set_name, folder in (("yang2025", "samples\\yang2025"), ("synthetic", "samples\\synthetic")):
            say(f"  {live.SET_LABELS[set_name]} ({folder}): 없음 - "
                + ", ".join(ROLE_TITLE[r] for r in missing[set_name]))
        say("실제 코호트 표본은 .venv\\Scripts\\python scripts\\make_live_samples.py 로 만듭니다.")
        return None
    if name == "synthetic":
        say("주의: 실제 코호트 표본이 없어 합성 표본으로 시연합니다. 없는 파일: "
            + ", ".join(ROLE_TITLE[r] for r in missing["yang2025"]))
    else:
        say(f"표본: {live.SET_LABELS[name]} 3개 확인")
    return name


def main():
    say("BEAT Audit Live 시작")
    name = check_samples()
    if name is None:
        return 1
    lock = take_lock()
    if lock is None:
        # Another launcher holds the lock; its server may still be binding.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            port = running_server()
            if port is not None:
                return reuse(port)
            time.sleep(0.5)
        say("오류: 다른 BEAT Audit Live 실행 창이 열려 있지만 응답하지 않습니다. 그 창을 닫고 다시 실행하세요.")
        return 1
    port = running_server()  # e.g. a server started by hand
    if port is not None:
        return reuse(port)

    workdir = tempfile.mkdtemp(prefix="beat-audit-live-")
    server = app = None
    for port in PORTS:
        try:
            server, app = live.make_server(port, workdir)
            break
        except OSError:
            say(f"포트 {port}은(는) 다른 프로그램이 쓰고 있어 다음 포트를 찾습니다.")
    if server is None:
        shutil.rmtree(workdir, ignore_errors=True)
        say(f"오류: 포트 {PORTS[0]}-{PORTS[-1]}을 모두 다른 프로그램이 쓰고 있습니다.")
        return 1

    stopped = threading.Event()

    def stop():
        if stopped.is_set():
            return
        stopped.set()
        server.shutdown()
        server.server_close()
        app.close()
        shutil.rmtree(workdir, ignore_errors=True)

    # Closing the console window (or logging off) ends this process within seconds; stop the
    # server and its workers first so no audit process or uploaded copy is left behind.
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)
    def on_console_event(event):
        if event in (2, 5, 6):  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
            stop()
            return True
        return False

    ctypes.windll.kernel32.SetConsoleCtrlHandler(on_console_event, True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    try:
        say(f"서버: {url}  (이 컴퓨터에서만 열림)")
        say("감사 도구를 불러오는 중 …")
        deadline = time.monotonic() + TOOL_TIMEOUT
        while not (app.tool_ready() and beat_info(port)):
            if app.spare.process.poll() is not None:
                say("오류: 감사 도구(beat_audit)를 불러오지 못했습니다. .venv에 설치되어 있는지 확인하세요.")
                return 1
            if time.monotonic() > deadline:
                say(f"오류: {TOOL_TIMEOUT}초 안에 감사 도구가 준비되지 않았습니다.")
                return 1
            time.sleep(0.25)
        say(f"준비 완료 · {live.SET_LABELS[name]} 3개")
        open_page(url)
        say()
        say("이 창을 닫으면 서버가 꺼집니다. 발표가 끝날 때까지 열어 두세요.")
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    finally:
        stop()
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
