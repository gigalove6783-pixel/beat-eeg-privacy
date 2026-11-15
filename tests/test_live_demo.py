"""BEAT Audit Live (demo/live): upload validation and one end-to-end run on a synthetic sample."""

import hashlib
import http.client
import importlib.util
import json
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "demo" / "live" / "samples" / "synthetic"
_spec = importlib.util.spec_from_file_location("live_server", ROOT / "demo" / "live" / "server.py")
live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live)

HEADER = "user,day,segment,order,target,C,outputs\n"


def _commands(users=6, days=2, rows=10):
    lines = [f"u{u},{d},1,{o},{o % 2},3,{o % 3}\n"
             for u in range(users) for d in range(1, days + 1) for o in range(rows)]
    return (HEADER + "".join(lines)).encode()


@pytest.mark.parametrize("filename, data, fragment", [
    ("outputs.txt", _commands(), "CSV 파일"),
    ("empty.csv", b"  \n", "빈 파일"),
    ("binary.csv", b"user,day,outputs\x00\n", "이진 파일"),
    ("latin.csv", "user,day,outputs\nü,1,0\n".encode("latin-1"), "UTF-8"),
    ("export.csv", b"subject;session;value\n1;2;3\n", "알 수 없는 열"),
    ("nouser.csv", b"day,outputs\n1,0\n", "'user'"),
    ("mixed.csv", b"user,day,outputs,outputs_0\n", "함께 쓸 수 없습니다"),
    ("gap.csv", b"user,day,outputs_0,outputs_2\n", "outputs_0부터"),
    ("ragged.csv", HEADER.encode() + b"u1,1,1,0,0,3\n", "칸 수"),
    ("header.csv", HEADER.encode(), "데이터 행이 없습니다"),
    ("one_user.csv", _commands(users=1), "1명뿐"),
    ("one_day.csv", _commands(days=1), "하루뿐"),
])
def test_upload_validation_refuses_with_korean_message(filename, data, fragment):
    with pytest.raises(live.InputError) as caught:
        live.validate_upload(filename, data)
    assert fragment in str(caught.value)
    assert "Traceback" not in str(caught.value)


def test_upload_validation_size_cap(monkeypatch):
    monkeypatch.setattr(live, "MAX_BYTES", 100)
    with pytest.raises(live.InputError) as caught:
        live.validate_upload("big.csv", _commands())
    assert caught.value.status == 413 and "너무 큽니다" in str(caught.value)


def test_upload_validation_accepts_tracked_samples():
    kinds = {path.name[0]: live.validate_upload(path.name, path.read_bytes())["kind"]
             for path in sorted(SYNTHETIC.glob("*.csv"))}
    assert kinds == {"A": "features", "B": "commands", "C": "features"}


@pytest.fixture
def served(tmp_path):
    server, app = live.make_server(0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()
    app.close()


def _request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    payload = json.loads(response.read() or b"null")
    connection.close()
    return response.status, payload


def _upload(port, name, data):
    return _request(port, "POST", "/api/runs", data,
                    {"X-File-Name": quote(name), "Content-Type": "text/csv"})


def _finish(port, run_id, timeout=120):
    deadline = time.monotonic() + timeout
    events = []
    while time.monotonic() < deadline:
        status, state = _request(port, "GET", f"/api/runs/{run_id}?since={len(events)}")
        assert status == 200
        events.extend(state["events"])
        if state["status"] != "running":
            return state, events
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_server_refuses_bad_uploads_and_stays_usable(served):
    status, body = _request(served, "POST", "/api/runs", b"x" * 10,
                            {"Content-Length": str(live.MAX_BYTES + 1), "X-File-Name": "big.csv"})
    assert status == 413 and "너무 큽니다" in body["error"]["message"]
    status, body = _upload(served, "one_user.csv", _commands(users=1))
    assert status == 400 and "1명뿐" in body["error"]["message"]
    status, body = _request(served, "GET", "/api/info", headers={"Host": "attacker.example"})
    assert status == 403
    # Passes the server's schema check but not the tool's: a Korean message, never a traceback.
    bad = _commands().replace(b",3,1\n", b",3,1.5\n", 1)
    status, body = _upload(served, "fractional.csv", bad)
    assert status == 202
    state, _ = _finish(served, body["id"])
    assert state["status"] == "error"
    assert "명령" in state["error"]["message"] and "Traceback" not in json.dumps(state)


def test_end_to_end_synthetic_features_run(served):
    path = SYNTHETIC / "A_synthetic_features.csv"
    data = path.read_bytes()
    status, body = _upload(served, path.name, data)
    assert status == 202
    assert body["sha256"] == hashlib.sha256(data).hexdigest()
    assert body["sample"]["role"] == "A" and body["stored"] is True
    state, events = _finish(served, body["id"])
    assert state["status"] == "done", state.get("error")
    summary = state["summary"]
    assert summary["input"]["kind"] == "features" and summary["input"]["users"] == 20
    assert summary["settings"]["permutations"] == 2000
    assert any(w.startswith("Nonstandard inference budget") for w in summary["warnings"])
    plan = next(e for e in events if e.get("phase") == "plan")
    assert sum(e.get("phase") == "point" for e in events) == plan["total"]
    cross = next(c for c in summary["conditions"] if c["relation"] == "cross-day")
    assert cross["cells"][-1]["wording"] == "linkage detected"
    status, stored = _request(served, "GET", f"/api/stored/{body['sha256']}")
    assert status == 200 and stored["stored"] is True
    assert stored["summary"]["input"]["sha256"] == body["sha256"]
