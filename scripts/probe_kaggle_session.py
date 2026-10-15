"""Probe Kaggle's documented direct Jupyter connection, then cancel the session."""

import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import struct
import time
import urllib.parse
import uuid


ROOT = Path(__file__).resolve().parents[1]


def session_id_from(value):
    if isinstance(value, dict):
        for name, entry in value.items():
            if name in ("kernelSessionId", "kernel_session_id", "sessionId", "session_id") and str(entry).isdigit():
                return int(entry)
            found = session_id_from(entry)
            if found:
                return found
    elif isinstance(value, list):
        for entry in value:
            found = session_id_from(entry)
            if found:
                return found
    return None


def execute_probe(server_response, source, diagnostic):
    import requests
    import websocket

    tokenized_url = server_response.get("tokenizedJupyterUrl") or server_response["jupyterUrl"]
    parsed = urllib.parse.urlsplit(tokenized_url)
    token = urllib.parse.parse_qs(parsed.query).get("token", [None])[0]
    diagnostic["jupyter_url_has_token"] = bool(token)
    diagnostic["jupyter_url_query_keys"] = sorted(urllib.parse.parse_qs(parsed.query))
    diagnostic["jupyter_server_host"] = parsed.hostname
    base_path = parsed.path
    if base_path.rstrip("/").endswith(("/tree", "/lab")):
        base_path = base_path.rstrip("/").rsplit("/", 1)[0] + "/"
    if not base_path.endswith("/"):
        base_path += "/"
    base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, base_path, "", ""))
    http = requests.Session()
    if token:
        http.headers["Authorization"] = "token " + token
    # Establish any cookies used by the Kaggle Jupyter proxy. This is the
    # scoped Jupyter token returned by Kaggle, never the Kaggle OAuth token.
    diagnostic["stage"] = "Jupyter landing GET"
    landing = http.get(tokenized_url, timeout=30)
    diagnostic["landing_http_status"] = landing.status_code
    # API-only Jupyter deployments may return 404 at the UI landing page.
    # Probe the documented API rather than treating that as an API failure.
    xsrf = http.cookies.get("_xsrf")
    if xsrf:
        http.headers["X-XSRFToken"] = xsrf
    if token:
        # Some notebook proxies consume the URL token before forwarding the
        # request to Jupyter, so preserve the documented token URL parameter.
        http.params = {"token": token}
    diagnostic["stage"] = "Jupyter API status GET"
    ready_deadline = time.monotonic() + 120
    ready_attempts = 0
    while True:
        ready_attempts += 1
        api_status = http.get(urllib.parse.urljoin(base, "api/status"), timeout=15)
        if api_status.status_code not in (404, 502, 503) or time.monotonic() >= ready_deadline:
            break
        time.sleep(3)
    diagnostic["jupyter_ready_attempts"] = ready_attempts
    diagnostic["jupyter_api_http_status"] = api_status.status_code
    api_status.raise_for_status()
    diagnostic["stage"] = "Jupyter kernel POST"
    kernel_response = http.post(urllib.parse.urljoin(base, "api/kernels"), json={"name": "python3"}, timeout=30)
    diagnostic["kernel_http_status"] = kernel_response.status_code
    kernel_response.raise_for_status()
    kernel_id = kernel_response.json()["id"]
    channel = None
    try:
        session = uuid.uuid4().hex
        message_id = uuid.uuid4().hex
        websocket_base = urllib.parse.urlsplit(urllib.parse.urljoin(base, f"api/kernels/{kernel_id}/channels"))
        query = {"session_id": session}
        if token:
            query["token"] = token
        websocket_url = urllib.parse.urlunsplit((
            "wss" if websocket_base.scheme == "https" else "ws", websocket_base.netloc,
            websocket_base.path, urllib.parse.urlencode(query), "",
        ))
        headers = ["Authorization: token " + token] if token else []
        cookie = "; ".join(f"{name}={value}" for name, value in http.cookies.items())
        diagnostic["stage"] = "Jupyter websocket connect"
        channel = websocket.create_connection(websocket_url, header=headers, cookie=cookie,
                                              origin=f"{parsed.scheme}://{parsed.netloc}", timeout=30)
        request = {
            "header": {"msg_id": message_id, "username": "codex", "session": session,
                       "date": dt.datetime.now(dt.timezone.utc).isoformat(), "msg_type": "execute_request", "version": "5.3"},
            "parent_header": {}, "metadata": {}, "channel": "shell",
            "content": {"code": source, "silent": False, "store_history": False,
                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        }
        diagnostic["stage"] = "execute environment probe"
        channel.send(json.dumps(request))
        deadline = time.monotonic() + 120
        output = []
        while time.monotonic() < deadline:
            frame = channel.recv()
            if isinstance(frame, bytes):
                count = struct.unpack("!I", frame[:4])[0]
                offsets = struct.unpack("!" + "I" * count, frame[4:4 + count * 4])
                frame = frame[offsets[0]:offsets[1] if count > 1 else len(frame)].decode()
            message = json.loads(frame)
            if message.get("parent_header", {}).get("msg_id") != message_id:
                continue
            kind = message.get("msg_type", message.get("header", {}).get("msg_type"))
            if kind == "stream":
                output.append(message["content"]["text"])
            elif kind == "error":
                raise RuntimeError("Remote probe failed: " + message["content"].get("ename", "unknown error"))
            elif kind == "status" and message["content"].get("execution_state") == "idle":
                for line in "".join(output).splitlines():
                    if line.startswith("KAGGLE_PREFLIGHT_RESULT="):
                        return json.loads(line.split("=", 1)[1])
                raise RuntimeError("Remote Jupyter probe produced no report")
        raise TimeoutError("Remote Jupyter probe did not complete within 120 seconds")
    finally:
        if channel:
            channel.close()
        http.delete(urllib.parse.urljoin(base, "api/kernels/" + kernel_id), timeout=20)


def main():
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: probe_kaggle_session.py PATH_TO_KERNEL_METADATA")
    metadata_path = Path(sys.argv[1]).resolve()
    metadata = json.loads(metadata_path.read_text())
    folder = metadata_path.parent
    result = {"method": "direct Kaggle Jupyter session", "model_training_started": False}
    session_id = None
    fd = os.open(folder / "private-jupyter-probe.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiCreateKernelSessionRequest, ApiCancelKernelSessionRequest
        api = KaggleApi()
        api.authenticate()
        with api.build_kaggle_client() as client:
            try:
                request = ApiCreateKernelSessionRequest()
                request.slug = metadata["id"]
                request.language = "python"
                request.kernel_type = "notebook"
                request.machine_shape = "NvidiaTeslaT4"
                request.docker_image = metadata["docker_image"]
                request.enable_internet = True
                operation = client.kernels.kernels_api_client.create_kernel_session(request)
                deadline = time.monotonic() + 180
                while True:
                    session_id = session_id or session_id_from(operation.metadata) or session_id_from(operation.response)
                    if operation.done or time.monotonic() >= deadline:
                        break
                    time.sleep(5)
                    operation = client.common.operations_client.get_operation(name=operation.name)
                if not operation.done or operation.error:
                    raise RuntimeError(operation.error.message if operation.error else "Session readiness timed out")
                source = (ROOT / "scripts/kaggle_preflight.py").read_text().rsplit('\nif __name__ == "__main__":', 1)[0]
                source += '\nprint("KAGGLE_PREFLIGHT_RESULT=" + json.dumps(collect_report()))\n'
                result["report"] = execute_probe(operation.response, source, result)
            except Exception as error:
                # Exception text from HTTP/websocket clients can contain signed
                # URLs. Store only the type in the public diagnostic summary.
                result["error_type"] = type(error).__name__
            finally:
                if session_id:
                    cancellation = ApiCancelKernelSessionRequest()
                    cancellation.kernel_session_id = session_id
                    response = client.kernels.kernels_api_client.cancel_kernel_session(cancellation)
                    result["session_cancelled"] = not bool(response.error_message)
    (folder / "direct-session-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
