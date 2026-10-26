"""Exercise the real LangChain HTTP/parsing path against a local protocol stub."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from test_workflow import fixture_repository

from papertrail.workflow import start_job


def test_langchain_recovery_over_openai_compatible_http(tmp_path, monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            context = json.loads(payload["messages"][-1]["content"])
            decision = {"failure_type": "missing_data", "updates": {"train_path": "data.json"},
                        "reason": "Local protocol stub decision; not a real model evaluation.",
                        "citations": [context["evidence"][0]["citation"]]}
            response = {"id": "chatcmpl-local-test", "object": "chat.completion",
                        "created": int(time.time()), "model": "local-protocol-stub",
                        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                            "role": "assistant", "content": None, "tool_calls": [{
                                "id": "repair1", "type": "function", "function": {
                                    "name": payload["tools"][0]["function"]["name"],
                                    "arguments": json.dumps(decision)}}]}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}}
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_API_KEY", "local-test-placeholder")
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("PAPERTRAIL_MODEL", "local-protocol-stub")
    source, store = fixture_repository(tmp_path)
    try:
        state = start_job(source, store=store, backend="local", policy="llm")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert state["status"] == "succeeded"
    assert state["model_calls"] == 1
    assert len(requests) == 1 and requests[0]["tools"]
    assert state["usage"][0]["total_tokens"] == 50
