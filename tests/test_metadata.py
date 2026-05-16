from __future__ import annotations

import json
import socket
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nomadic import agent
from nomadic.search.metadata import search_metadata


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _make_handler(*, status: int = 200, body: bytes = b"", content_type: str = "application/json"):
    class Handler(BaseHTTPRequestHandler):
        last_request_path: str | None = None
        last_request_body: bytes | None = None

        def log_message(self, format: str, *args):  # noqa: A003
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("content-length") or 0)
            self.__class__.last_request_path = self.path
            self.__class__.last_request_body = self.rfile.read(length) if length else b""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

    return Handler


class MetadataSearchTests(unittest.TestCase):
    def _serve(self, handler_cls):
        port = _find_free_port()
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return base_url, handler_cls

    def test_search_metadata_happy_path(self):
        response = {
            "results": [
                {
                    "title": "Alpha",
                    "snippet": "First result",
                    "url": "https://example.com/a",
                    "score": 0.92,
                    "metadata": {"kind": "doc"},
                }
            ]
        }
        handler_cls = _make_handler(body=json.dumps(response).encode("utf-8"))
        base_url, handler_cls = self._serve(handler_cls)

        payload = search_metadata("policy", base_url=base_url, limit=3, timeout_seconds=1.0)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["query"], "policy")
        self.assertEqual(payload["limit"], 3)
        self.assertEqual(payload["base_url"], base_url)
        self.assertEqual(payload["endpoint"], f"{base_url}/search")
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["title"], "Alpha")
        self.assertEqual(handler_cls.last_request_path, "/search")
        self.assertEqual(
            json.loads(handler_cls.last_request_body.decode("utf-8")),
            {"query": "policy", "limit": 3},
        )

    def test_search_metadata_non_json_response(self):
        handler_cls = _make_handler(
            content_type="text/plain",
            body=b"not json",
        )
        base_url, _ = self._serve(handler_cls)

        payload = search_metadata("policy", base_url=base_url, timeout_seconds=1.0)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error_type"], "invalid_json")

    def test_search_metadata_http_error(self):
        handler_cls = _make_handler(
            status=503,
            body=json.dumps({"error": "down"}).encode("utf-8"),
        )
        base_url, _ = self._serve(handler_cls)

        payload = search_metadata("policy", base_url=base_url, timeout_seconds=1.0)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "http_error")
        self.assertEqual(payload["http_status"], 503)
        self.assertEqual(payload["raw_response"], {"error": "down"})

    def test_search_metadata_connection_failure(self):
        base_url = f"http://127.0.0.1:{_find_free_port()}"

        payload = search_metadata("policy", base_url=base_url, timeout_seconds=0.5)

        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error_type"], "connection_error")
        self.assertIn("could not reach metadata service", payload["error"])

    def test_registry_includes_metadata_search_tool_and_executes_it(self):
        response = {
            "results": [
                {
                    "title": "Alpha",
                    "snippet": "First result",
                    "url": "https://example.com/a",
                }
            ]
        }
        handler_cls = _make_handler(body=json.dumps(response).encode("utf-8"))
        base_url, handler_cls = self._serve(handler_cls)

        registry = agent.build_default_registry()
        tool_names = [tool.name for tool in registry.tools]
        self.assertIn("metadata_search", tool_names)

        tool_call, parse_error = agent.parse_tool_call(
            '{"tool_call":{"name":"metadata_search","arguments":{"query":"policy","limit":1}}}'
        )
        self.assertIsNone(parse_error)
        self.assertIsNotNone(tool_call)

        runtime = agent.AgentRuntime(
            state=types.SimpleNamespace(
                cfg=types.SimpleNamespace(metadata_base_url=base_url),
            ),
            ui=types.SimpleNamespace(),
        )
        result = registry.execute(tool_call, runtime)

        self.assertEqual(result.tool_name, "metadata_search")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.payload["result_count"], 1)
        self.assertEqual(handler_cls.last_request_path, "/search")


if __name__ == "__main__":
    unittest.main()
