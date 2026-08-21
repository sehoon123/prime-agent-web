"""Offline tests for Gemini Files API uploads and the oversized-PDF path.

The protocol is exercised against a mock transport that implements the documented
resumable flow, including the `404` behaviour of gateways that proxy only
`generateContent`.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Callable, Optional
from unittest import mock

import httpx

from webfetch import _files, _gemini
from webfetch._files import FilesApiUnsupported, upload_base

from .test_webfetch_gemini import CORP, FakeEndpoint, answer_payload, client_for, patch_endpoints

UPLOAD_SESSION = "https://gw.example.com/session/abc"


class UploadBaseTest(unittest.TestCase):
    def test_version_prefix_is_moved(self) -> None:
        self.assertEqual(
            upload_base("https://generativelanguage.googleapis.com/v1beta"),
            "https://generativelanguage.googleapis.com/upload/v1beta",
        )
        self.assertEqual(
            upload_base("https://gw.example.com/ica/v1beta/"),
            "https://gw.example.com/ica/upload/v1beta",
        )

    def test_nested_path_keeps_its_prefix(self) -> None:
        self.assertEqual(
            upload_base("https://api.example.com/ica/v1beta"),
            "https://api.example.com/ica/upload/v1beta",
        )

    def test_base_without_a_path_never_mangles_the_host(self) -> None:
        # Splitting the raw string would yield https://upload/example.com here.
        self.assertEqual(upload_base("https://example.com"), "https://example.com/upload")
        self.assertEqual(upload_base("https://example.com/"), "https://example.com/upload")

    def test_port_and_scheme_survive(self) -> None:
        self.assertEqual(upload_base("http://gw:8080/a/v1"), "http://gw:8080/a/upload/v1")


def resumable_handler(
    *,
    state: str = "ACTIVE",
    start_status: int = 200,
    give_session: bool = True,
    seen: Optional[list[str]] = None,
    poll_states: Optional[list[str]] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """A mock transport implementing the resumable upload protocol."""
    polls = list(poll_states or [])

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if seen is not None:
            seen.append(f"{request.method} {url}")

        if url.endswith("/upload/v1beta/files") and request.method == "POST":
            if start_status != 200:
                return httpx.Response(start_status, text="no")
            session_url = str(request.url.copy_with(path="/session/abc", query=None))
            headers = {"x-goog-upload-url": session_url} if give_session else {}
            return httpx.Response(200, json={}, headers=headers)

        if request.url.path == "/session/abc":
            return httpx.Response(
                200, json={"file": {"uri": "https://files.example.com/f/1", "name": "files/1", "state": state}}
            )

        if "/files/1" in url and request.method == "GET":
            next_state = polls.pop(0) if polls else "ACTIVE"
            return httpx.Response(200, json={"file": {"name": "files/1", "state": next_state}})

        if "/files/1" in url and request.method == "DELETE":
            return httpx.Response(200, json={})

        if "generateContent" in url:
            return httpx.Response(200, json=answer_payload("# Transcribed large document"))

        return httpx.Response(404, text="unexpected")

    return handler


class UploadProtocolTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_then_finalize(self) -> None:
        seen: list[str] = []
        async with client_for(resumable_handler(seen=seen)) as client:
            uploaded = await _files.upload(
                client, "https://gw.example.com/v1beta", "k", b"payload", "application/pdf"
            )
        self.assertEqual(uploaded.uri, "https://files.example.com/f/1")
        self.assertEqual(uploaded.name, "files/1")
        self.assertEqual(
            seen,
            ["POST https://gw.example.com/upload/v1beta/files", f"POST {UPLOAD_SESSION}"],
        )

    async def test_start_sends_protocol_headers(self) -> None:
        captured: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/files") and request.method == "POST":
                captured.update(dict(request.headers))
                captured["body"] = json.loads(request.content)
                return httpx.Response(200, json={}, headers={"x-goog-upload-url": UPLOAD_SESSION})
            return httpx.Response(
                200, json={"file": {"uri": "u", "name": "files/1", "state": "ACTIVE"}}
            )

        async with client_for(handler) as client:
            await _files.upload(
                client,
                "https://gw.example.com/v1beta",
                "k",
                b"12345",
                "application/pdf",
                display_name="probe",
            )
        self.assertEqual(captured["x-goog-upload-protocol"], "resumable")
        self.assertEqual(captured["x-goog-upload-command"], "start")
        self.assertEqual(captured["x-goog-upload-header-content-length"], "5")
        self.assertEqual(captured["x-goog-upload-header-content-type"], "application/pdf")
        self.assertEqual(captured["body"], {"file": {"display_name": "probe"}})

    async def test_processing_is_polled_until_active(self) -> None:
        seen: list[str] = []
        handler = resumable_handler(state="PROCESSING", poll_states=["PROCESSING", "ACTIVE"], seen=seen)
        with mock.patch.object(_files, "POLL_INITIAL_DELAY", 0.0), mock.patch.object(
            _files, "POLL_MAX_DELAY", 0.0
        ):
            async with client_for(handler) as client:
                uploaded = await _files.upload(
                    client, "https://gw.example.com/v1beta", "k", b"payload", "application/pdf"
                )
        self.assertEqual(uploaded.name, "files/1")
        self.assertEqual(sum(1 for entry in seen if entry.startswith("GET")), 2)

    async def test_failed_processing_raises_and_deletes(self) -> None:
        seen: list[str] = []
        handler = resumable_handler(
            state="PROCESSING", poll_states=["FAILED"], seen=seen
        )
        with mock.patch.object(_files, "POLL_INITIAL_DELAY", 0.0):
            async with client_for(handler) as client:
                with self.assertRaises(RuntimeError) as ctx:
                    await _files.upload(
                        client, "https://gw.example.com/v1beta", "k", b"payload", "application/pdf"
                    )
        self.assertIn("failed to process", str(ctx.exception))
        self.assertTrue(any(entry.startswith("DELETE") for entry in seen))

    async def test_finalize_never_forwards_key_or_crosses_origin(self) -> None:
        finalize_headers: dict[str, str] = {}

        def safe_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/files"):
                return httpx.Response(
                    200,
                    headers={"x-goog-upload-url": UPLOAD_SESSION},
                )
            finalize_headers.update(dict(request.headers))
            return httpx.Response(
                200,
                json={"file": {"uri": "u", "name": "files/1", "state": "ACTIVE"}},
            )

        async with client_for(safe_handler) as client:
            await _files.upload(
                client,
                "https://gw.example.com/v1beta",
                "TOPSECRET-123456789",
                b"payload",
                "application/pdf",
            )
        self.assertNotIn("x-goog-api-key", finalize_headers)
        self.assertEqual(
            _files._validated_session_url(
                "https://gw.example.com/v1beta", "https://gw.example.com:443/session"
            ),
            "https://gw.example.com:443/session",
        )
        with self.assertRaises(RuntimeError):
            _files._validated_session_url(
                "https://gw.example.com/v1beta", "https://user:pass@gw.example.com/session"
            )

        requested: list[str] = []

        def hostile_handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            return httpx.Response(
                200,
                headers={
                    "x-goog-upload-url": "http://169.254.169.254/private-upload"
                },
            )

        async with client_for(hostile_handler) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _files.upload(
                    client,
                    "https://gw.example.com/v1beta",
                    "TOPSECRET-123456789",
                    b"payload",
                    "application/pdf",
                )
        self.assertIn("cross-origin", str(ctx.exception))
        self.assertEqual(requested, ["https://gw.example.com/upload/v1beta/files"])

    async def test_gateway_404_is_unsupported_not_a_failure(self) -> None:
        async with client_for(resumable_handler(start_status=404)) as client:
            with self.assertRaises(FilesApiUnsupported) as ctx:
                await _files.upload(
                    client, "https://gw.example.com/v1beta", "k", b"payload", "application/pdf"
                )
        self.assertIn("does not expose the Gemini Files API", str(ctx.exception))

    async def test_missing_upload_url_is_unsupported(self) -> None:
        async with client_for(resumable_handler(give_session=False)) as client:
            with self.assertRaises(FilesApiUnsupported):
                await _files.upload(
                    client, "https://gw.example.com/v1beta", "k", b"payload", "application/pdf"
                )

    async def test_delete_is_best_effort(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        async with client_for(handler) as client:
            await _files.delete(client, "https://gw.example.com/v1beta", "k", "files/1")  # no raise
            await _files.delete(client, "https://gw.example.com/v1beta", "k", "")


class GenerateWithUploadTest(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_then_asks_then_deletes(self) -> None:
        patch_endpoints(self, CORP)
        seen: list[str] = []
        async with client_for(resumable_handler(seen=seen)) as client:
            answer = await _gemini.generate_with_upload(
                client, b"payload", "application/pdf", "Transcribe it."
            )
        self.assertEqual(answer.text, "# Transcribed large document")
        self.assertTrue(any("generateContent" in entry for entry in seen))
        self.assertTrue(any(entry.startswith("DELETE") for entry in seen))

    async def test_file_reference_is_used_in_the_request(self) -> None:
        patch_endpoints(self, CORP)
        bodies: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "generateContent" in str(request.url):
                bodies.append(json.loads(request.content))
                return httpx.Response(200, json=answer_payload("ok"))
            return resumable_handler()(request)

        async with client_for(handler) as client:
            await _gemini.generate_with_upload(client, b"payload", "application/pdf", "Transcribe it.")

        parts = bodies[0]["contents"][0]["parts"]
        self.assertEqual(
            parts[0],
            {"fileData": {"mimeType": "application/pdf", "fileUri": "https://files.example.com/f/1"}},
        )
        self.assertEqual(parts[1]["text"], "Transcribe it.")

    async def test_endpoint_without_files_api_is_skipped(self) -> None:
        gateway = FakeEndpoint("gateway", "https://gw.example.com/v1beta", ("m",), ("k1", "k2"))
        studio = FakeEndpoint("studio", "https://studio.example.com/v1beta", ("m",), ("k3",))
        patch_endpoints(self, gateway, studio)
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(f"{request.method} {request.url}")
            if request.url.host == "gw.example.com" and str(request.url).endswith("/files"):
                return httpx.Response(404, text="Not Found")
            return resumable_handler()(request)

        async with client_for(handler) as client:
            answer = await _gemini.generate_with_upload(
                client, b"payload", "application/pdf", "Transcribe it."
            )
        self.assertEqual(answer.detail, "studio/m")
        # Only one attempt on the gateway: a missing Files API is per endpoint, not per key.
        self.assertEqual(sum(1 for entry in seen if "gw.example.com" in entry), 1)

    async def test_no_endpoint_supports_uploads(self) -> None:
        patch_endpoints(self, CORP)
        async with client_for(resumable_handler(start_status=404)) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _gemini.generate_with_upload(
                    client, b"payload", "application/pdf", "Transcribe it."
                )
        self.assertIn("Files API", str(ctx.exception))

    async def test_unavailable_without_endpoints(self) -> None:
        patch_endpoints(self)
        async with client_for(resumable_handler()) as client:
            with self.assertRaises(_gemini.GeminiUnavailable):
                await _gemini.generate_with_upload(client, b"x", "application/pdf", "p")


class OversizedPdfRoutingTest(unittest.IsolatedAsyncioTestCase):
    def test_inline_raw_limit_accounts_for_base64_expansion(self) -> None:
        encoded_size = 4 * ((_gemini.MAX_INLINE_BYTES + 2) // 3)
        self.assertLess(encoded_size, 20 * 1024 * 1024)

    def oversized(self) -> bytes:
        return b"%PDF-" + b"x" * _gemini.MAX_INLINE_BYTES

    async def test_large_pdf_goes_through_the_upload_path(self) -> None:
        patch_endpoints(self, CORP)
        seen: list[str] = []
        async with client_for(resumable_handler(seen=seen)) as client:
            answer = await _gemini.read_pdf(client, self.oversized())
        self.assertEqual(answer.text, "# Transcribed large document")
        self.assertTrue(any("/upload/v1beta/files" in entry for entry in seen))

    async def test_small_pdf_stays_inline(self) -> None:
        patch_endpoints(self, CORP)
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json=answer_payload("inline result"))

        async with client_for(handler) as client:
            answer = await _gemini.read_pdf(client, b"%PDF-small")
        self.assertEqual(answer.text, "inline result")
        self.assertTrue(all("generateContent" in entry for entry in seen))

    async def test_upload_failure_message_is_actionable(self) -> None:
        patch_endpoints(self, CORP)
        async with client_for(resumable_handler(start_status=404)) as client:
            with self.assertRaises(RuntimeError) as ctx:
                await _gemini.read_pdf(client, self.oversized())
        message = str(ctx.exception)
        self.assertIn("inline limit", message)
        self.assertIn("max_pages", message)
        self.assertIn("gemini=False", message)


if __name__ == "__main__":
    unittest.main()
