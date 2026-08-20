"""Offline tests for extraction, rendering, and the fetch orchestration."""

from __future__ import annotations

import json
import unittest
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest import mock

import httpx

import webfetch
from webfetch import _extract

HTML_PAGE = """<!doctype html>
<html>
<head><title>  Guide &amp; Reference </title>
<script>var tracking = 1;</script>
<style>body { color: red }</style>
</head>
<body>
  <nav><a href="/nav">navigation noise</a></nav>
  <header>header noise</header>
  <main>
    <h1>Install</h1>
    <p>Run the <a href="https://example.com/cli">CLI</a> first.</p>
    <h2>Options</h2>
    <pre><code>pip install thing</code></pre>
    <ul><li>alpha</li><li>beta</li></ul>
  </main>
  <aside>related links noise</aside>
  <footer>footer noise</footer>
</body>
</html>
"""


def client_for(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5.0)


async def public_resolver(hostname: str) -> Sequence[str]:
    return ["93.184.216.34"]


def make_pdf(page_count: int) -> bytes:
    """Build a real PDF so extraction is exercised end to end.

    Blank pages have no text layer, which is exactly the "scanned document" case
    the extractor must report instead of returning empty text silently.
    """
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)  # already appends to the writer
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class HtmlExtractionTest(unittest.TestCase):
    def test_markdown_keeps_structure_and_drops_boilerplate(self) -> None:
        result = _extract.html_to_markdown(HTML_PAGE)
        self.assertEqual(result.kind, "html")
        self.assertEqual(result.title, "Guide & Reference")
        self.assertIn("# Install", result.text)
        self.assertIn("## Options", result.text)
        self.assertIn("[CLI](https://example.com/cli)", result.text)
        self.assertIn("pip install thing", result.text)
        self.assertIn("- alpha", result.text)
        for noise in ("navigation noise", "header noise", "footer noise", "related links noise", "tracking"):
            self.assertNotIn(noise, result.text)
        self.assertIn("extracted from <main>", result.notes)

    def test_text_mode_strips_markup(self) -> None:
        result = _extract.html_to_text(HTML_PAGE)
        self.assertIn("Install", result.text)
        self.assertNotIn("<h1>", result.text)
        self.assertNotIn("footer noise", result.text)

    def test_title_falls_back_to_h1(self) -> None:
        result = _extract.html_to_markdown("<html><body><main><h1>Only Heading</h1></main></body></html>")
        self.assertEqual(result.title, "Only Heading")

    def test_page_without_main_uses_body(self) -> None:
        html = "<html><body><div><h1>Bare</h1><p>text</p></div></body></html>"
        result = _extract.html_to_markdown(html)
        self.assertIn("# Bare", result.text)
        self.assertEqual(result.notes, [])

    def test_tidy_collapses_blank_lines(self) -> None:
        self.assertEqual(_extract.tidy("a\n\n\n\n\nb   \n"), "a\n\nb")


class SniffingTest(unittest.TestCase):
    def test_pdf_by_magic_bytes_without_content_type(self) -> None:
        self.assertTrue(_extract.looks_like_pdf(b"%PDF-1.7 ...", ""))
        self.assertTrue(_extract.looks_like_pdf(b"anything", "application/pdf"))
        self.assertFalse(_extract.looks_like_pdf(b"<html>", "text/html"))

    def test_html_by_sniffing_when_content_type_missing(self) -> None:
        self.assertTrue(_extract.looks_like_html(b"<!DOCTYPE html><html>", ""))
        self.assertTrue(_extract.looks_like_html(b"", "text/html"))
        self.assertFalse(_extract.looks_like_html(b"{}", "application/json"))

    def test_text_detection_rejects_binary(self) -> None:
        self.assertTrue(_extract.looks_like_text(b'{"a": 1}', "application/json"))
        self.assertTrue(_extract.looks_like_text(b"plain", ""))
        self.assertFalse(_extract.looks_like_text(b"\x00\x01\x02binary", ""))
        self.assertFalse(_extract.looks_like_text(b"anything", "image/png"))


class PdfExtractionTest(unittest.TestCase):
    def test_page_markers_and_page_count(self) -> None:
        pdf = make_pdf(2)
        result = _extract.pdf_to_text(pdf)
        self.assertEqual(result.kind, "pdf")
        self.assertEqual(result.pages, 2)
        self.assertIn("--- page 1 ---", result.text)
        self.assertIn("--- page 2 ---", result.text)

    def test_scanned_pdf_is_reported(self) -> None:
        result = _extract.pdf_to_text(make_pdf(1))
        self.assertTrue(any("scanned" in note for note in result.notes))

    def test_max_pages_limits_and_notes(self) -> None:
        result = _extract.pdf_to_text(make_pdf(3), max_pages=2)
        self.assertIn("--- page 2 ---", result.text)
        self.assertNotIn("--- page 3 ---", result.text)
        self.assertTrue(any("first 2 of 3 pages" in note for note in result.notes))

    def test_corrupt_pdf_raises_runtime_error(self) -> None:
        with self.assertRaises(RuntimeError):
            _extract.pdf_to_text(b"%PDF-1.4 truncated garbage")


class UrlRewriteTest(unittest.TestCase):
    def test_github_blob_becomes_raw(self) -> None:
        url, note = _extract.rewrite_url("https://github.com/o/r/blob/main/src/app.py")
        self.assertEqual(url, "https://raw.githubusercontent.com/o/r/main/src/app.py")
        assert note
        self.assertIn("raw.githubusercontent.com", note)

    def test_repo_root_gets_clone_hint(self) -> None:
        url, note = _extract.rewrite_url("https://github.com/o/r")
        self.assertEqual(url, "https://github.com/o/r")
        assert note
        self.assertIn("git clone", note)

    def test_other_urls_untouched(self) -> None:
        for candidate in (
            "https://example.com/a",
            "https://github.com/o/r/issues/12",
            "https://raw.githubusercontent.com/o/r/main/x",
        ):
            url, note = _extract.rewrite_url(candidate)
            self.assertEqual(url, candidate)
            self.assertIsNone(note)


class BinaryHandlingTest(unittest.TestCase):
    def test_image_is_saved_and_reported(self) -> None:
        result = _extract.save_binary(b"\x89PNG\r\n\x1a\nfake", "https://example.com/a.png", "image/png")
        self.assertEqual(result.kind, "binary")
        assert result.saved_path
        path = Path(result.saved_path)
        self.addCleanup(path.unlink, missing_ok=True)
        self.assertTrue(path.exists())
        self.assertTrue(path.name.endswith(".png"))
        self.assertTrue(any("attach-image" in note for note in result.notes))

    def test_suffix_falls_back_to_url_extension(self) -> None:
        result = _extract.save_binary(b"data", "https://example.com/archive.tar", "")
        assert result.saved_path
        self.addCleanup(Path(result.saved_path).unlink, missing_ok=True)
        self.assertTrue(result.saved_path.endswith(".tar"))


class FetchTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        patcher = mock.patch.dict(
            "os.environ", {"PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS": "0"}, clear=False
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    async def test_html_document_fields(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=HTML_PAGE, headers={"content-type": "text/html; charset=utf-8"})

        document = await webfetch.fetch(
            "https://example.com/guide", resolver=public_resolver, transport=httpx.MockTransport(handler)
        )
        assert isinstance(document, webfetch.Document)
        self.assertTrue(document.ok)
        self.assertEqual(document.kind, "html")
        self.assertEqual(document.title, "Guide & Reference")
        self.assertEqual(document.content_type, "text/html")
        self.assertIn("# Install", document.text)
        self.assertEqual(len(document), len(document.text))

    async def test_raw_mode_returns_body_verbatim(self) -> None:
        payload = {"a": 1, "b": [2, 3]}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

        document = await webfetch.fetch(
            "https://api.example.com/x",
            mode="raw",
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(json.loads(document.text), payload)
        self.assertEqual(document.kind, "text")

    async def test_error_becomes_a_document_not_an_exception(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        document = await webfetch.fetch(
            "https://example.com/x", resolver=public_resolver, transport=httpx.MockTransport(handler)
        )
        assert isinstance(document, webfetch.Document)
        self.assertFalse(document.ok)
        self.assertEqual(document.kind, "error")
        self.assertIn("HTTP 500", document.error or "")

    async def test_unsafe_url_becomes_a_document(self) -> None:
        document = await webfetch.fetch(
            "http://169.254.169.254/latest/meta-data/",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="x")),
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.kind, "error")
        self.assertIn("non-public", document.error or "")

    async def test_truncated_pdf_reports_size_not_a_parse_error(self) -> None:
        big = b"%PDF-1.7" + b"x" * 50_000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=big, headers={"content-type": "application/pdf"})

        document = await webfetch.fetch(
            "https://example.com/paper.pdf",
            max_bytes=2048,
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.kind, "error")
        self.assertIn("max_bytes=", document.error or "")
        self.assertNotIn("EOF marker", document.error or "")

    async def test_multiple_urls_are_fetched_concurrently(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if request.url.path == "/bad":
                return httpx.Response(404, text="missing")
            return httpx.Response(200, text=HTML_PAGE, headers={"content-type": "text/html"})

        documents = await webfetch.fetch(
            ["https://example.com/a", "https://example.com/bad", "https://example.com/c"],
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(documents, list)
        self.assertEqual(len(documents), 3)
        self.assertEqual([document.ok for document in documents], [True, False, True])

    async def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            await webfetch.fetch("https://example.com", mode="pdf")

    async def test_empty_url_list_raises(self) -> None:
        with self.assertRaises(ValueError):
            await webfetch.fetch([])

    async def test_robots_refusal_short_circuits(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            if request.url.path == "/robots.txt":
                return httpx.Response(200, text="User-agent: *\nDisallow: /\n")
            return httpx.Response(200, text=HTML_PAGE, headers={"content-type": "text/html"})

        with mock.patch.dict("os.environ", {"PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS": "1"}, clear=False):
            document = await webfetch.fetch(
                "https://example.com/page", resolver=public_resolver, transport=httpx.MockTransport(handler)
            )
        assert isinstance(document, webfetch.Document)
        self.assertEqual(document.kind, "error")
        self.assertIn("robots.txt", document.error or "")
        self.assertEqual(requested, ["https://example.com/robots.txt"])

    async def test_respect_robots_false_skips_the_check(self) -> None:
        requested: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(str(request.url))
            return httpx.Response(200, text=HTML_PAGE, headers={"content-type": "text/html"})

        document = await webfetch.fetch(
            "https://example.com/page",
            respect_robots=False,
            resolver=public_resolver,
            transport=httpx.MockTransport(handler),
        )
        assert isinstance(document, webfetch.Document)
        self.assertTrue(document.ok)
        self.assertEqual(requested, ["https://example.com/page"])


class RenderTest(unittest.IsolatedAsyncioTestCase):
    """Rendering is a pure function of a Document, so it needs no HTTP."""

    async def document_for(self, handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> Any:
        with mock.patch.dict("os.environ", {"PRIME_AGENT_WEBFETCH_RESPECT_ROBOTS": "0"}, clear=False):
            return await webfetch.fetch(
                "https://example.com/guide",
                resolver=public_resolver,
                transport=httpx.MockTransport(handler),
                **kwargs,
            )

    async def test_header_facts_and_body(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=HTML_PAGE, headers={"content-type": "text/html"})

        document = await self.document_for(handler)
        text = webfetch._render(document, 20_000)
        self.assertIn("# webfetch: https://example.com/guide", text)
        self.assertIn("**Guide & Reference**", text)
        self.assertIn("text/html", text)
        self.assertIn("# Install", text)

    async def test_truncation_points_at_fetch(self) -> None:
        long_html = "<html><body><main><p>" + ("word " * 5000) + "</p></main></body></html>"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=long_html, headers={"content-type": "text/html"})

        document = await self.document_for(handler)
        text = webfetch._render(document, 500)
        self.assertIn("[truncated at 500 of", text)
        self.assertIn("webfetch.fetch(url)", text)

    async def test_no_truncation_when_limit_is_zero(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=HTML_PAGE, headers={"content-type": "text/html"})

        document = await self.document_for(handler)
        self.assertNotIn("[truncated", webfetch._render(document, 0))

    async def test_error_document_renders_reason(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        document = await self.document_for(handler)
        text = webfetch._render(document, 20_000)
        self.assertIn("# webfetch failed", text)
        self.assertIn("HTTP 503", text)

    async def test_pdf_facts_are_reported(self) -> None:
        pdf = make_pdf(2)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

        document = await self.document_for(handler)
        text = webfetch._render(document, 20_000)
        self.assertIn("2 pages", text)
        self.assertIn("--- page 1 ---", text)

    async def test_binary_reports_saved_path(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"\x89PNG\r\n\x1a\nfake", headers={"content-type": "image/png"})

        document = await self.document_for(handler)
        assert document.saved_path
        self.addCleanup(Path(document.saved_path).unlink, missing_ok=True)
        text = webfetch._render(document, 20_000)
        self.assertIn("saved to", text)
        self.assertIn(document.saved_path, text)

    async def test_run_reports_failure_as_text_without_network(self) -> None:
        text = await webfetch.run("file:///etc/passwd")
        self.assertIn("webfetch failed", text)
        self.assertIn("open()", text)


if __name__ == "__main__":
    unittest.main()
