"""Gemini Files API uploads, for payloads too large to inline.

`generateContent` carries inline data in the request body, which caps a document at
roughly 18 MB. Larger files must be uploaded first and then referenced by URI. This
implements Google's documented resumable upload protocol.

Not every deployment exposes it: corporate Gemini gateways commonly proxy
`generateContent` only, and answer `404` on the files endpoints. That is detected and
reported as `FilesApiUnsupported` so the caller can fall back instead of failing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from . import _provider

# Google deletes uploaded files after 48h; this skill also deletes them after use.
POLL_ATTEMPTS = 12
POLL_INITIAL_DELAY = 0.75
POLL_MAX_DELAY = 4.0


class FilesApiUnsupported(RuntimeError):
    """This endpoint does not expose the Files API (typically a gateway)."""


@dataclass
class UploadedFile:
    uri: str
    name: str
    """Resource name such as `files/abc123`, used to delete it again."""
    mime_type: str


def upload_base(base_url: str) -> str:
    """Derive the upload prefix: `<root>/<version>` becomes `<root>/upload/<version>`.

    Only the path is rewritten. Splitting the raw string instead would break on a
    base URL without a path, where the last `/` belongs to the scheme and the
    result would silently point at a host called `upload`.
    """
    parts = urlsplit(base_url.rstrip("/"))
    segments = [segment for segment in parts.path.split("/") if segment]
    if segments:
        path = "/" + "/".join([*segments[:-1], "upload", segments[-1]])
    else:
        path = "/upload"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _validated_session_url(base_url: str, session_url: str) -> str:
    """Accept only credential-free upload URLs on the configured origin."""
    try:
        base = urlsplit(base_url)
        session = urlsplit(session_url)
        base_scheme = base.scheme.lower()
        session_scheme = session.scheme.lower()
        base_port = base.port or (443 if base_scheme == "https" else 80 if base_scheme == "http" else None)
        session_port = session.port or (
            443 if session_scheme == "https" else 80 if session_scheme == "http" else None
        )
        if session.username is not None or session.password is not None:
            raise RuntimeError("upload service returned a credential-bearing session URL")
        same_origin = (
            base_scheme,
            (base.hostname or "").lower().rstrip("."),
            base_port,
        ) == (
            session_scheme,
            (session.hostname or "").lower().rstrip("."),
            session_port,
        )
    except ValueError as error:
        raise RuntimeError("upload service returned a malformed session URL") from error

    if not same_origin:
        raise RuntimeError("upload service returned a cross-origin session URL")
    return session_url


def _file_from_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    candidate = payload.get("file")
    if isinstance(candidate, dict):
        return candidate
    return payload if "uri" in payload or "name" in payload else None


async def _wait_until_active(
    client: httpx.AsyncClient,
    base_url: str,
    key: str,
    name: str,
    timeout: Optional[float],
) -> None:
    """Poll a freshly uploaded file until the service finishes processing it."""
    delay = POLL_INITIAL_DELAY
    for _ in range(POLL_ATTEMPTS):
        await asyncio.sleep(delay)
        delay = min(delay * 1.6, POLL_MAX_DELAY)
        try:
            response = await _provider.request(
                client,
                "GET",
                f"{base_url}/{name}",
                headers={"x-goog-api-key": key},
                timeout=timeout,
            )
        except httpx.HTTPError:
            continue
        if response.status_code == 404:
            raise FilesApiUnsupported(f"{base_url}/{name} is not available")
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"file status check for {name} returned HTTP {response.status_code}"
            )
        if response.status_code >= 400:
            continue
        try:
            file = _file_from_payload(response.json()) or {}
        except ValueError:
            continue
        state = str(file.get("state") or "").upper()
        if state in ("ACTIVE", ""):
            return
        if state == "FAILED":
            raise RuntimeError(f"the service failed to process {name}")
    raise RuntimeError(f"{name} was still processing after {POLL_ATTEMPTS} checks")


async def upload(
    client: httpx.AsyncClient,
    base_url: str,
    key: str,
    content: bytes,
    mime_type: str,
    *,
    display_name: str = "webfetch-upload",
    timeout: Optional[float] = None,
) -> UploadedFile:
    """Upload bytes with the resumable protocol and return the usable file reference."""
    headers = {
        "x-goog-api-key": key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(content)),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "content-type": "application/json",
    }
    start_url = f"{upload_base(base_url)}/files"
    try:
        start = await _provider.request(
            client,
            "POST",
            start_url,
            headers=headers,
            json={"file": {"display_name": display_name}},
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise RuntimeError(f"upload could not start: {type(error).__name__}") from error

    if start.status_code == 404:
        raise FilesApiUnsupported(
            f"{start_url} returned 404; this endpoint does not expose the Gemini Files API"
        )
    if start.status_code >= 400:
        raise RuntimeError(f"upload could not start: HTTP {start.status_code}")

    session_url = start.headers.get("x-goog-upload-url") or start.headers.get("X-Goog-Upload-URL")
    if not session_url:
        raise FilesApiUnsupported(
            f"{start_url} accepted the request but returned no upload URL; "
            "the Files API is not usable here"
        )
    session_url = _validated_session_url(base_url, session_url)

    try:
        finish = await _provider.request(
            client,
            "POST",
            session_url,
            headers={
                "Content-Length": str(len(content)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=content,
            timeout=timeout,
        )
    except httpx.HTTPError as error:
        raise RuntimeError(f"upload failed: {type(error).__name__}") from error
    if finish.status_code >= 400:
        raise RuntimeError(f"upload failed: HTTP {finish.status_code}")

    try:
        file = _file_from_payload(finish.json())
    except ValueError:
        file = None
    if not file:
        raise RuntimeError("upload finished but the service returned no file record")

    uri = file.get("uri")
    name = file.get("name") or ""
    state = str(file.get("state") or "").upper()
    if not isinstance(uri, str) or not uri:
        await delete(client, base_url, key, name, timeout=timeout)
        raise RuntimeError("upload finished but the service returned no file URI")
    if state == "FAILED":
        await delete(client, base_url, key, name, timeout=timeout)
        raise RuntimeError(f"the service failed to process {name or 'the uploaded file'}")
    if state == "PROCESSING":
        if not name:
            raise RuntimeError("upload is processing but the service returned no file name")
        try:
            if timeout is None:
                await _wait_until_active(client, base_url, key, name, timeout)
            else:
                try:
                    await asyncio.wait_for(
                        _wait_until_active(client, base_url, key, name, timeout),
                        timeout=max(0.001, timeout),
                    )
                except asyncio.TimeoutError as error:
                    raise RuntimeError(f"timed out waiting for {name} to become active") from error
        except BaseException:
            # upload() has not returned yet, so its caller cannot run the normal
            # finally cleanup. Delete the partially processed resource here.
            await delete(client, base_url, key, name, timeout=timeout)
            raise

    return UploadedFile(uri=uri, name=name, mime_type=mime_type)


async def delete(
    client: httpx.AsyncClient,
    base_url: str,
    key: str,
    name: str,
    *,
    timeout: Optional[float] = None,
) -> None:
    """Delete an uploaded file. Best effort: the service also expires it on its own."""
    if not name:
        return
    try:
        await _provider.request(
            client,
            "DELETE",
            f"{base_url}/{name}",
            headers={"x-goog-api-key": key},
            timeout=timeout,
        )
    except (httpx.HTTPError, RuntimeError):
        pass
