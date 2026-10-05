"""Download GGUF models from Hugging Face into a local models directory."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp

from ..const import LOGGER

CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[dict[str, Any]], None]

_WRITE_FLUSH_BYTES = 8 * 1024 * 1024


def safe_gguf_filename(name: Any) -> str | None:
    """Return ``name`` when it is a plain ``*.gguf`` basename, else ``None``.

    Rejects empty names, path separators, ``..`` components, and anything whose
    basename differs from the input so LLM-supplied names can never escape the
    models directory.
    """
    if not isinstance(name, str):
        return None
    candidate = name.strip()
    if not candidate or candidate != name:
        return None
    if "/" in candidate or "\\" in candidate or ".." in candidate:
        return None
    if "\x00" in candidate or candidate in {".", ""}:
        return None
    if not candidate.lower().endswith(".gguf"):
        return None
    if Path(candidate).name != candidate:
        return None
    return candidate


def _ensure_within_dir(dest_path: Path, models_dir: Path | None) -> Path:
    """Resolve ``dest_path`` and raise ``ValueError`` if it escapes ``models_dir``."""
    base = (models_dir if models_dir is not None else dest_path.parent).resolve()
    resolved = dest_path.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError(f"Refusing to touch {dest_path}: outside {base}")
    if resolved == base:
        raise ValueError(f"Refusing to touch {dest_path}: not a file path")
    return resolved


class _ExecutorFileWriter:
    """Write a file from async code with every blocking call in the executor.

    The file is opened once; chunks are buffered in memory and flushed in
    ``_WRITE_FLUSH_BYTES`` batches so the event loop does one executor hop per
    batch instead of one per network chunk.
    """

    def __init__(self, path: Path, loop: asyncio.AbstractEventLoop) -> None:
        self._path = path
        self._loop = loop
        self._handle: Any = None
        self._buffer = bytearray()

    def _open_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("wb")

    def _flush_sync(self, data: bytes) -> None:
        if self._handle is not None and data:
            self._handle.write(data)

    def _close_sync(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _discard_sync(self) -> None:
        self._close_sync()
        self._path.unlink(missing_ok=True)

    async def open(self) -> None:
        await self._loop.run_in_executor(None, self._open_sync)

    async def write(self, chunk: bytes) -> None:
        self._buffer.extend(chunk)
        if len(self._buffer) >= _WRITE_FLUSH_BYTES:
            await self._flush()

    async def _flush(self) -> None:
        if not self._buffer:
            return
        data = bytes(self._buffer)
        self._buffer.clear()
        await self._loop.run_in_executor(None, self._flush_sync, data)

    async def close(self) -> None:
        await self._flush()
        await self._loop.run_in_executor(None, self._close_sync)

    async def discard(self) -> None:
        self._buffer.clear()
        try:
            await self._loop.run_in_executor(None, self._discard_sync)
        except OSError as err:
            LOGGER.debug("Could not discard partial download %s: %s", self._path, err)


def hf_download_url(repo_id: str, filename: str) -> str:
    if safe_gguf_filename(filename) is None:
        raise ValueError(f"Invalid GGUF filename: {filename!r}")
    encoded_repo = quote(repo_id, safe="")
    encoded_file = quote(filename, safe="")
    return f"https://huggingface.co/{encoded_repo}/resolve/main/{encoded_file}"


async def download_hf_gguf(
    session: aiohttp.ClientSession,
    *,
    repo_id: str,
    filename: str,
    dest_path: Path,
    models_dir: Path | str | None = None,
    cancel_check: CancelCheck | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Stream a GGUF file from Hugging Face to dest_path.

    All filesystem work runs in the executor; the destination file is opened
    once and chunks are flushed in large batches. Raises ``ValueError`` when
    ``filename`` is not a plain GGUF basename or ``dest_path`` escapes
    ``models_dir`` (defaults to ``dest_path.parent``).
    """
    if safe_gguf_filename(filename) is None:
        raise ValueError(f"Invalid GGUF filename: {filename!r}")
    base_dir = Path(models_dir) if models_dir is not None else None
    loop = asyncio.get_running_loop()
    # Resolving touches the filesystem; keep it off the event loop.
    resolved = await loop.run_in_executor(None, _ensure_within_dir, dest_path, base_dir)
    url = hf_download_url(repo_id, filename)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    bytes_done = 0
    writer = _ExecutorFileWriter(resolved, loop)
    try:
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                body = await response.text()
                return {
                    "ok": False,
                    "path": str(dest_path),
                    "error": f"HTTP {response.status}: {body[:200]}",
                }
            total = int(response.headers.get("Content-Length") or 0)
            await writer.open()

            async for chunk in response.content.iter_chunked(1024 * 256):
                if cancel_check and cancel_check():
                    await writer.discard()
                    return {
                        "ok": False,
                        "path": str(dest_path),
                        "error": "Download cancelled.",
                        "cancelled": True,
                    }
                if not chunk:
                    continue
                await writer.write(chunk)
                bytes_done += len(chunk)
                if on_progress:
                    on_progress(
                        {
                            "bytes_done": bytes_done,
                            "bytes_total": total,
                            "filename": filename,
                        }
                    )
            await writer.close()
    except (TimeoutError, aiohttp.ClientError, OSError) as err:
        LOGGER.warning("HF download failed for %s/%s: %s", repo_id, filename, err)
        await writer.discard()
        return {"ok": False, "path": str(dest_path), "error": str(err)}
    except Exception:
        await writer.discard()
        raise

    return {
        "ok": True,
        "path": str(dest_path),
        "bytes_done": bytes_done,
        "url": url,
    }


async def download_via_webhook(
    session: aiohttp.ClientSession,
    webhook_url: str,
    *,
    model_id: str,
    hf_repo: str,
    hf_filename: str,
    source_url: str | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Ask a host-side webhook to download a model into the llama server cache."""
    payload = {
        "model_id": model_id,
        "hf_repo": hf_repo,
        "hf_filename": hf_filename,
        "source_url": source_url or hf_download_url(hf_repo, hf_filename),
    }
    timeout = aiohttp.ClientTimeout(total=7200)
    try:
        async with session.post(
            webhook_url,
            json=payload,
            timeout=timeout,
        ) as response:
            body = await response.text()
            ok = response.status in {200, 204}
            return {
                "ok": ok,
                "mode": "webhook",
                "status": response.status,
                "response": body[:500],
                "error": None if ok else body[:300],
            }
    except (TimeoutError, aiohttp.ClientError) as err:
        if cancel_check and cancel_check():
            return {
                "ok": False,
                "mode": "webhook",
                "cancelled": True,
                "error": str(err),
            }
        return {"ok": False, "mode": "webhook", "error": str(err)}


def manual_download_hint(hf_repo: str, hf_filename: str) -> dict[str, str]:
    """Return URLs and a sample docker exec hint for manual host download."""
    url = hf_download_url(hf_repo, hf_filename)
    return {
        "hf_url": url,
        "docker_hint": (
            f"docker exec -it <llama-container> huggingface-cli download "
            f"{hf_repo} {hf_filename} --local-dir /models"
        ),
        "llama_cli_hint": f"llama-server -hf {hf_repo} --hf-file {hf_filename}",
    }


def delete_local_model_file(
    path: str | None,
    *,
    models_dir: Path | str | None = None,
) -> bool:
    """Remove a downloaded GGUF file when a trial is rejected.

    Blocking; call from an executor. Raises ``ValueError`` when the file name
    is not a plain ``*.gguf`` basename or the resolved path escapes
    ``models_dir`` (defaults to the file's parent directory).
    """
    if not path:
        return False
    target = Path(path)
    if safe_gguf_filename(target.name) is None:
        raise ValueError(f"Refusing to delete non-GGUF path: {path!r}")
    base_dir = Path(models_dir) if models_dir is not None else None
    resolved = _ensure_within_dir(target, base_dir)
    if not resolved.is_file():
        return False
    try:
        resolved.unlink()
        return True
    except OSError as err:
        LOGGER.warning("Could not delete model file %s: %s", path, err)
        return False
