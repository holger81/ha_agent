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


def hf_download_url(repo_id: str, filename: str) -> str:
    encoded_repo = quote(repo_id, safe="")
    encoded_file = quote(filename, safe="/")
    return f"https://huggingface.co/{encoded_repo}/resolve/main/{encoded_file}"


async def download_hf_gguf(
    session: aiohttp.ClientSession,
    *,
    repo_id: str,
    filename: str,
    dest_path: Path,
    cancel_check: CancelCheck | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Stream a GGUF file from Hugging Face to dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = hf_download_url(repo_id, filename)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    bytes_done = 0
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
            loop = asyncio.get_running_loop()

            def _write_chunk(chunk: bytes) -> None:
                with dest_path.open("ab") as handle:
                    handle.write(chunk)

            if dest_path.exists():
                dest_path.unlink()

            async for chunk in response.content.iter_chunked(1024 * 256):
                if cancel_check and cancel_check():
                    if dest_path.exists():
                        dest_path.unlink()
                    return {
                        "ok": False,
                        "path": str(dest_path),
                        "error": "Download cancelled.",
                        "cancelled": True,
                    }
                if not chunk:
                    continue
                await loop.run_in_executor(None, _write_chunk, chunk)
                bytes_done += len(chunk)
                if on_progress:
                    on_progress(
                        {
                            "bytes_done": bytes_done,
                            "bytes_total": total,
                            "filename": filename,
                        }
                    )
    except (TimeoutError, aiohttp.ClientError, OSError) as err:
        LOGGER.warning("HF download failed for %s/%s: %s", repo_id, filename, err)
        if dest_path.exists():
            dest_path.unlink(missing_ok=True)
        return {"ok": False, "path": str(dest_path), "error": str(err)}

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


def delete_local_model_file(path: str | None) -> bool:
    """Remove a downloaded GGUF file when a trial is rejected."""
    if not path:
        return False
    target = Path(path)
    if not target.is_file():
        return False
    try:
        target.unlink()
        return True
    except OSError as err:
        LOGGER.warning("Could not delete model file %s: %s", path, err)
        return False
