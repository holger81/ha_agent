"""Tests for GGUF filename validation and safe local model downloads."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

COMPONENT = Path(__file__).resolve().parents[1] / "custom_components" / "ha_agent"


def _ensure_ha_stubs() -> None:
    if "homeassistant.core" not in sys.modules:
        ha_pkg = types.ModuleType("homeassistant")
        ha_core = types.ModuleType("homeassistant.core")

        def callback(func):
            return func

        ha_core.HomeAssistant = object
        ha_core.callback = callback
        sys.modules["homeassistant"] = ha_pkg
        sys.modules["homeassistant.core"] = ha_core


def _load(name: str, path: Path):
    module_name = f"ha_agent.{name.replace('/', '.')}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if "ha_agent" not in sys.modules:
        package = types.ModuleType("ha_agent")
        package.__path__ = [str(COMPONENT)]  # type: ignore[attr-defined]
        sys.modules["ha_agent"] = package
    _ensure_ha_stubs()
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


model_download = _load(
    "eval.model_download",
    COMPONENT / "eval" / "model_download.py",
)
discover_runner = _load(
    "eval.discover_runner",
    COMPONENT / "eval" / "discover_runner.py",
)
eval_models = _load("eval.models", COMPONENT / "eval" / "models.py")
model_registry = _load("eval.model_registry", COMPONENT / "eval" / "model_registry.py")


class _FakeContent:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _size: int):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks: list[bytes], status: int = 200) -> None:
        self.status = status
        self.headers = {"Content-Length": str(sum(len(c) for c in chunks))}
        self.content = _FakeContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def text(self) -> str:
        return ""


class _FakeSession:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.requested: list[str] = []

    def get(self, url: str, timeout=None):
        self.requested.append(url)
        return _FakeResponse(self.chunks)


@pytest.mark.parametrize(
    "name",
    [
        "model-Q4_K_M.gguf",
        "Model.GGUF",
        "weird name with spaces.gguf",
    ],
)
def test_safe_gguf_filename_accepts_plain_basenames(name: str) -> None:
    assert model_download.safe_gguf_filename(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        None,
        42,
        "model.bin",
        "sub/model.gguf",
        "..\\model.gguf",
        "../model.gguf",
        "..",
        "model..gguf",
        "/etc/passwd.gguf",
        " model.gguf",
        "model.gguf\n",
        "model\x00.gguf",
    ],
)
def test_safe_gguf_filename_rejects_traversal(name) -> None:
    assert model_download.safe_gguf_filename(name) is None


def test_delete_local_model_file_rejects_escape(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    outside = tmp_path / "secret.gguf"
    outside.write_bytes(b"keep me")

    with pytest.raises(ValueError):
        model_download.delete_local_model_file(
            str(models_dir / ".." / "secret.gguf"), models_dir=models_dir
        )
    with pytest.raises(ValueError):
        model_download.delete_local_model_file(str(outside), models_dir=models_dir)
    with pytest.raises(ValueError):
        model_download.delete_local_model_file(str(tmp_path / "notes.txt"))
    assert outside.exists()


def test_delete_local_model_file_rejects_symlink_escape(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    outside = tmp_path / "secret.gguf"
    outside.write_bytes(b"keep me")
    link = models_dir / "link.gguf"
    link.symlink_to(outside)

    with pytest.raises(ValueError):
        model_download.delete_local_model_file(str(link), models_dir=models_dir)
    assert outside.exists()


def test_delete_local_model_file_deletes_inside_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    target = models_dir / "model.gguf"
    target.write_bytes(b"x")
    assert model_download.delete_local_model_file(str(target), models_dir=models_dir)
    assert not target.exists()
    assert (
        model_download.delete_local_model_file(str(target), models_dir=models_dir)
        is False
    )
    assert model_download.delete_local_model_file(None) is False


@pytest.mark.asyncio
async def test_download_hf_gguf_rejects_traversal_before_network(
    tmp_path: Path,
) -> None:
    session = _FakeSession([b"data"])
    with pytest.raises(ValueError):
        await model_download.download_hf_gguf(
            session,
            repo_id="org/repo",
            filename="../evil.gguf",
            dest_path=tmp_path / "models" / ".." / "evil.gguf",
            models_dir=tmp_path / "models",
        )
    with pytest.raises(ValueError):
        await model_download.download_hf_gguf(
            session,
            repo_id="org/repo",
            filename="evil.gguf",
            dest_path=tmp_path / "evil.gguf",
            models_dir=tmp_path / "models",
        )
    assert session.requested == []
    assert not (tmp_path / "evil.gguf").exists()


@pytest.mark.asyncio
async def test_download_hf_gguf_streams_to_file(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    chunks = [b"abc", b"", b"def", b"ghi"]
    session = _FakeSession(chunks)
    progress: list[dict] = []
    dest = models_dir / "model.gguf"

    result = await model_download.download_hf_gguf(
        session,
        repo_id="org/repo",
        filename="model.gguf",
        dest_path=dest,
        models_dir=models_dir,
        on_progress=progress.append,
    )

    assert result["ok"] is True
    assert result["bytes_done"] == 9
    assert dest.read_bytes() == b"abcdefghi"
    assert progress[-1]["bytes_done"] == 9
    assert len(session.requested) == 1
    assert "model.gguf" in session.requested[0]


@pytest.mark.asyncio
async def test_download_hf_gguf_cancel_discards_partial(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    dest = models_dir / "model.gguf"
    calls = {"n": 0}

    def _cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    result = await model_download.download_hf_gguf(
        _FakeSession([b"abc", b"def"]),
        repo_id="org/repo",
        filename="model.gguf",
        dest_path=dest,
        models_dir=models_dir,
        cancel_check=_cancel,
    )

    assert result["ok"] is False
    assert result.get("cancelled") is True
    assert not dest.exists()


@pytest.mark.asyncio
async def test_ensure_model_available_rejects_unsafe_filename(tmp_path: Path) -> None:
    run = eval_models.DiscoverRun(
        id="1",
        entry_id="entry",
        status="running",
        started_at=0.0,
    )
    state = eval_models.DiscoverRunState(run=run)
    proposal = model_registry.ModelProposal(
        model_id="org/repo:evil",
        source_url=None,
        reason="test",
        hf_repo="org/repo",
        hf_filename="../evil.gguf",
    )

    class _Hass:
        async def async_add_executor_job(self, func, *args):
            return await asyncio.to_thread(func, *args)

    async def _boom(*_args, **_kwargs):
        raise AssertionError("probe_server must not run for unsafe filenames")

    original = discover_runner.probe_server
    discover_runner.probe_server = _boom
    try:
        ready, local_path = await discover_runner._ensure_model_available(
            _Hass(),
            object(),
            state,
            object(),
            object(),
            proposal,
            index=1,
            total=1,
            models_dir=str(tmp_path),
            webhook_url=None,
        )
    finally:
        discover_runner.probe_server = original

    assert ready is False
    assert local_path is None
    assert "invalid GGUF filename" in state.run.progress["message"]
    assert not (tmp_path.parent / "evil.gguf").exists()
