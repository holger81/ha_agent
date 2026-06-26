"""Probe llama.cpp server capabilities beyond the OpenAI /v1 API."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlparse

import aiohttp

from .config_helpers import LlmBackend
from .const import LOGGER


def server_root_from_base_url(base_url: str) -> str:
    """Derive the llama.cpp server root from an OpenAI-compatible base URL."""
    parsed = urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")


@dataclass(slots=True)
class ServerHealth:
    """Result from GET /health or /v1/health."""

    status: str
    slots_idle: int | None = None
    slots_processing: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServerProps:
    """Result from GET /props."""

    total_slots: int | None = None
    model_path: str | None = None
    n_ctx: int | None = None
    build_info: str | None = None
    chat_template_caps: dict[str, Any] = field(default_factory=dict)
    modalities: dict[str, Any] = field(default_factory=dict)
    is_sleeping: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServerSlot:
    """One slot from GET /slots."""

    id: int
    n_ctx: int | None = None
    is_processing: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ServerMetrics:
    """Parsed Prometheus text from GET /metrics when enabled."""

    raw_text: str = ""
    parsed: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ModelInfo:
    """One model entry from GET /v1/models."""

    model_id: str
    status: str = "unknown"
    source: str = "unknown"
    is_preset: bool = False
    path: str | None = None
    input_modalities: list[str] = field(default_factory=list)
    output_modalities: list[str] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)
    failed: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def model_info_to_dict(info: ModelInfo) -> dict[str, Any]:
    """Serialize one catalog model for API responses."""
    return {
        "model_id": info.model_id,
        "status": info.status,
        "source": info.source,
        "is_preset": info.is_preset,
        "path": info.path,
        "input_modalities": list(info.input_modalities),
        "output_modalities": list(info.output_modalities),
        "progress": dict(info.progress),
        "failed": info.failed,
    }


def model_suitable_for_voice_agent(info: ModelInfo) -> bool:
    """Return False for multimodal-only models unsuitable for text/voice eval."""
    inputs = {item.lower() for item in info.input_modalities}
    outputs = {item.lower() for item in info.output_modalities}
    if inputs.intersection({"image", "audio"}):
        return False
    return not (outputs and "text" not in outputs)


def hf_repo_suitable_for_voice_agent(repo_id: str) -> bool:
    """Heuristic filter for HF repos before they appear in the catalog."""
    lowered = repo_id.lower()
    skip_markers = ("-vl-", "-audio-", "/vl-", "vision", "mmproj", "-audio")
    return not any(marker in lowered for marker in skip_markers)


def normalize_download_progress(data: dict[str, Any]) -> dict[str, Any]:
    """Extract byte/percent progress from SSE, catalog, or nested payloads."""
    if not isinstance(data, dict):
        return {}
    layers: list[dict[str, Any]] = [data]
    for key in ("payload", "progress", "data"):
        inner = data.get(key)
        if isinstance(inner, dict):
            layers.append(inner)
    bytes_done: int | None = None
    bytes_total: int | None = None
    percent: float | None = None
    for layer in layers:
        for done_key, total_key in (
            ("bytes_done", "bytes_total"),
            ("n_done", "n_total"),
            ("downloaded", "total"),
        ):
            done = layer.get(done_key)
            total = layer.get(total_key)
            if done is not None and total:
                bytes_done = int(done)
                bytes_total = int(total)
        raw_percent = layer.get("percent")
        if raw_percent is None:
            raw_percent = layer.get("progress")
        if isinstance(raw_percent, (int, float)):
            percent = float(raw_percent)
    result: dict[str, Any] = {}
    if bytes_done is not None and bytes_total:
        result["bytes_done"] = bytes_done
        result["bytes_total"] = bytes_total
    if percent is not None:
        result["percent"] = percent
    return result


def download_progress_percent(data: dict[str, Any] | None) -> int | None:
    """Return an integer 0-100 download percent when progress data is available."""
    if not data:
        return None
    percent = data.get("percent")
    if isinstance(percent, (int, float)):
        value = float(percent)
        if value <= 1:
            value *= 100
        return max(0, min(100, int(value)))
    bytes_done = data.get("bytes_done")
    bytes_total = data.get("bytes_total")
    if bytes_done is not None and bytes_total:
        return max(0, min(100, int((bytes_done / bytes_total) * 100)))
    return None


@dataclass(slots=True)
class ServerCapabilities:
    """Aggregated llama.cpp server probe for eval and tuning."""

    server_root: str
    models: list[str] = field(default_factory=list)
    loaded_models: list[str] = field(default_factory=list)
    model_details: list[ModelInfo] = field(default_factory=list)
    router_role: str | None = None
    max_instances: int | None = None
    models_autoload: bool | None = None
    models_download_via_api: bool = False
    props_writable: bool = False
    health: ServerHealth | None = None
    props: ServerProps | None = None
    slots: list[ServerSlot] = field(default_factory=list)
    metrics: ServerMetrics | None = None
    endpoints_available: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses and LLM prompts."""
        return {
            "server_root": self.server_root,
            "models": list(self.models),
            "loaded_models": list(self.loaded_models),
            "model_details": [
                model_info_to_dict(item) for item in self.model_details
            ],
            "router_role": self.router_role,
            "max_instances": self.max_instances,
            "models_autoload": self.models_autoload,
            "models_download_via_api": self.models_download_via_api,
            "props_writable": self.props_writable,
            "health": self.health.raw if self.health else None,
            "props": self.props.raw if self.props else None,
            "slots": [slot.raw for slot in self.slots],
            "metrics": self.metrics.parsed if self.metrics else None,
            "endpoints_available": list(self.endpoints_available),
            "errors": list(self.errors),
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        """Return a compact summary for recommendation prompts."""
        props = self.props
        health = self.health
        return {
            "model_count": len(self.models),
            "loaded_model_count": len(self.loaded_models),
            "models": self.models[:20],
            "loaded_models": self.loaded_models[:20],
            "router_role": self.router_role,
            "max_instances": self.max_instances,
            "models_autoload": self.models_autoload,
            "models_download_via_api": self.models_download_via_api,
            "props_writable": self.props_writable,
            "total_slots": props.total_slots if props else None,
            "n_ctx": props.n_ctx if props else None,
            "model_path": props.model_path if props else None,
            "build_info": props.build_info if props else None,
            "is_sleeping": props.is_sleeping if props else None,
            "modalities": props.modalities if props else {},
            "health_status": health.status if health else None,
            "slots_idle": health.slots_idle if health else None,
            "slots_processing": health.slots_processing if health else None,
            "slot_count": len(self.slots),
            "metrics": self.metrics.parsed if self.metrics else {},
            "endpoints_available": list(self.endpoints_available),
        }


_METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$"
)


def _parse_prometheus(text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE.match(line)
        if not match:
            continue
        try:
            parsed[match.group("name")] = float(match.group("value"))
        except ValueError:
            continue
    return parsed


def _headers(backend: LlmBackend) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json, text/plain"}
    if backend.api_key:
        headers["Authorization"] = f"Bearer {backend.api_key}"
    return headers


class LlmServerProbe:
    """Async client for llama.cpp server management endpoints."""

    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def probe(
        self,
        backend: LlmBackend,
        *,
        models: list[str] | None = None,
    ) -> ServerCapabilities:
        """Probe health, props, slots, metrics, and model list."""
        root = server_root_from_base_url(backend.base_url)
        caps = ServerCapabilities(server_root=root, models=list(models or []))
        headers = _headers(backend)
        timeout = aiohttp.ClientTimeout(total=15)

        if not caps.models:
            await self._fetch_models(backend, headers, timeout, caps)

        for path, handler in (
            ("/health", self._parse_health),
            ("/v1/health", self._parse_health),
            ("/props", self._parse_props),
            ("/metrics", self._parse_metrics),
        ):
            await self._try_endpoint(root, path, headers, timeout, caps, handler)

        await self._enrich_router_model_details(root, headers, timeout, caps)
        await self._probe_props_writable(root, headers, timeout, caps)
        await self._probe_models_download_api(root, headers, timeout, caps)
        return caps

    async def _fetch_models(
        self,
        backend: LlmBackend,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
        caps: ServerCapabilities,
        *,
        reload: bool = False,
    ) -> None:
        url = f"{backend.base_url.rstrip('/')}/models"
        if reload:
            url = f"{url}?reload=1"
        try:
            async with self._session.get(
                url, headers=headers, timeout=timeout
            ) as response:
                body = await response.text()
                if response.status != 200:
                    caps.errors.append(f"/models HTTP {response.status}")
                    return
                data = json.loads(body)
        except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as err:
            caps.errors.append(f"/models failed: {err}")
            return

        models: list[str] = []
        loaded: list[str] = []
        details: list[ModelInfo] = []
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            parsed = _parse_model_entry(item)
            if parsed is None:
                continue
            models.append(parsed.model_id)
            details.append(parsed)
            if parsed.status == "loaded":
                loaded.append(parsed.model_id)
        caps.models = sorted(models)
        caps.loaded_models = sorted(loaded)
        caps.model_details = details

    async def _enrich_router_model_details(
        self,
        root: str,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
        caps: ServerCapabilities,
    ) -> None:
        """Fetch per-model props/slots when running in llama.cpp router mode."""
        sample_model = caps.loaded_models[0] if caps.loaded_models else None
        if not sample_model:
            return
        query = f"?model={quote(sample_model, safe='')}"
        await self._try_endpoint(
            root,
            f"/props{query}",
            headers,
            timeout,
            caps,
            self._parse_props,
        )
        await self._try_endpoint(
            root,
            f"/slots{query}",
            headers,
            timeout,
            caps,
            self._parse_slots,
        )

    async def _probe_props_writable(
        self,
        root: str,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
        caps: ServerCapabilities,
    ) -> None:
        """Detect whether POST /props is enabled on the server."""
        if caps.router_role == "router":
            caps.props_writable = False
            return
        url = f"{root}/props"
        try:
            async with self._session.post(
                url,
                json={},
                headers={**headers, "Content-Type": "application/json"},
                timeout=timeout,
            ) as response:
                caps.props_writable = response.status in {200, 204}
        except (TimeoutError, aiohttp.ClientError):
            caps.props_writable = False

    async def _probe_models_download_api(
        self,
        root: str,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
        caps: ServerCapabilities,
    ) -> None:
        """Detect router HF download support via GET /models/sse."""
        if caps.router_role != "router":
            return
        url = f"{root}/models/sse"
        try:
            async with self._session.get(
                url,
                headers={**headers, "Accept": "text/event-stream"},
                timeout=aiohttp.ClientTimeout(total=5, sock_connect=5),
            ) as response:
                if response.status == 200:
                    caps.models_download_via_api = True
                    if "/models/sse" not in caps.endpoints_available:
                        caps.endpoints_available.append("/models/sse")
        except (TimeoutError, aiohttp.ClientError):
            # Router builds without SSE may still accept POST /models.
            caps.models_download_via_api = True

    async def _try_endpoint(
        self,
        root: str,
        path: str,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
        caps: ServerCapabilities,
        parser,
    ) -> None:
        url = f"{root}{path}"
        try:
            async with self._session.get(
                url, headers=headers, timeout=timeout
            ) as response:
                body = await response.text()
                if response.status != 200:
                    return
                caps.endpoints_available.append(path)
                parser(caps, body)
        except (TimeoutError, aiohttp.ClientError) as err:
            LOGGER.debug("llama.cpp probe %s unavailable: %s", path, err)

    @staticmethod
    def _parse_health(caps: ServerCapabilities, body: str) -> None:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            caps.errors.append("/health returned invalid JSON")
            return
        if not isinstance(data, dict):
            return
        caps.health = ServerHealth(
            status=str(
                data.get("status")
                or data.get("error", {}).get("message")
                or "unknown"
            ),
            slots_idle=_optional_int(data.get("slots_idle")),
            slots_processing=_optional_int(data.get("slots_processing")),
            raw=data,
        )

    @staticmethod
    def _parse_props(caps: ServerCapabilities, body: str) -> None:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            caps.errors.append("/props returned invalid JSON")
            return
        if not isinstance(data, dict):
            return
        caps.router_role = str(data["role"]) if data.get("role") else caps.router_role
        caps.max_instances = (
            _optional_int(data.get("max_instances")) or caps.max_instances
        )
        if "models_autoload" in data:
            caps.models_autoload = bool(data.get("models_autoload"))
        default = data.get("default_generation_settings") or {}
        n_ctx = default.get("n_ctx") if isinstance(default, dict) else None
        if not n_ctx and isinstance(default, dict):
            params = default.get("params")
            if isinstance(params, dict):
                n_ctx = params.get("n_ctx")
        caps.props = ServerProps(
            total_slots=_optional_int(data.get("total_slots")) or caps.max_instances,
            model_path=data.get("model_path"),
            n_ctx=_optional_int(n_ctx),
            build_info=data.get("build_info"),
            chat_template_caps=(
                data.get("chat_template_caps")
                if isinstance(data.get("chat_template_caps"), dict)
                else {}
            ),
            modalities=(
                data.get("modalities")
                if isinstance(data.get("modalities"), dict)
                else {}
            ),
            is_sleeping=bool(data.get("is_sleeping")),
            raw=data,
        )

    @staticmethod
    def _parse_slots(caps: ServerCapabilities, body: str) -> None:
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            caps.errors.append("/slots returned invalid JSON")
            return
        if not isinstance(data, list):
            return
        slots: list[ServerSlot] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            slots.append(
                ServerSlot(
                    id=int(item.get("id", len(slots))),
                    n_ctx=_optional_int(item.get("n_ctx")),
                    is_processing=bool(item.get("is_processing")),
                    raw=item,
                )
            )
        caps.slots = slots

    @staticmethod
    def _parse_metrics(caps: ServerCapabilities, body: str) -> None:
        caps.metrics = ServerMetrics(raw_text=body, parsed=_parse_prometheus(body))


def _model_status(item: dict[str, Any]) -> str:
    status = item.get("status")
    if isinstance(status, dict):
        value = status.get("value")
        if isinstance(value, str) and value:
            return value
    if isinstance(status, str) and status:
        return status
    return "unknown"


def _parse_model_entry(item: dict[str, Any]) -> ModelInfo | None:
    model_id = item.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    status_obj = item.get("status")
    status_dict = status_obj if isinstance(status_obj, dict) else {}
    status = _model_status(item)
    architecture = item.get("architecture")
    arch = architecture if isinstance(architecture, dict) else {}
    input_modalities = [
        str(value)
        for value in arch.get("input_modalities", [])
        if isinstance(value, str)
    ]
    output_modalities = [
        str(value)
        for value in arch.get("output_modalities", [])
        if isinstance(value, str)
    ]
    is_preset = bool(status_dict.get("preset"))
    source = "preset" if is_preset else ("cache" if item.get("path") else "unknown")
    progress = status_dict.get("progress")
    return ModelInfo(
        model_id=model_id,
        status=status,
        source=source,
        is_preset=is_preset,
        path=item.get("path") if isinstance(item.get("path"), str) else None,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        progress=progress if isinstance(progress, dict) else {},
        failed=bool(status_dict.get("failed")),
        raw=item,
    )


async def fetch_models_catalog(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    *,
    reload: bool = False,
) -> ServerCapabilities:
    """Fetch the llama.cpp model catalog, optionally forcing a disk reload."""
    probe = LlmServerProbe(session)
    root = server_root_from_base_url(backend.base_url)
    caps = ServerCapabilities(server_root=root)
    await probe._fetch_models(
        backend,
        _headers(backend),
        aiohttp.ClientTimeout(total=30),
        caps,
        reload=reload,
    )
    return caps


async def refresh_models_catalog(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
) -> ServerCapabilities:
    """Reload and return the llama.cpp model catalog."""
    return await fetch_models_catalog(session, backend, reload=True)


def eval_candidate_models(
    capabilities: ServerCapabilities,
    *,
    configured_models: list[str],
    explicit_models: list[str] | None = None,
    include_unloaded: bool = False,
) -> list[str]:
    """Return models to benchmark, preferring loaded and configured ones."""
    if explicit_models:
        return list(dict.fromkeys(explicit_models))

    candidates: list[str] = []
    for model in configured_models:
        if model:
            candidates.append(model)
    if include_unloaded:
        candidates.extend(capabilities.models)
    else:
        candidates.extend(capabilities.loaded_models)
    deduped = list(dict.fromkeys(candidates))
    if deduped:
        return deduped
    return configured_models[:1] if configured_models else capabilities.models[:1]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def probe_server(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    *,
    models: list[str] | None = None,
) -> ServerCapabilities:
    """Convenience wrapper for a one-shot server probe."""
    probe = LlmServerProbe(session)
    return await probe.probe(backend, models=models)


async def load_model(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
) -> dict[str, Any]:
    """Load a model on a llama.cpp router server."""
    root = server_root_from_base_url(backend.base_url)
    url = f"{root}/models/load"
    headers = _headers(backend)
    headers["Content-Type"] = "application/json"
    timeout = aiohttp.ClientTimeout(total=300)
    try:
        async with session.post(
            url,
            json={"model": model_id},
            headers=headers,
            timeout=timeout,
        ) as response:
            body = await response.text()
            ok = response.status in {200, 204}
            parsed: dict[str, Any] = {}
            if body.strip():
                try:
                    loaded = json.loads(body)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = {"raw": body[:500]}
            return {
                "model": model_id,
                "ok": ok or bool(parsed.get("success")),
                "status": response.status,
                "response": parsed,
                "error": None if ok or parsed.get("success") else body[:300],
            }
    except (TimeoutError, aiohttp.ClientError) as err:
        return {
            "model": model_id,
            "ok": False,
            "status": None,
            "response": {},
            "error": str(err),
        }


async def unload_model(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
) -> dict[str, Any]:
    """Unload a model from a llama.cpp router server."""
    root = server_root_from_base_url(backend.base_url)
    url = f"{root}/models/unload"
    headers = _headers(backend)
    headers["Content-Type"] = "application/json"
    timeout = aiohttp.ClientTimeout(total=120)
    try:
        async with session.post(
            url,
            json={"model": model_id},
            headers=headers,
            timeout=timeout,
        ) as response:
            body = await response.text()
            ok = response.status in {200, 204}
            parsed: dict[str, Any] = {}
            if body.strip():
                try:
                    loaded = json.loads(body)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = {"raw": body[:500]}
            return {
                "model": model_id,
                "ok": ok or bool(parsed.get("success")),
                "status": response.status,
                "response": parsed,
                "error": None if ok or parsed.get("success") else body[:300],
            }
    except (TimeoutError, aiohttp.ClientError) as err:
        return {
            "model": model_id,
            "ok": False,
            "status": None,
            "response": {},
            "error": str(err),
        }


def models_delete_url(server_root: str, model_id: str) -> str:
    """Build DELETE /models?model=... for router cache removal."""
    return f"{server_root.rstrip('/')}/models?model={quote(model_id, safe='')}"


async def delete_model_from_router(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
) -> dict[str, Any]:
    """Delete a cached Hugging Face model from a llama.cpp router server."""
    root = server_root_from_base_url(backend.base_url)
    url = models_delete_url(root, model_id)
    headers = _headers(backend)
    timeout = aiohttp.ClientTimeout(total=120)
    try:
        async with session.delete(url, headers=headers, timeout=timeout) as response:
            body = await response.text()
            parsed: dict[str, Any] = {}
            if body.strip():
                try:
                    loaded = json.loads(body)
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = {"raw": body[:500]}
            ok = response.status in {200, 204} or bool(parsed.get("success"))
            preset = response.status == 400 and "preset" in body.lower()
            result = {
                "model": model_id,
                "ok": ok,
                "status": response.status,
                "response": parsed,
                "preset_model": preset,
                "error": None if ok else body[:300],
            }
            if ok:
                await refresh_models_catalog(session, backend)
            return result
    except (TimeoutError, aiohttp.ClientError) as err:
        return {
            "model": model_id,
            "ok": False,
            "status": None,
            "response": {},
            "error": str(err),
        }


async def preload_models(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_ids: list[str],
    *,
    loaded_models: list[str] | None = None,
    capabilities: ServerCapabilities | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    abort_on_cancel: bool = False,
) -> list[dict[str, Any]]:
    """Load models that are not already resident on the router."""
    caps = capabilities or await probe_server(session, backend)
    already_loaded = set(loaded_models or caps.loaded_models)
    results: list[dict[str, Any]] = []
    for model_id in model_ids:
        if cancel_check and cancel_check():
            results.append(
                {
                    "model": model_id,
                    "ok": False,
                    "cancelled": True,
                }
            )
            break
        if model_id in already_loaded:
            results.append(
                {
                    "model": model_id,
                    "ok": True,
                    "skipped": True,
                    "reason": "already loaded",
                }
            )
            continue

        def _progress(
            data: dict[str, Any],
            *,
            _model_id: str = model_id,
        ) -> None:
            if on_progress:
                on_progress({**data, "model": _model_id, "phase": "load"})

        result = await load_model_with_progress(
            session,
            backend,
            model_id,
            capabilities=caps,
            cancel_check=cancel_check,
            on_progress=_progress if on_progress else None,
            abort_on_cancel=abort_on_cancel,
        )
        result["skipped"] = False
        results.append(result)
        if result.get("ok"):
            already_loaded.add(model_id)
    return results


async def model_available_on_server(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
) -> bool:
    """Return True when model_id appears in the llama.cpp model catalog."""
    probe = LlmServerProbe(session)
    caps = await probe.probe(backend)
    return model_id in caps.models


def router_supports_hf_download(capabilities: ServerCapabilities) -> bool:
    """Return True when the server can download HF models via router HTTP API."""
    return (
        capabilities.router_role == "router"
        and capabilities.models_download_via_api
    )


def _catalog_status_ready(status: str) -> bool:
    """Return True when a model is present and not actively downloading."""
    return status in {"unloaded", "loaded", "sleeping", "unknown"}


def _parse_sse_event_block(event_name: str, data_line: str) -> dict[str, Any] | None:
    payload = data_line[5:].strip() if data_line.startswith("data:") else data_line
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if event_name:
        data.setdefault("event", event_name)
    return data


def _sse_model_event_outcome(
    event_name: str,
    data: dict[str, Any],
    *,
    model_id: str,
    wait_kind: str,
) -> dict[str, Any] | None:
    """Parse one models/sse event for download or load waits."""
    target = str(
        data.get("model")
        or data.get("model_id")
        or data.get("id")
        or model_id
    )
    if target not in {model_id, "*"}:
        return None
    inner = data.get("data") if isinstance(data.get("data"), dict) else data
    name = (event_name or str(data.get("event") or data.get("type") or "")).lower()
    status = str(inner.get("status") or inner.get("value") or "").lower()

    if (
        name in {"download_finished", "download_complete", "finished"}
        and wait_kind == "download"
    ):
        return {"ok": True, "model": model_id, "via": "sse", "event": name}
    if name in {"download_failed", "download_error", "failed"}:
        return {
            "ok": False,
            "model": model_id,
            "via": "sse",
            "error": str(data.get("error") or inner.get("error") or name),
        }
    if name in {"download_progress", "progress", "downloading"}:
        return {
            "progress": True,
            "model": model_id,
            "via": "sse",
            "event": name,
            "payload": inner or data,
        }
    if name == "model_status":
        if wait_kind == "load" and status == "loaded":
            return {"ok": True, "model": model_id, "via": "sse", "event": name}
        if wait_kind == "download" and status in {"unloaded", "loaded"}:
            return {"ok": True, "model": model_id, "via": "sse", "event": name}
        if status in {"loading", "downloading"}:
            progress = inner.get("progress")
            return {
                "progress": True,
                "model": model_id,
                "via": "sse",
                "event": name,
                "status": status,
                "payload": progress if isinstance(progress, dict) else inner,
            }
        if status == "failed" or inner.get("failed"):
            return {
                "ok": False,
                "model": model_id,
                "via": "sse",
                "error": "model_status failed",
            }
    return None


def _sse_download_outcome(
    event_name: str,
    data: dict[str, Any],
    *,
    model_id: str,
) -> dict[str, Any] | None:
    """Backward-compatible wrapper for download SSE parsing."""
    outcome = _sse_model_event_outcome(
        event_name,
        data,
        model_id=model_id,
        wait_kind="download",
    )
    if outcome is None or outcome.get("progress"):
        return None
    return outcome


async def request_model_download(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
) -> dict[str, Any]:
    """Ask a llama.cpp router to download a Hugging Face model by id."""
    root = server_root_from_base_url(backend.base_url)
    headers = _headers(backend)
    headers["Content-Type"] = "application/json"
    payload = {"model": model_id}
    timeout = aiohttp.ClientTimeout(total=120)
    last_error = "Router download endpoint unavailable."
    for path in ("/models", "/models/download"):
        url = f"{root}{path}"
        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                body = await response.text()
                parsed: dict[str, Any] = {}
                if body.strip():
                    try:
                        loaded = json.loads(body)
                        if isinstance(loaded, dict):
                            parsed = loaded
                    except json.JSONDecodeError:
                        parsed = {"raw": body[:500]}
                if response.status == 404:
                    last_error = f"{path} returned HTTP 404"
                    continue
                ok = response.status in {200, 204} or bool(parsed.get("success"))
                return {
                    "model": model_id,
                    "ok": ok,
                    "path": path,
                    "status": response.status,
                    "response": parsed,
                    "error": None if ok else body[:300],
                }
        except (TimeoutError, aiohttp.ClientError) as err:
            last_error = str(err)
    return {
        "model": model_id,
        "ok": False,
        "status": None,
        "response": {},
        "error": last_error,
    }


async def _get_catalog_model(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    reload: bool = False,
) -> ModelInfo | None:
    caps = await fetch_models_catalog(session, backend, reload=reload)
    for item in caps.model_details:
        if item.model_id == model_id:
            return item
    return None


async def _model_catalog_status(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    reload: bool = False,
) -> str | None:
    entry = await _get_catalog_model(session, backend, model_id, reload=reload)
    return None if entry is None else entry.status


async def _consume_models_sse(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    wait_kind: str,
    cancel_check: Callable[[], bool] | None,
    on_progress: Callable[[dict[str, Any]], None] | None,
    started: float,
    timeout: float,
) -> dict[str, Any] | None:
    root = server_root_from_base_url(backend.base_url)
    headers = _headers(backend)
    headers["Accept"] = "text/event-stream"
    url = f"{root}/models/sse"
    sse_timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=90)
    try:
        async with session.get(url, headers=headers, timeout=sse_timeout) as response:
            if response.status != 200:
                return None
            current_event = ""
            buffer = ""
            async for chunk in response.content.iter_any():
                if cancel_check and cancel_check():
                    return {"ok": False, "cancelled": True, "model": model_id}
                elapsed = time.monotonic() - started
                if elapsed > timeout:
                    return None
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    stripped = line.strip()
                    if stripped.startswith("event:"):
                        current_event = stripped[6:].strip()
                        continue
                    if not stripped.startswith("data:"):
                        if not stripped:
                            current_event = ""
                        continue
                    data = _parse_sse_event_block(current_event, stripped)
                    if data is None:
                        continue
                    outcome = _sse_model_event_outcome(
                        current_event,
                        data,
                        model_id=model_id,
                        wait_kind=wait_kind,
                    )
                    if outcome is None:
                        continue
                    if outcome.get("progress"):
                        if on_progress:
                            on_progress(
                                {
                                    **outcome,
                                    "wait_seconds": int(elapsed),
                                }
                            )
                        continue
                    if outcome.get("ok"):
                        outcome["wait_seconds"] = int(elapsed)
                    return outcome
    except (TimeoutError, aiohttp.ClientError) as err:
        LOGGER.debug("models/sse stream ended: %s", err)
    return None


async def _wait_for_router_model(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    wait_kind: str,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    poll_interval: float = 10.0,
    timeout: float = 7200.0,
    use_sse: bool = True,
    abort_on_cancel: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    if use_sse:
        outcome = await _consume_models_sse(
            session,
            backend,
            model_id,
            wait_kind=wait_kind,
            cancel_check=cancel_check,
            on_progress=on_progress,
            started=started,
            timeout=timeout,
        )
        if outcome is not None:
            if outcome.get("cancelled") and abort_on_cancel:
                await unload_model(session, backend, model_id)
            return outcome

    reload_next = False
    while True:
        if cancel_check and cancel_check():
            if abort_on_cancel:
                await unload_model(session, backend, model_id)
            return {"ok": False, "cancelled": True, "model": model_id}
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            return {
                "ok": False,
                "model": model_id,
                "error": f"Timed out after {int(timeout)}s waiting for {wait_kind}.",
            }
        entry = await _get_catalog_model(
            session,
            backend,
            model_id,
            reload=reload_next,
        )
        reload_next = False
        if entry is not None:
            if entry.failed:
                return {
                    "ok": False,
                    "model": model_id,
                    "error": "Model entered failed state.",
                    "status": entry.status,
                }
            if wait_kind == "download" and _catalog_status_ready(entry.status):
                return {
                    "ok": True,
                    "model": model_id,
                    "wait_seconds": int(elapsed),
                    "via": "catalog",
                    "status": entry.status,
                    "info": model_info_to_dict(entry),
                }
            if wait_kind == "load" and entry.status in {"loaded", "sleeping"}:
                return {
                    "ok": True,
                    "model": model_id,
                    "wait_seconds": int(elapsed),
                    "via": "catalog",
                    "status": entry.status,
                    "info": model_info_to_dict(entry),
                }
            if entry.progress and on_progress:
                on_progress(
                    {
                        "model": model_id,
                        "via": "catalog",
                        "wait_seconds": int(elapsed),
                        "status": entry.status,
                        "progress": entry.progress,
                    }
                )
        if on_progress:
            on_progress(
                {
                    "model": model_id,
                    "wait_seconds": int(elapsed),
                    "poll_interval": poll_interval,
                    "status": entry.status if entry else None,
                    "via": "catalog",
                }
            )
        await asyncio.sleep(poll_interval)


async def wait_for_model_download(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    poll_interval: float = 10.0,
    timeout: float = 7200.0,
    use_sse: bool = True,
    abort_on_cancel: bool = False,
) -> dict[str, Any]:
    """Wait for a router HF download to finish via SSE and/or catalog polling."""
    return await _wait_for_router_model(
        session,
        backend,
        model_id,
        wait_kind="download",
        cancel_check=cancel_check,
        on_progress=on_progress,
        poll_interval=poll_interval,
        timeout=timeout,
        use_sse=use_sse,
        abort_on_cancel=abort_on_cancel,
    )


async def wait_for_model_load(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    poll_interval: float = 5.0,
    timeout: float = 1200.0,
    use_sse: bool = True,
    abort_on_cancel: bool = False,
) -> dict[str, Any]:
    """Wait for a router model load to finish via SSE and/or catalog polling."""
    return await _wait_for_router_model(
        session,
        backend,
        model_id,
        wait_kind="load",
        cancel_check=cancel_check,
        on_progress=on_progress,
        poll_interval=poll_interval,
        timeout=timeout,
        use_sse=use_sse,
        abort_on_cancel=abort_on_cancel,
    )


async def load_model_with_progress(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    capabilities: ServerCapabilities | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    timeout: float = 1200.0,
    abort_on_cancel: bool = False,
) -> dict[str, Any]:
    """Load a model and wait for loaded status with SSE/catalog progress."""
    caps = capabilities or await probe_server(session, backend)
    if model_id in caps.loaded_models:
        return {
            "model": model_id,
            "ok": True,
            "skipped": True,
            "reason": "already loaded",
        }
    result = await load_model(session, backend, model_id)
    if not result.get("ok"):
        return result
    wait = await wait_for_model_load(
        session,
        backend,
        model_id,
        cancel_check=cancel_check,
        on_progress=on_progress,
        timeout=timeout,
        use_sse=caps.models_download_via_api,
        abort_on_cancel=abort_on_cancel,
    )
    return {**result, **wait}


async def download_model_on_router(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    capabilities: ServerCapabilities | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    timeout: float = 7200.0,
    abort_on_cancel: bool = False,
) -> dict[str, Any]:
    """Download a Hugging Face model through llama.cpp router HTTP APIs."""
    caps = capabilities or await probe_server(session, backend)
    if not router_supports_hf_download(caps):
        return {
            "ok": False,
            "model": model_id,
            "unsupported": True,
            "error": "Router HF download API not available.",
        }

    if model_id in caps.models:
        for item in caps.model_details:
            if item.model_id == model_id and _catalog_status_ready(item.status):
                return {
                    "ok": True,
                    "model": model_id,
                    "already_present": True,
                    "status": item.status,
                }

    request = await request_model_download(session, backend, model_id)
    if not request.get("ok"):
        return request

    wait = await wait_for_model_download(
        session,
        backend,
        model_id,
        cancel_check=cancel_check,
        on_progress=on_progress,
        timeout=timeout,
        use_sse=caps.models_download_via_api,
        abort_on_cancel=abort_on_cancel,
    )
    if wait.get("ok"):
        wait["request_path"] = request.get("path")
        await refresh_models_catalog(session, backend)
    return wait


async def wait_for_model_on_server(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    model_id: str,
    *,
    cancel_check: Callable[[], bool] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    poll_interval: float = 10.0,
    timeout: float = 7200.0,
) -> dict[str, Any]:
    """Poll /v1/models until a model shows up (manual or remote download)."""
    started = time.monotonic()
    while True:
        if cancel_check and cancel_check():
            return {"ok": False, "cancelled": True, "model": model_id}
        elapsed = time.monotonic() - started
        if elapsed > timeout:
            return {
                "ok": False,
                "model": model_id,
                "error": (
                    f"Timed out after {int(timeout)}s waiting for model on server."
                ),
            }
        if await model_available_on_server(session, backend, model_id):
            return {"ok": True, "model": model_id, "wait_seconds": int(elapsed)}
        if on_progress:
            on_progress(
                {
                    "model": model_id,
                    "wait_seconds": int(elapsed),
                    "poll_interval": poll_interval,
                }
            )
        await asyncio.sleep(poll_interval)


async def apply_props_settings(
    session: aiohttp.ClientSession,
    backend: LlmBackend,
    settings: dict[str, str],
) -> list[dict[str, Any]]:
    """Attempt to apply llama.cpp server settings via POST /props."""
    root = server_root_from_base_url(backend.base_url)
    url = f"{root}/props"
    headers = _headers(backend)
    headers["Content-Type"] = "application/json"
    timeout = aiohttp.ClientTimeout(total=15)
    results: list[dict[str, Any]] = []
    for key, value in settings.items():
        entry = {"setting": key, "value": value, "ok": False}
        try:
            async with session.post(
                url,
                json={key: value},
                headers=headers,
                timeout=timeout,
            ) as response:
                body = await response.text()
                entry["status"] = response.status
                entry["ok"] = response.status in {200, 204}
                if not entry["ok"]:
                    entry["error"] = body[:200]
        except (TimeoutError, aiohttp.ClientError) as err:
            entry["error"] = str(err)
        results.append(entry)
    return results
