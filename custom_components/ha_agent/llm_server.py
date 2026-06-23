"""Probe llama.cpp server capabilities beyond the OpenAI /v1 API."""

from __future__ import annotations

import json
import re
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
    raw: dict[str, Any] = field(default_factory=dict)


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
                {"model_id": item.model_id, "status": item.status}
                for item in self.model_details
            ],
            "router_role": self.router_role,
            "max_instances": self.max_instances,
            "models_autoload": self.models_autoload,
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
        return caps

    async def _fetch_models(
        self,
        backend: LlmBackend,
        headers: dict[str, str],
        timeout: aiohttp.ClientTimeout,
        caps: ServerCapabilities,
    ) -> None:
        url = f"{backend.base_url.rstrip('/')}/models"
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
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            status = _model_status(item)
            models.append(model_id)
            details.append(ModelInfo(model_id=model_id, status=status, raw=item))
            if status == "loaded":
                loaded.append(model_id)
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
