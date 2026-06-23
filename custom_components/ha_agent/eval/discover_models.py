"""Web discovery of candidate GGUF models for eval trials."""

from __future__ import annotations

import json
import re
from typing import Any

import aiohttp

from ..config_helpers import LlmBackend
from ..const import LOGGER
from ..llm_client import LlmClient
from ..llm_server import ServerCapabilities
from .host_context import build_host_context
from .model_registry import ModelProposal

_HF_MODELS_API = "https://huggingface.co/api/models"
_HF_TREE_API = "https://huggingface.co/api/models/{repo_id}/tree/main"

_DISCOVERY_PROMPT = (
    "You recommend GGUF models for a Home Assistant voice agent on this hardware.\n\n"
    "Server capabilities:\n{capabilities_json}\n\n"
    "Host context:\n{host_context_json}\n\n"
    "Already on server (skip these):\n{existing_models}\n\n"
    "Hugging Face candidates:\n{candidates_json}\n\n"
    "Pick up to {max_models} NEW models to trial. Prefer models suited for:\n"
    "- chat: general Q&A\n"
    "- action: tool calling / device control\n"
    "- classifier: fast routing (smaller models ok)\n\n"
    "Return ONLY JSON:\n"
    "{{\n"
    '  "proposals": [\n'
    "    {{\n"
    '      "model_id": "router model id (repo:quant if applicable)",\n'
    '      "hf_repo": "org/repo",\n'
    '      "hf_filename": "file.gguf",\n'
    '      "source_url": "https://huggingface.co/...",\n'
    '      "reason": "why this fits the hardware and tasks",\n'
    '      "expected_benefit": "what should improve vs current models"\n'
    "    }}\n"
    "  ]\n"
    "}}"
)


def _extract_json(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def search_hf_gguf_models(
    session: aiohttp.ClientSession,
    *,
    query: str = "gguf",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Search Hugging Face for popular GGUF model repos."""
    params = {
        "search": query,
        "filter": "text-generation",
        "sort": "downloads",
        "direction": -1,
        "limit": limit,
    }
    try:
        async with session.get(
            _HF_MODELS_API,
            params=params,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status != 200:
                return []
            payload = await response.json()
    except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError) as err:
        LOGGER.debug("HF model search failed: %s", err)
        return []

    if not isinstance(payload, list):
        return []
    results: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            continue
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        tag_text = " ".join(str(tag) for tag in tags).lower()
        if "gguf" not in model_id.lower() and "gguf" not in tag_text:
            continue
        results.append(
            {
                "repo_id": model_id,
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "tags": tags[:8],
            }
        )
    return results


async def list_gguf_files(
    session: aiohttp.ClientSession,
    repo_id: str,
) -> list[str]:
    """Return GGUF filenames in a Hugging Face repo."""
    url = _HF_TREE_API.format(repo_id=repo_id)
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as response:
            if response.status != 200:
                return []
            payload = await response.json()
    except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []
    files: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        if isinstance(path, str) and path.lower().endswith(".gguf"):
            files.append(path)
    return sorted(files)


def _pick_quant_file(filenames: list[str]) -> str | None:
    if not filenames:
        return None
    preferred = (
        "Q4_K_M",
        "Q5_K_M",
        "IQ4_XS",
        "Q4_K_S",
        "Q5_K_S",
        "Q6_K",
        "Q8_0",
    )
    upper_names = {name: name.upper() for name in filenames}
    for quant in preferred:
        for name, upper in upper_names.items():
            if quant in upper:
                return name
    return filenames[0]


def _router_model_id(repo_id: str, filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    if ":" in repo_id:
        return repo_id
    return f"{repo_id}:{stem}"


async def enrich_hf_candidates(
    session: aiohttp.ClientSession,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a default GGUF filename and router id to HF search hits."""
    enriched: list[dict[str, Any]] = []
    for item in candidates:
        repo_id = str(item.get("repo_id") or "")
        if not repo_id:
            continue
        files = await list_gguf_files(session, repo_id)
        filename = _pick_quant_file(files)
        if not filename:
            continue
        enriched.append(
            {
                **item,
                "hf_repo": repo_id,
                "hf_filename": filename,
                "router_model_id": _router_model_id(repo_id, filename),
                "source_url": f"https://huggingface.co/{repo_id}",
            }
        )
    return enriched


def _fallback_proposals(
    candidates: list[dict[str, Any]],
    *,
    max_models: int,
    existing_models: set[str],
    skip_ids: set[str],
) -> list[ModelProposal]:
    proposals: list[ModelProposal] = []
    for item in candidates:
        model_id = str(item.get("router_model_id") or "")
        if not model_id or model_id in existing_models or model_id in skip_ids:
            continue
        proposals.append(
            ModelProposal(
                model_id=model_id,
                source_url=str(item.get("source_url") or ""),
                reason="Popular GGUF model from Hugging Face search.",
                expected_benefit="May improve eval scores for chat or action tasks.",
                hf_repo=str(item.get("hf_repo") or ""),
                hf_filename=str(item.get("hf_filename") or ""),
            )
        )
        if len(proposals) >= max_models:
            break
    return proposals


async def propose_models_from_web(
    session: aiohttp.ClientSession,
    llm: LlmClient,
    backend: LlmBackend,
    *,
    capabilities: ServerCapabilities,
    max_models: int = 3,
    skip_model_ids: set[str] | None = None,
) -> list[ModelProposal]:
    """Search Hugging Face and rank candidate models for this setup."""
    skip_ids = set(skip_model_ids or [])
    existing_models = set(capabilities.models)
    search_terms = ("gguf agent", "gguf instruct", "gguf tool")
    raw_candidates: list[dict[str, Any]] = []
    seen_repos: set[str] = set()
    for term in search_terms:
        for item in await search_hf_gguf_models(session, query=term, limit=12):
            repo_id = str(item.get("repo_id") or "")
            if repo_id in seen_repos:
                continue
            seen_repos.add(repo_id)
            raw_candidates.append(item)
        if len(raw_candidates) >= max_models * 4:
            break

    candidates = await enrich_hf_candidates(session, raw_candidates)
    if not candidates:
        return []

    prompt = _DISCOVERY_PROMPT.format(
        capabilities_json=json.dumps(capabilities.summary(), ensure_ascii=False),
        host_context_json=json.dumps(
            build_host_context(capabilities),
            ensure_ascii=False,
        ),
        existing_models=json.dumps(sorted(existing_models), ensure_ascii=False),
        candidates_json=json.dumps(candidates[:20], ensure_ascii=False),
        max_models=max_models,
    )
    try:
        result = await llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are a local LLM deployment advisor.",
                },
                {"role": "user", "content": prompt},
            ],
            LlmBackend(
                base_url=backend.base_url,
                model=backend.model,
                api_key=backend.api_key,
                max_tokens=1200,
                temperature=0.2,
                timeout=backend.timeout,
                thinking_level="off",
            ),
        )
        parsed = _extract_json(result.content or "")
    except Exception as err:
        LOGGER.warning("Model discovery LLM ranking failed: %s", err)
        parsed = None

    if not parsed:
        return _fallback_proposals(
            candidates,
            max_models=max_models,
            existing_models=existing_models,
            skip_ids=skip_ids,
        )

    proposals: list[ModelProposal] = []
    for item in parsed.get("proposals") or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("model_id") or "").strip()
        hf_repo = str(item.get("hf_repo") or "").strip()
        hf_filename = str(item.get("hf_filename") or "").strip()
        if not model_id or not hf_repo or not hf_filename:
            continue
        if model_id in existing_models or model_id in skip_ids:
            continue
        proposals.append(
            ModelProposal(
                model_id=model_id,
                source_url=str(item.get("source_url") or f"https://huggingface.co/{hf_repo}"),
                reason=str(item.get("reason") or "Recommended by discovery agent."),
                expected_benefit=str(item.get("expected_benefit") or ""),
                hf_repo=hf_repo,
                hf_filename=hf_filename,
            )
        )
        if len(proposals) >= max_models:
            break

    if proposals:
        return proposals
    return _fallback_proposals(
        candidates,
        max_models=max_models,
        existing_models=existing_models,
        skip_ids=skip_ids,
    )
