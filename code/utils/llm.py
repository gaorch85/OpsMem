from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


OPSMEM_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
_CLIENT_CACHE: Dict[Tuple[str, str], Any] = {}
_MODEL_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _load_model_config() -> Dict[str, Any]:
    global _MODEL_CONFIG_CACHE
    if _MODEL_CONFIG_CACHE is None:
        with OPSMEM_CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
            root_config = yaml.safe_load(f) or {}
        _MODEL_CONFIG_CACHE = root_config.get("model") or {}
    return _MODEL_CONFIG_CACHE


def get_current_model_name() -> str:
    cfg = _load_model_config()
    current_model = (os.environ.get("OPSMEM_MODEL_NAME") or cfg.get("current_model") or "").strip()
    if not current_model:
        raise ValueError(f"'current_model' is missing in {OPSMEM_CONFIG_PATH}")
    return current_model


def _resolve_model_spec(model_name: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    cfg = _load_model_config()
    resolved_name = (model_name or os.environ.get("OPSMEM_MODEL_NAME") or cfg.get("current_model") or "").strip()
    models = cfg.get("models", {}) or {}
    if not resolved_name:
        raise ValueError(f"No model name provided and 'current_model' is missing in {OPSMEM_CONFIG_PATH}")
    if resolved_name not in models:
        raise KeyError(f"Model '{resolved_name}' is not defined in {OPSMEM_CONFIG_PATH}")
    spec = models[resolved_name] or {}
    if not spec.get("base_url"):
        raise ValueError(f"Model '{resolved_name}' is missing 'base_url' in {OPSMEM_CONFIG_PATH}")
    if not spec.get("model"):
        raise ValueError(f"Model '{resolved_name}' is missing 'model' in {OPSMEM_CONFIG_PATH}")
    return resolved_name, spec


def _get_llm_client(base_url: str, api_key: str):
    cache_key = (base_url, api_key)
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]
    if OpenAI is None:
        raise ImportError("Missing dependency 'openai'. Install it with: pip install openai")
    client = OpenAI(api_key=api_key, base_url=base_url)
    _CLIENT_CACHE[cache_key] = client
    return client


def parse_json_response(raw: Any) -> Any:
    """Parse a JSON object from an LLM response, including fenced JSON blocks."""
    if isinstance(raw, (dict, list)):
        return raw
    if raw is None:
        raise ValueError("Empty response to parse as JSON")

    text = str(raw).strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fence_match:
        text = fence_match.group(1).strip()
    elif text.lower().startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch in ("{", "["):
                try:
                    obj, _ = decoder.raw_decode(text, idx=idx)
                    return obj
                except json.JSONDecodeError:
                    continue
        raise exc


@dataclass
class UsageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def update(self, meta: Dict[str, Any]) -> None:
        self.calls += 1
        self.prompt_tokens += meta.get("prompt_tokens", 0)
        self.completion_tokens += meta.get("completion_tokens", 0)
        self.total_tokens += meta.get("total_tokens", 0)


_usage_stats = UsageStats()


def _record_usage(meta: Dict[str, Any]) -> None:
    _usage_stats.update(meta)


def print_usage_summary() -> None:
    print(get_usage_summary())


def get_usage_summary() -> str:
    return (
        "[LLM-API] total | "
        f"calls: {_usage_stats.calls} | "
        f"prompt_tokens: {_usage_stats.prompt_tokens} | "
        f"completion_tokens: {_usage_stats.completion_tokens} | "
        f"total_tokens: {_usage_stats.total_tokens}"
    )


def llm(
    system_prompt: str,
    user_prompt: str,
    model_path: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    return_meta: bool = False,
) -> Any:
    model_name, model_spec = _resolve_model_spec(model_path)

    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt.strip()})

    resp = _get_llm_client(
        base_url=model_spec["base_url"],
        api_key=model_spec.get("api_key", "EMPTY"),
    ).chat.completions.create(
        model=model_spec["model"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choice = resp.choices[0]
    content = (choice.message.content or "").strip()
    usage = getattr(resp, "usage", None)

    meta = {
        "finish_reason": getattr(choice, "finish_reason", None),
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        "response": content,
    }

    _record_usage(meta)

    if return_meta:
        return content, meta
    return content




