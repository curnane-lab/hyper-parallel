# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Schema adapters that normalize heterogeneous parquet records for VLM training.

Each adapter converts one raw record (a dict of parquet columns) into the
canonical VLM record format::

    {"messages": [{"role": ..., "content": ...}], "images": [...]}

Adapters are registered by name and selected in the dataset YAML via
``data_config.schema_adapter``. The ``auto`` adapter inspects the record
columns and picks the first matching adapter.
"""

import base64
import io
from typing import Any, Callable, TypeAlias

from PIL import Image

AdapterFn: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]

_ADAPTER_REGISTRY: dict[str, AdapterFn] = {}


def register_adapter(name: str):
    """Decorator that registers a schema-adapter function under ``name``."""

    def _wrap(fn: AdapterFn) -> AdapterFn:
        if name in _ADAPTER_REGISTRY:
            raise ValueError(f"VLM schema adapter {name!r} is already registered")
        _ADAPTER_REGISTRY[name] = fn
        return fn

    return _wrap


def get_adapter(name: str) -> AdapterFn:
    """Return the adapter registered under ``name``."""
    if name not in _ADAPTER_REGISTRY:
        known = ", ".join(sorted(_ADAPTER_REGISTRY))
        raise ValueError(f"Unknown VLM schema adapter {name!r}; known adapters: {known}")
    return _ADAPTER_REGISTRY[name]


def list_adapters() -> list[str]:
    """Return the names of all registered adapters."""
    return sorted(_ADAPTER_REGISTRY)


def _encode_image_bytes(raw: Any) -> str:
    """Encode one raw image (bytes, BinaryScalar, dict, or PIL) as a data URI."""
    if isinstance(raw, dict):
        raw = raw.get("bytes")
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    if isinstance(raw, Image.Image):
        buf = io.BytesIO()
        raw.save(buf, format="PNG")
        raw = buf.getvalue()
    if isinstance(raw, str) and raw.startswith("data:image/"):
        return raw
    if isinstance(raw, str):
        return "data:image/png;base64," + raw
    if isinstance(raw, (bytes, bytearray)):
        return "data:image/png;base64," + base64.b64encode(bytes(raw)).decode()
    raise TypeError(f"Cannot encode image from type {type(raw).__name__}")


def _normalize_image_list(raw: Any) -> list[str]:
    """Normalize any images-like column into a list of data URIs."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if hasattr(raw, "as_py"):
        raw = raw.as_py()
    if not isinstance(raw, list):
        raw = [raw]
    return [_encode_image_bytes(item) for item in raw]


@register_adapter("qwen_vl")
def _adapter_qwen_vl(record: dict[str, Any]) -> dict[str, Any]:
    """Pass through the canonical ``messages`` + ``images`` format."""
    return {"messages": record["messages"], "images": _normalize_image_list(record.get("images"))}


@register_adapter("llava")
def _adapter_llava(record: dict[str, Any]) -> dict[str, Any]:
    """Map LLaVA-style ``conversations`` + ``image`` columns."""
    messages = []
    for turn in record.get("conversations", []):
        role = "user" if turn.get("from") == "human" else "assistant"
        messages.append({"role": role, "content": turn.get("value", "")})
    images = record.get("image", [])
    if not isinstance(images, list):
        images = [images]
    return {"messages": messages, "images": _normalize_image_list(images)}


def _build_otsl_target(otsl_tokens: Any, cells: Any) -> str:
    """Reconstruct the OTSL text target from structural tokens and cell texts."""
    if hasattr(otsl_tokens, "as_py"):
        otsl_tokens = otsl_tokens.as_py()
    if hasattr(cells, "as_py"):
        cells = cells.as_py()
    if cells and isinstance(cells[0], list):
        cells = cells[0]
    cell_texts = []
    for cell in cells or []:
        tokens = cell.get("tokens", []) if isinstance(cell, dict) else []
        cell_texts.append("".join(str(token) for token in tokens))
    cell_iter = iter(cell_texts)
    parts = []
    for tok in otsl_tokens or []:
        tok = str(tok)
        if tok == "fcel":
            text = next(cell_iter, "")
            parts.append(f"<fcel> {text}" if text else "<fcel>")
        elif tok in ("ched", "rhed", "srow"):
            text = next(cell_iter, "")
            parts.append(f"<{tok}> {text}" if text else f"<{tok}>")
        elif tok:
            parts.append(f"<{tok}>")
    return " ".join(parts)


@register_adapter("fintabnet_otsl")
def _adapter_fintabnet_otsl(record: dict[str, Any]) -> dict[str, Any]:
    """Map FinTabNet-OTSL columns into an image-to-OTSL conversation."""
    target = _build_otsl_target(record.get("otsl"), record.get("cells"))
    if not target.strip():
        raise ValueError("record has an empty OTSL target")
    messages = [
        {"role": "user", "content": "<image>\nRecognize the table structure and content in OTSL format."},
        {"role": "assistant", "content": target},
    ]
    return {"messages": messages, "images": _normalize_image_list(record.get("image"))}


@register_adapter("image_text_pair")
def _adapter_image_text_pair(record: dict[str, Any]) -> dict[str, Any]:
    """Map flat ``image`` + ``text`` (or ``caption``) columns."""
    text = record.get("text") or record.get("caption") or record.get("answer") or ""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text)
    messages = [
        {"role": "user", "content": "<image>\nDescribe the image."},
        {"role": "assistant", "content": str(text)},
    ]
    return {"messages": messages, "images": _normalize_image_list(record.get("image"))}


def detect_adapter(record: dict[str, Any]) -> str:
    """Auto-detect the best adapter for one raw record."""
    columns = set(record.keys())
    if "messages" in columns and ("images" in columns or "image" in columns):
        return "qwen_vl"
    if "conversations" in columns:
        return "llava"
    if "otsl" in columns and ("image" in columns or "images" in columns):
        return "fintabnet_otsl"
    if ("image" in columns or "images" in columns) and ("text" in columns or "caption" in columns):
        return "image_text_pair"
    raise ValueError(
        f"Cannot auto-detect VLM schema for columns {sorted(columns)}; "
        f"known adapters: {list_adapters()}"
    )


def adapt_record(record: dict[str, Any], adapter: str) -> dict[str, Any]:
    """Apply one adapter (or auto-detect when ``adapter == "auto"``)."""
    if adapter == "auto":
        adapter = detect_adapter(record)
    return get_adapter(adapter)(record)


__all__ = [
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "detect_adapter",
    "adapt_record",
]
