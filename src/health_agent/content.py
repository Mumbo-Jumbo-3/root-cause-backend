"""Loaders for pre-defined featured queries and ingestible pages.

Content is stored as `meta.json` + `response.md` per slug under
`src/health_agent/content/{featured,ingestibles}/{slug}/`. The frontend pulls
these via the `/featured` and `/ingestibles` endpoints so editing a response no
longer requires a frontend redeploy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTENT_ROOT = Path(__file__).parent / "content"
_FEATURED_ROOT = CONTENT_ROOT / "featured"
_INGESTIBLES_ROOT = CONTENT_ROOT / "ingestibles"


def _load_meta(root: Path, slug: str) -> dict[str, Any] | None:
    meta_path = root / slug / "meta.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_full(root: Path, slug: str) -> dict[str, Any] | None:
    meta = _load_meta(root, slug)
    if meta is None:
        return None
    response_path = root / slug / "response.md"
    if not response_path.is_file():
        return None
    meta["response_markdown"] = response_path.read_text(encoding="utf-8")
    return meta


def _list_metas(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        meta = _load_meta(root, entry.name)
        if meta is not None:
            items.append(meta)
    return items


def list_featured() -> list[dict[str, Any]]:
    return _list_metas(_FEATURED_ROOT)


def get_featured(slug: str) -> dict[str, Any] | None:
    return _load_full(_FEATURED_ROOT, slug)


def list_ingestibles() -> list[dict[str, Any]]:
    return _list_metas(_INGESTIBLES_ROOT)


def get_ingestible(slug: str) -> dict[str, Any] | None:
    return _load_full(_INGESTIBLES_ROOT, slug)
