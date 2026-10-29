"""Loaders for pre-defined featured queries and nutrient pages.

Content is stored as `meta.json` + `response.md` per slug under
`src/health_agent/content/{featured,nutrients}/{slug}/`. The frontend pulls
these via the `/featured` and `/nutrients` endpoints so editing a response no
longer requires a frontend redeploy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from health_agent.nutrient_prompts import build_prompt

CONTENT_ROOT = Path(__file__).parent / "content"
_FEATURED_ROOT = CONTENT_ROOT / "featured"
_NUTRIENTS_ROOT = CONTENT_ROOT / "nutrients"


def _with_nutrient_prompt(meta: dict[str, Any]) -> dict[str, Any]:
    name = meta.get("name") or meta.get("slug") or ""
    meta["prompt"] = build_prompt(name)
    return meta


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


def list_nutrients() -> list[dict[str, Any]]:
    return [_with_nutrient_prompt(meta) for meta in _list_metas(_NUTRIENTS_ROOT)]


def get_nutrient(slug: str) -> dict[str, Any] | None:
    full = _load_full(_NUTRIENTS_ROOT, slug)
    return _with_nutrient_prompt(full) if full is not None else None


def render_nutrient_resource(meta: dict[str, Any], response_markdown: str) -> str:
    """Render a nutrient page as the markdown document the RAG index ingests."""
    header = f"# {meta['name']}\n\n"
    aliases = meta.get("aliases") or []
    if aliases:
        header += f"Aliases: {', '.join(aliases)}\n\n"
    return header + response_markdown.strip() + "\n"


def nutrient_resources(root: Path = _NUTRIENTS_ROOT) -> dict[str, str]:
    """Map `{slug}.md` filenames to rendered RAG documents for every nutrient page."""
    resources: dict[str, str] = {}
    for meta in _list_metas(root):
        full = _load_full(root, meta["slug"])
        if full is not None:
            resources[f"{meta['slug']}.md"] = render_nutrient_resource(
                full, full["response_markdown"]
            )
    return resources


def sync_nutrient_resources(dest: Path, root: Path = _NUTRIENTS_ROOT) -> int:
    """Write rendered nutrient documents to `dest`, removing ones with no source page."""
    resources = nutrient_resources(root)
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.md"):
        if stale.name not in resources:
            stale.unlink()
    for name, text in resources.items():
        (dest / name).write_text(text, encoding="utf-8")
    return len(resources)
