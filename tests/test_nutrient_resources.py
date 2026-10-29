import json
from pathlib import Path

from health_agent.content import (
    nutrient_resources,
    render_nutrient_resource,
    sync_nutrient_resources,
)


COMMITTED_NUTRIENT_RESOURCES = Path(__file__).parent.parent / "resources" / "nutrients"


def _write_nutrient(root: Path, slug: str, name: str, aliases: list[str], body: str) -> None:
    page = root / slug
    page.mkdir(parents=True)
    (page / "meta.json").write_text(
        json.dumps({"slug": slug, "name": name, "aliases": aliases}), encoding="utf-8"
    )
    (page / "response.md").write_text(body, encoding="utf-8")


def test_render_nutrient_resource_adds_title_and_aliases():
    rendered = render_nutrient_resource(
        {"name": "Zinc", "aliases": ["Zn", "zinc picolinate"]}, "## Benefits\n\nBody.\n\n"
    )

    assert rendered == "# Zinc\n\nAliases: Zn, zinc picolinate\n\n## Benefits\n\nBody.\n"


def test_render_nutrient_resource_omits_empty_aliases_and_body():
    assert render_nutrient_resource({"name": "Honey", "aliases": []}, "") == "# Honey\n\n\n"


def test_sync_nutrient_resources_writes_pages_and_removes_stale(tmp_path: Path):
    content_root = tmp_path / "content"
    _write_nutrient(content_root, "zinc", "Zinc", [], "Body.\n")
    dest = tmp_path / "resources" / "nutrients"
    dest.mkdir(parents=True)
    (dest / "removed.md").write_text("old", encoding="utf-8")

    count = sync_nutrient_resources(dest, root=content_root)

    assert count == 1
    assert sorted(p.name for p in dest.iterdir()) == ["zinc.md"]
    assert (dest / "zinc.md").read_text(encoding="utf-8") == "# Zinc\n\nBody.\n"


def test_committed_nutrient_resources_match_content():
    committed = {
        path.name: path.read_text(encoding="utf-8")
        for path in COMMITTED_NUTRIENT_RESOURCES.glob("*.md")
    }

    assert committed == nutrient_resources(), (
        "resources/nutrients is out of date; run `health-agent sync-nutrient-resources`."
    )
