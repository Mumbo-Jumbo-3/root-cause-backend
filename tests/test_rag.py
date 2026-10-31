import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from health_agent.config import Settings
from health_agent.db.models import AgentResourceChunk
from health_agent.content import nutrient_resources
from health_agent.rag.ingest import _embed_texts, chunk_document, ingest_resources
from health_agent.rag.resources import (
    expected_index_manifest,
    filesystem_resource_manifest,
    hash_text,
    needs_reindex,
    resource_files,
)

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"


def _chunk_file(path: Path, settings: Settings):
    return chunk_document(path.read_text(encoding="utf-8"), path.name, settings)


def test_markdown_chunking_preserves_headers(tmp_resources, test_settings: Settings):
    """Markdown header metadata is captured in chunk metadata."""
    chunks = _chunk_file(tmp_resources / "nutrition.md", test_settings)

    h1_values = {c.metadata.get("h1") for c in chunks}
    h2_values = {c.metadata.get("h2") for c in chunks}
    assert "Nutrition Guide" in h1_values
    assert "Hydration" in h2_values


def test_contextual_header_prepend(tmp_resources, test_settings: Settings):
    """Chunk page_content starts with the header path for context."""
    chunks = _chunk_file(tmp_resources / "nutrition.md", test_settings)

    # Find a chunk from the Electrolytes subsection
    electrolyte_chunks = [c for c in chunks if "electrolyte" in c.page_content.lower()]
    assert electrolyte_chunks
    # Should have the header path prepended
    content = electrolyte_chunks[0].page_content
    assert "Nutrition Guide" in content
    assert "Hydration" in content


def test_title_and_author_extraction(tmp_resources, test_settings: Settings):
    """Title and author are extracted into chunk metadata."""
    chunks = _chunk_file(tmp_resources / "peat_thyroid.md", test_settings)

    assert all(c.metadata["title"] == "Thyroid and Metabolism" for c in chunks)
    assert all(c.metadata["author"] == "Dr. Ray Peat" for c in chunks)


def test_txt_files_skip_markdown_splitting(tmp_resources, test_settings: Settings):
    """Plain text files are chunked without markdown header splitting."""
    chunks = _chunk_file(tmp_resources / "exercise.txt", test_settings)

    assert len(chunks) >= 1
    assert all(c.metadata["source"] == "exercise.txt" for c in chunks)


def test_chunking_preserves_source_metadata(tmp_resources, test_settings: Settings):
    """Every chunk retains the source filename in metadata."""
    chunks = _chunk_file(tmp_resources / "nutrition.md", test_settings)

    assert len(chunks) > 0
    assert all(c.metadata["source"] == "nutrition.md" for c in chunks)


def test_author_from_filename_prefix(tmp_resources, test_settings: Settings):
    """Author is inferred from filename prefix when no explicit author line."""
    # grimhood_ prefix file
    (tmp_resources / "grimhood_test.md").write_text(
        "# Test Article\n\nSome content about health."
    )
    chunks = _chunk_file(tmp_resources / "grimhood_test.md", test_settings)
    assert all(c.metadata["author"] == "Grimhood" for c in chunks)


def test_title_never_comes_from_a_body_line(tmp_resources, test_settings: Settings):
    """Without an H1, the title is built from the filename, not the first line."""
    (tmp_resources / "ferman_system_3.md").write_text(
        "DISCLAIMER: This PDF does not provide medical advice.\n\nSome lesson text."
    )
    chunks = _chunk_file(tmp_resources / "ferman_system_3.md", test_settings)
    assert {c.metadata["title"] for c in chunks} == {"System 3"}
    assert {c.metadata["author"] for c in chunks} == {"George Ferman"}


def test_title_is_first_h1_even_after_preamble(tmp_resources, test_settings: Settings):
    (tmp_resources / "peat_letter.md").write_text(
        "a fragment of the previous article.\n\n# **Failures Of Immunity**\n\nBody."
    )
    chunks = _chunk_file(tmp_resources / "peat_letter.md", test_settings)
    assert {c.metadata["title"] for c in chunks} == {"Failures Of Immunity"}


def test_author_line_used_for_unprefixed_files(tmp_resources, test_settings: Settings):
    (tmp_resources / "guest.md").write_text("# Guest Post\n\n**Author: Jane Doe**\n\nBody.")
    chunks = _chunk_file(tmp_resources / "guest.md", test_settings)
    assert {c.metadata["author"] for c in chunks} == {"Jane Doe"}


_BOILERPLATE = re.compile(
    r"This PDF does not provide medical advice"
    r"|This program was created by"
    r"|\(lesson \d+\) written by George"
    r"|^\W*Sold to\W*$",
    re.IGNORECASE | re.MULTILINE,
)


def test_archive_resources_have_titles_and_no_boilerplate():
    """Every curated resource names itself with an H1 and carries no PDF boilerplate."""
    files = resource_files(RESOURCES_DIR)
    assert files

    untitled = [
        f.name
        for f in files
        if not re.search(r"^# +\S", f.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    with_boilerplate = [
        f.name for f in files if _BOILERPLATE.search(f.read_text(encoding="utf-8"))
    ]
    assert untitled == []
    assert with_boilerplate == []


def test_needs_reindex_when_index_metadata_missing(test_settings: Settings):
    with patch("health_agent.rag.resources.database_resource_manifest", return_value={}):
        assert needs_reindex(test_settings) is True


def test_needs_reindex_false_when_database_manifest_matches(
    tmp_resources, test_settings: Settings
):
    manifest = expected_index_manifest(test_settings)
    with patch("health_agent.rag.resources.database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is False


def test_needs_reindex_when_resource_content_changes(
    tmp_resources, test_settings: Settings
):
    manifest = expected_index_manifest(test_settings)

    nutrition_file = tmp_resources / "nutrition.md"
    nutrition_file.write_text(nutrition_file.read_text() + "\nExtra guidance.")

    with patch("health_agent.rag.resources.database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is True


def test_needs_reindex_when_resource_set_changes(tmp_resources, test_settings: Settings):
    manifest = expected_index_manifest(test_settings)

    (tmp_resources / "new-guide.md").write_text("# New Guide\n\nFresh content.")
    with patch("health_agent.rag.resources.database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is True


def test_needs_reindex_when_database_manifest_has_extra_resource(
    tmp_resources, test_settings: Settings
):
    manifest = expected_index_manifest(test_settings)
    manifest["missing.md"] = "stale-hash"

    with patch("health_agent.rag.resources.database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is True


def test_pipeline_change_marks_index_stale(tmp_resources, test_settings: Settings):
    """Re-chunking or a new INGEST_VERSION re-ingests even when no file changed."""
    stored = expected_index_manifest(test_settings)
    rechunked = test_settings.model_copy(update={"chunk_size": 300})

    with patch("health_agent.rag.resources.database_resource_manifest", return_value=stored):
        assert needs_reindex(rechunked) is True
    with patch("health_agent.rag.resources.INGEST_VERSION", 2):
        assert expected_index_manifest(test_settings) != stored


def test_oracle_corpus_manifest_hashes_content_only(tmp_resources, test_settings: Settings):
    """The oracle's corpus hash moves only when a resource's text does."""
    nutrition = (tmp_resources / "nutrition.md").read_text(encoding="utf-8")

    assert filesystem_resource_manifest(tmp_resources)["nutrition.md"] == hash_text(nutrition)
    assert expected_index_manifest(test_settings)["nutrition.md"] != hash_text(nutrition)


def test_ingest_refuses_to_empty_the_index(tmp_path: Path, test_settings: Settings):
    """A missing or empty resources dir must not read as "every resource was deleted"."""
    settings = test_settings.model_copy(update={"resources_dir": tmp_path / "missing"})

    with patch("health_agent.rag.ingest.get_session_factory") as session_factory:
        with pytest.raises(RuntimeError, match="refusing to empty the index"):
            ingest_resources(settings)
    session_factory.assert_not_called()


def test_embed_texts_batches_documents_into_float32_rows(test_settings: Settings):
    client = MagicMock()
    client.embed.side_effect = lambda batch, **_: MagicMock(
        embeddings=[[float(len(text))] * 1024 for text in batch]
    )

    with (
        patch("health_agent.rag.ingest.voyage_client", return_value=client),
        patch("health_agent.rag.ingest._EMBED_BATCH_SIZE", 2),
    ):
        vectors = _embed_texts(["a", "bb", "ccc"], test_settings)

    assert vectors.dtype == np.float32
    assert vectors.shape == (3, 1024)
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0]
    assert client.embed.call_count == 2
    client.embed.assert_called_with(
        ["ccc"], model="voyage-4-large", input_type="document", output_dimension=1024
    )


def test_nutrient_resources_match_their_source_pages():
    """resources/nutrients/ is generated: run `health-agent sync-nutrient-resources` after editing a nutrient page."""
    on_disk = {
        path.name: path.read_text(encoding="utf-8")
        for path in (RESOURCES_DIR / "nutrients").glob("*.md")
    }
    assert on_disk == nutrient_resources()


def test_resource_chunk_model_includes_generated_search_vector():
    search_vector = AgentResourceChunk.__table__.c["search_vector"]

    assert search_vector.computed is not None
    assert "to_tsvector" in str(search_vector.computed.sqltext)


def test_search_vector_migration_adds_gin_index():
    migration = Path("alembic/versions/20260524_01_resource_search_vector.py").read_text()

    assert "search_vector" in migration
    assert 'postgresql_using="gin"' in migration
