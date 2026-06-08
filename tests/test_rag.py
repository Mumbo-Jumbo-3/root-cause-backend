from pathlib import Path
from unittest.mock import patch

from health_agent.config import Settings
from health_agent.db.models import AgentResourceChunk
from health_agent.rag.ingest import chunk_document
from health_agent.rag.resources import filesystem_resource_manifest
from health_agent.rag.retriever import needs_reindex


def test_markdown_chunking_preserves_headers(tmp_resources, test_settings: Settings):
    """Markdown header metadata is captured in chunk metadata."""
    chunks = chunk_document(tmp_resources / "nutrition.md", test_settings)

    h1_values = {c.metadata.get("h1") for c in chunks}
    h2_values = {c.metadata.get("h2") for c in chunks}
    assert "Nutrition Guide" in h1_values
    assert "Hydration" in h2_values


def test_contextual_header_prepend(tmp_resources, test_settings: Settings):
    """Chunk page_content starts with the header path for context."""
    chunks = chunk_document(tmp_resources / "nutrition.md", test_settings)

    # Find a chunk from the Electrolytes subsection
    electrolyte_chunks = [c for c in chunks if "electrolyte" in c.page_content.lower()]
    assert electrolyte_chunks
    # Should have the header path prepended
    content = electrolyte_chunks[0].page_content
    assert "Nutrition Guide" in content
    assert "Hydration" in content


def test_title_and_author_extraction(tmp_resources, test_settings: Settings):
    """Title and author are extracted into chunk metadata."""
    chunks = chunk_document(tmp_resources / "peat_thyroid.md", test_settings)

    assert all(c.metadata["title"] == "Thyroid and Metabolism" for c in chunks)
    assert all(c.metadata["author"] == "Dr. Ray Peat" for c in chunks)


def test_txt_files_skip_markdown_splitting(tmp_resources, test_settings: Settings):
    """Plain text files are chunked without markdown header splitting."""
    chunks = chunk_document(tmp_resources / "exercise.txt", test_settings)

    assert len(chunks) >= 1
    assert all(c.metadata["source"] == "exercise.txt" for c in chunks)


def test_chunking_preserves_source_metadata(tmp_resources, test_settings: Settings):
    """Every chunk retains the source filename in metadata."""
    chunks = chunk_document(tmp_resources / "nutrition.md", test_settings)

    assert len(chunks) > 0
    assert all(c.metadata["source"] == "nutrition.md" for c in chunks)


def test_author_from_filename_prefix(tmp_resources, test_settings: Settings):
    """Author is inferred from filename prefix when no explicit author line."""
    # grimhood_ prefix file
    (tmp_resources / "grimhood_test.md").write_text(
        "# Test Article\n\nSome content about health."
    )
    chunks = chunk_document(tmp_resources / "grimhood_test.md", test_settings)
    assert all(c.metadata["author"] == "Grimhood" for c in chunks)


def test_needs_reindex_when_index_metadata_missing(test_settings: Settings):
    with patch("health_agent.rag.retriever._database_resource_manifest", return_value={}):
        assert needs_reindex(test_settings) is True


def test_needs_reindex_false_when_database_manifest_matches(
    tmp_resources, test_settings: Settings
):
    manifest = filesystem_resource_manifest(tmp_resources)
    with patch("health_agent.rag.retriever._database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is False


def test_needs_reindex_when_resource_content_changes(
    tmp_resources, test_settings: Settings
):
    manifest = filesystem_resource_manifest(tmp_resources)

    nutrition_file = tmp_resources / "nutrition.md"
    nutrition_file.write_text(nutrition_file.read_text() + "\nExtra guidance.")

    with patch("health_agent.rag.retriever._database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is True


def test_needs_reindex_when_resource_set_changes(tmp_resources, test_settings: Settings):
    manifest = filesystem_resource_manifest(tmp_resources)

    (tmp_resources / "new-guide.md").write_text("# New Guide\n\nFresh content.")
    with patch("health_agent.rag.retriever._database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is True


def test_needs_reindex_when_database_manifest_has_extra_resource(
    tmp_resources, test_settings: Settings
):
    manifest = filesystem_resource_manifest(tmp_resources)
    manifest["missing.md"] = "stale-hash"

    with patch("health_agent.rag.retriever._database_resource_manifest", return_value=manifest):
        assert needs_reindex(test_settings) is True


def test_resource_chunk_model_includes_generated_search_vector():
    search_vector = AgentResourceChunk.__table__.c["search_vector"]

    assert search_vector.computed is not None
    assert "to_tsvector" in str(search_vector.computed.sqltext)


def test_search_vector_migration_adds_gin_index():
    migration = Path("alembic/versions/20260524_01_resource_search_vector.py").read_text()

    assert "search_vector" in migration
    assert 'postgresql_using="gin"' in migration
