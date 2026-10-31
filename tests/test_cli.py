from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from health_agent.cli import app


def test_predeploy_migrates_then_ingests():
    calls = []
    stats = MagicMock(
        added_resources=0, updated_resources=0, deleted_resources=0, chunk_rows_written=0
    )
    with (
        patch.dict("os.environ", {"DATABASE_URL": "postgresql://localhost/test"}),
        patch("alembic.command.upgrade", side_effect=lambda cfg, rev: calls.append(("upgrade", rev))),
        patch(
            "health_agent.rag.ingest.ingest_resources",
            side_effect=lambda settings, force: calls.append(("ingest", force)) or stats,
        ),
    ):
        result = CliRunner().invoke(app, ["predeploy"])

    assert result.exit_code == 0, result.output
    assert calls == [("upgrade", "head"), ("ingest", False)]
