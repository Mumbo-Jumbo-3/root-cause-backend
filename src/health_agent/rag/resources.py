from hashlib import sha256
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from health_agent.config import Settings
from health_agent.db import AgentResource, get_session_factory

# Bump when chunking or title/author extraction changes, so the next ingest re-embeds everything.
INGEST_VERSION = 1


def resource_files(resource_path: Path) -> list[Path]:
    files = list(resource_path.glob("**/*.txt")) + list(resource_path.glob("**/*.md"))
    return sorted(files, key=lambda path: str(path.relative_to(resource_path)))


def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def resource_hash(text: str, settings: Settings) -> str:
    """Hash a resource's text together with everything else that shapes its chunks and vectors.

    Ingest stores this hash, so changing the pipeline marks every resource stale, just as
    editing a file marks that one stale.
    """
    pipeline = (
        f"v{INGEST_VERSION}|{settings.embedding_model}|"
        f"{settings.chunk_size}|{settings.chunk_overlap}"
    )
    return hash_text(f"{pipeline}\n{text}")


def filesystem_resource_manifest(resource_path: Path) -> dict[str, str]:
    """Map each resource to a hash of its content alone."""
    return {
        str(file_path.relative_to(resource_path)): hash_text(file_path.read_text(encoding="utf-8"))
        for file_path in resource_files(resource_path)
    }


def expected_index_manifest(settings: Settings) -> dict[str, str]:
    """Map each resource to the hash ingest would store for it."""
    resource_path = settings.resources_dir
    return {
        str(file_path.relative_to(resource_path)): resource_hash(
            file_path.read_text(encoding="utf-8"), settings
        )
        for file_path in resource_files(resource_path)
    }


def database_resource_manifest(session: Session) -> dict[str, str]:
    """Map each indexed resource to the hash ingest stored for it."""
    rows = session.execute(select(AgentResource.source_path, AgentResource.content_hash)).all()
    return {source_path: content_hash for source_path, content_hash in rows}


def needs_reindex(settings: Settings) -> bool:
    if not settings.database_url.strip():
        return True

    try:
        with get_session_factory(settings)() as session:
            stored_manifest = database_resource_manifest(session)
    except SQLAlchemyError:
        return True

    return expected_index_manifest(settings) != stored_manifest
