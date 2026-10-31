from dataclasses import dataclass
import re
import uuid
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from sqlalchemy import delete, func, insert, select

from health_agent.config import Settings
from health_agent.db import AgentResource, AgentResourceChunk, get_session_factory
from health_agent.db.models import EMBEDDING_DIMENSIONS
from health_agent.models import voyage_client
from health_agent.rag.resources import (
    database_resource_manifest,
    hash_text,
    resource_files,
    resource_hash,
)

_AUTHOR_PREFIXES = {
    "peat_": "Dr. Ray Peat",
    "grimhood_": "Grimhood",
    "ferman_": "George Ferman",
}

# Held for the write transaction, so a deploy and a manual run never write at once.
_INGEST_LOCK_KEY = 7_270_301
_EMBED_BATCH_SIZE = 128
# Each vector becomes ~20 KB of text on its way to pgvector; inserting in slices keeps that bounded.
_INSERT_BATCH_SIZE = 500


@dataclass
class IngestStats:
    added_resources: int
    updated_resources: int
    deleted_resources: int
    chunk_rows_written: int


@dataclass
class ResourceFileRecord:
    source_path: str
    source_name: str
    raw_content: str
    content_hash: str


_H1_RE = re.compile(r"^# +(.+?)\s*$", re.MULTILINE)
_AUTHOR_LINE_RE = re.compile(r"^[#*_\s]*author:\s*(.+?)[*_\s]*$", re.IGNORECASE | re.MULTILINE)


def _extract_title(text: str, filename: str) -> str:
    """Use the document's first H1, or a title built from the filename.

    Titles are shown to users and to Claude, so they never come from a body
    line: in PDF-extracted sources the first line is often a disclaimer or byline.
    """
    match = _H1_RE.search(text)
    if match:
        title = match.group(1).strip("*_ ")
        if title:
            return title

    stem = Path(filename).stem
    for prefix in _AUTHOR_PREFIXES:
        stem = stem.removeprefix(prefix)
    return re.sub(r"[_-]+", " ", stem).strip().title() or "Untitled"


def _extract_author(text: str, filename: str) -> str:
    """Name the author from the collection's filename prefix or an `Author:` line."""
    for prefix, author in _AUTHOR_PREFIXES.items():
        if filename.startswith(prefix):
            return author

    head = "\n".join(text.splitlines()[:20])
    match = _AUTHOR_LINE_RE.search(head)
    return match.group(1) if match else ""


def chunk_document(text: str, filename: str, settings: Settings) -> list[Document]:
    """Chunk a single file using markdown-aware splitting with contextual headers."""
    title = _extract_title(text, filename)
    author = _extract_author(text, filename)
    base_metadata = {"source": filename, "title": title, "author": author}

    # Stage A: markdown header splitting for .md files
    if Path(filename).suffix == ".md":
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=True,
        )
        md_chunks = md_splitter.split_text(text)
    else:
        md_chunks = [Document(page_content=text, metadata={})]

    # Stage B: sub-split large sections, respecting header prefix budget
    final_chunks: list[Document] = []
    for chunk in md_chunks:
        meta = {**base_metadata, **chunk.metadata}

        # Build contextual header path
        header_parts = [meta[h] for h in ("h1", "h2", "h3") if meta.get(h)]
        header_path = " > ".join(header_parts)
        meta["header_path"] = header_path or None
        prefix = f"{header_path}\n\n" if header_path else ""

        body = chunk.page_content
        target_size = settings.chunk_size - len(prefix)

        if len(prefix) + len(body) > settings.chunk_size:
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max(target_size, 200),
                chunk_overlap=settings.chunk_overlap,
            )
            for sub_text in sub_splitter.split_text(body):
                final_chunks.append(Document(page_content=prefix + sub_text, metadata=meta))
        else:
            final_chunks.append(Document(page_content=prefix + body, metadata=meta))

    return final_chunks


def _resource_record(
    file_path: Path, resource_path: Path, settings: Settings
) -> ResourceFileRecord:
    raw_content = file_path.read_text(encoding="utf-8")
    return ResourceFileRecord(
        source_path=str(file_path.relative_to(resource_path)),
        source_name=file_path.name,
        raw_content=raw_content,
        content_hash=resource_hash(raw_content, settings),
    )


def _embed_texts(texts: list[str], settings: Settings) -> np.ndarray:
    """Embed chunks as float32 rows, so a full rebuild fits in a small instance's memory."""
    client = voyage_client(settings.voyage_api_key)
    vectors = np.empty((len(texts), EMBEDDING_DIMENSIONS), dtype=np.float32)
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        result = client.embed(
            batch,
            model=settings.embedding_model,
            input_type="document",
            output_dimension=EMBEDDING_DIMENSIONS,
        )
        vectors[start : start + len(batch)] = result.embeddings
    return vectors


def ingest_resources(settings: Settings, force: bool = False) -> IngestStats:
    """Make the index match `resources_dir`, embedding only the resources that changed.

    All embedding happens before the write transaction opens, so a failed run leaves the
    index as it was, and the write that replaces it takes seconds.
    """
    resource_path = settings.resources_dir
    files = resource_files(resource_path)
    if not files:
        raise RuntimeError(
            f"No resources found in {resource_path.resolve()}; refusing to empty the index."
        )

    records = [_resource_record(file_path, resource_path, settings) for file_path in files]
    session_factory = get_session_factory(settings)
    with session_factory() as session:
        stored = database_resource_manifest(session)

    changed = [
        record
        for record in records
        if force or stored.get(record.source_path) != record.content_hash
    ]
    removed = sorted(stored.keys() - {record.source_path for record in records})

    chunks_by_record = [
        chunk_document(record.raw_content, record.source_name, settings) for record in changed
    ]
    vectors = iter(
        _embed_texts(
            [chunk.page_content for chunks in chunks_by_record for chunk in chunks], settings
        )
    )

    resource_rows = []
    chunk_rows = []
    for record, chunks in zip(changed, chunks_by_record):
        resource_id = uuid.uuid4()
        resource_rows.append(
            {
                "id": resource_id,
                "source_path": record.source_path,
                "source_name": record.source_name,
                "title": _extract_title(record.raw_content, record.source_name),
                "author": _extract_author(record.raw_content, record.source_name),
                "raw_content": record.raw_content,
                "content_hash": record.content_hash,
            }
        )
        for index, chunk in enumerate(chunks):
            chunk_rows.append(
                {
                    "resource_id": resource_id,
                    "chunk_index": index,
                    "source_path": record.source_path,
                    "source": record.source_name,
                    "title": chunk.metadata["title"],
                    "author": chunk.metadata["author"],
                    "h1": chunk.metadata.get("h1"),
                    "h2": chunk.metadata.get("h2"),
                    "h3": chunk.metadata.get("h3"),
                    "header_path": chunk.metadata.get("header_path"),
                    "content": chunk.page_content,
                    "content_hash": hash_text(chunk.page_content),
                    "embedding": next(vectors),
                }
            )

    with session_factory() as session, session.begin():
        session.execute(select(func.pg_advisory_xact_lock(_INGEST_LOCK_KEY)))
        if database_resource_manifest(session) != stored:
            raise RuntimeError("The index changed while this ingest was embedding; run it again.")

        # Updated resources are replaced outright; their chunks go with them (ON DELETE CASCADE).
        stale_paths = removed + [record.source_path for record in changed]
        if stale_paths:
            session.execute(
                delete(AgentResource).where(AgentResource.source_path.in_(stale_paths))
            )
        if resource_rows:
            session.execute(insert(AgentResource), resource_rows)
        for start in range(0, len(chunk_rows), _INSERT_BATCH_SIZE):
            session.execute(
                insert(AgentResourceChunk), chunk_rows[start : start + _INSERT_BATCH_SIZE]
            )

    return IngestStats(
        added_resources=sum(record.source_path not in stored for record in changed),
        updated_resources=sum(record.source_path in stored for record in changed),
        deleted_resources=len(removed),
        chunk_rows_written=len(chunk_rows),
    )
