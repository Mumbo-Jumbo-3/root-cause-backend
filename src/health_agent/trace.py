"""Research-trace payloads shown in the frontend's stage timeline.

Each node attaches a `detail` payload to its completed phase event: the findings,
queries, and documents that step produced. The same events are collected in
`AgentState.trace` and saved on the answer's AIMessage, so a reloaded thread or
share link renders the identical trace and can resolve the answer's citations.
"""

import hashlib
import re
from typing import Any

from langchain_core.documents import Document

TRACE_VERSION = 1
TRACE_RESET = "__reset__"

_STATUS_ID_RE = re.compile(r"/status(?:es)?/(\d+)")
_HANDLE_RE = re.compile(
    r"^https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,15})/status",
    re.IGNORECASE,
)


def merge_trace(left: list[dict] | None, right: list[dict] | str | None) -> list[dict]:
    """State reducer: append phase events; TRACE_RESET starts a fresh turn."""
    if right == TRACE_RESET:
        return []
    return [*(left or []), *(right or [])]


def tweet_status_id(url: str) -> str | None:
    match = _STATUS_ID_RE.search(url)
    return match.group(1) if match else None


def tweet_handle(url: str) -> str | None:
    match = _HANDLE_RE.match(url)
    if not match or match.group(1).lower() == "i":
        return None
    return match.group(1)


def grok_cited_urls(raw: Any) -> list[str]:
    """URLs xAI reports the search actually visited (`citations` / url annotations)."""
    urls: list[str] = []
    additional = getattr(raw, "additional_kwargs", None) or {}
    for url in additional.get("citations") or []:
        if isinstance(url, str):
            urls.append(url)
    content = getattr(raw, "content", None)
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            for annotation in block.get("annotations") or []:
                if isinstance(annotation, dict) and isinstance(annotation.get("url"), str):
                    urls.append(annotation["url"])
    return urls


def filter_uncited_urls(
    findings: list[dict[str, object]], cited_urls: list[str]
) -> list[dict[str, object]]:
    """Drop finding URLs the search never visited (likely fabricated).

    Matching is by post ID because xAI cites `x.com/i/status/<id>` while findings
    often use `x.com/<handle>/status/<id>`. With no citations reported there is
    nothing to check against, so findings pass through unchanged.
    """
    if not cited_urls:
        return findings
    cited_ids = {status_id for url in cited_urls if (status_id := tweet_status_id(url))}
    cited_set = set(cited_urls)
    filtered: list[dict[str, object]] = []
    for finding in findings:
        urls = [
            url
            for url in finding.get("source_urls", [])  # type: ignore[union-attr]
            if url in cited_set or tweet_status_id(url) in cited_ids
        ]
        filtered.append({**finding, "source_urls": urls})
    return filtered


def finding_items(findings: list[dict[str, object]], prefix: str) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for idx, finding in enumerate(findings, start=1):
        urls = [url for url in finding.get("source_urls", []) if isinstance(url, str)]  # type: ignore[union-attr]
        handles: list[str] = []
        for url in urls:
            handle = tweet_handle(url)
            if handle and handle not in handles:
                handles.append(handle)
        item: dict[str, object] = {
            "id": f"{prefix}-{idx}",
            "claim": finding.get("claim", ""),
            "relevance": finding.get("relevance", ""),
            "source_urls": urls,
            "handles": handles,
        }
        if finding.get("matched"):
            # Linked by text similarity (x_posts), not cited by Grok itself.
            item["matched"] = True
        items.append(item)
    return items


MAX_POSTS = 20


def post_urls(cited_urls: list[str]) -> list[str]:
    """X posts the search visited, one URL per post, for embedding in the trace."""
    posts: list[str] = []
    seen: set[str] = set()
    for url in cited_urls:
        status_id = tweet_status_id(url)
        if status_id and status_id not in seen:
            seen.add(status_id)
            posts.append(url)
    return posts[:MAX_POSTS]


def findings_detail(
    findings: list[dict[str, object]],
    prefix: str,
    queries: list[str] | None = None,
    posts: list[str] | None = None,
) -> dict[str, object]:
    """Findings plus the posts the search read.

    Grok rarely attributes post URLs to individual findings, so `posts` (the
    search's own citations) is what lets the trace show the tweets it read.
    """
    detail: dict[str, object] = {"findings": finding_items(findings, prefix)}
    if queries is not None:
        detail["queries"] = queries
    if posts:
        detail["posts"] = posts
    return detail


def doc_id(doc: Document) -> str:
    source = doc.metadata.get("source_path") or doc.metadata.get("source") or ""
    chunk = doc.metadata.get("chunk_index", "")
    digest = hashlib.sha1(f"{source}:{chunk}:{doc.page_content}".encode()).hexdigest()
    return f"doc-{digest[:12]}"


def doc_text(doc: Document) -> str:
    """Chunk text without the header-path prefix ingest prepends to every chunk."""
    text = doc.page_content
    header_path = doc.metadata.get("header_path")
    if header_path and text.startswith(f"{header_path}\n\n"):
        return text[len(header_path) + 2 :]
    return text


def doc_info(doc: Document) -> dict[str, object]:
    title = doc.metadata.get("title") or "Untitled"
    header_path = doc.metadata.get("header_path") or ""
    section = [part.strip() for part in header_path.split(">") if part.strip()]
    # Most documents open with an H1 that repeats the title.
    if section and section[0].lower() == title.strip().lower():
        section = section[1:]
    return {
        "id": doc_id(doc),
        "title": title,
        "author": doc.metadata.get("author") or "",
        "section": section,
        "text": doc_text(doc),
    }


def _doc_score(doc: Document) -> float | None:
    score = doc.metadata.get("relevance_score")
    if isinstance(score, (int, float)):
        return round(float(score), 4)
    retrieval = doc.metadata.get("retrieval")
    if isinstance(retrieval, dict):
        distance = retrieval.get("vector_distance")
        if isinstance(distance, (int, float)):
            return round(max(0.0, min(1.0, 1.0 - float(distance))), 4)
    return None


def _doc_queries(doc: Document) -> list[str]:
    retrieval = doc.metadata.get("retrieval")
    if not isinstance(retrieval, dict):
        return []
    return [q for q in retrieval.get("queries", []) if isinstance(q, str)]


def docs_detail(
    docs: list[Document],
    *,
    queries: list[str] | None = None,
    base_ids: set[str] | None = None,
    enrich_ids: set[str] | None = None,
    sent_ids: set[str] | None = None,
) -> dict[str, object]:
    """Ranked hits plus the text of each doc they reference.

    `queries` groups hits by the refined queries that retrieved them; base/enrich
    ids mark where each merged doc came from. Docs in `sent_ids` were carried by
    an earlier step's event, so only their hits are repeated: the trace keeps each
    doc's text once, and the frontend resolves hits against every event's docs.
    """
    query_keys = {q.strip().lower(): q for q in queries or []}
    hits: list[dict[str, object]] = []
    infos: dict[str, dict[str, object]] = {}
    for doc in docs:
        info = doc_info(doc)
        hit: dict[str, object] = {"id": info["id"], "score": _doc_score(doc)}
        if queries is not None:
            hit["queries"] = [
                query_keys[key]
                for q in _doc_queries(doc)
                if (key := q.strip().lower()) in query_keys
            ]
        if base_ids is not None and enrich_ids is not None:
            in_base = info["id"] in base_ids
            in_enrich = info["id"] in enrich_ids
            hit["origin"] = "both" if in_base and in_enrich else "expanded" if in_enrich else "base"
        hits.append(hit)
        if info["id"] not in (sent_ids or set()):
            infos[str(info["id"])] = info
    detail: dict[str, object] = {"hits": hits, "docs": infos}
    if queries is not None:
        detail["queries"] = queries
    return detail


def phase_event(
    phase: str,
    status: str,
    meta: dict[str, object] | None = None,
    detail: dict[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {"phase": phase, "status": status, "meta": meta or {}}
    if detail is not None:
        event["detail"] = detail
    return event


def finding_document(
    item: dict[str, object], channel: str, posts: list[dict[str, str]]
) -> dict[str, object]:
    """A citable Claude document for one X finding.

    The source posts' own text is what gets cited, with Grok's claim as context. A
    finding whose posts couldn't be fetched falls back to citing the claim.
    """
    handles = item.get("handles") or []
    title = channel + (f" — {', '.join('@' + h for h in handles)}" if handles else "")  # type: ignore[union-attr]
    why = f"Why it matters: {item['relevance']}"
    if posts:
        data = "\n\n".join(f"@{post['handle']}: {post['text']}" for post in posts)
        context = f"Summary: {item['claim']}\n{why}"
    else:
        data, context = str(item["claim"]), why
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": data},
        "title": title,
        "context": context,
        "citations": {"enabled": True},
    }


def archive_document(info: dict[str, object]) -> dict[str, object]:
    title = str(info["title"])
    if info.get("author"):
        title = f"{title} — {info['author']}"
    document: dict[str, object] = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": info["text"]},
        "title": title,
        "citations": {"enabled": True},
    }
    if info.get("section"):
        document["context"] = "Section: " + " > ".join(info["section"])  # type: ignore[arg-type]
    return document


def cited_text_and_citations(
    content: Any, evidence_ids: list[str]
) -> tuple[str, list[dict[str, object]]]:
    """Flatten Claude's cited text blocks into one string plus citation anchors.

    Each citation anchors at the end of the text block it supports; the frontend
    renders a numbered marker there. `document_index` maps into `evidence_ids`.
    """
    if isinstance(content, str):
        return content, []
    text = ""
    citations: list[dict[str, object]] = []
    for block in content or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text += block.get("text") or ""
        for citation in block.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            index = citation.get("document_index")
            if not isinstance(index, int) or not 0 <= index < len(evidence_ids):
                continue
            citations.append(
                {
                    "offset": len(text),
                    "evidence_id": evidence_ids[index],
                    "quote": citation.get("cited_text") or "",
                }
            )
    return text, citations
