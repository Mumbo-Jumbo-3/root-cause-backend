"""FastAPI application that embeds the LangGraph agent as a library.

Replaces `langgraph dev` / the Agent Server API surface with our own narrow set of
endpoints, backed by AsyncPostgresSaver for durable thread persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import desc, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sse_starlette.sse import EventSourceResponse

from health_agent.auth import require_clerk_user, require_webhook_secret
from health_agent.config import get_settings
from health_agent.content import (
    get_featured,
    get_product,
    list_featured,
    list_products,
)
from health_agent.db.core import get_session_factory
from health_agent.db.models import SharedConversation, Thread, User
from health_agent.graph import build_graph


logger = logging.getLogger(__name__)

SHARE_TITLE_LIMIT = 120
SHARE_FIRST_MESSAGE_LIMIT = 500
THREAD_TITLE_LIMIT = 120


def _psycopg_dsn(database_url: str) -> str:
    """AsyncPostgresSaver wants a plain libpq DSN, not SQLAlchemy's `+psycopg` variant."""
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            base: dict[str, Any] = {
                "id": msg.id,
                "type": msg.type,
                "content": msg.content,
            }
            if msg.type == "ai":
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    base["tool_calls"] = tool_calls
            elif msg.type == "tool":
                base["tool_call_id"] = getattr(msg, "tool_call_id", None)
                base["name"] = getattr(msg, "name", None)
            out.append(base)
        elif isinstance(msg, dict):
            out.append(msg)
    return out


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    dsn = _psycopg_dsn(settings.database_url)
    if not dsn:
        raise RuntimeError("DATABASE_URL must be set")

    pool = AsyncConnectionPool(
        conninfo=dsn,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0},
        open=False,
    )
    await pool.open()
    try:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app.state.graph = build_graph(settings, checkpointer=checkpointer)
        app.state.pool = pool
        logger.info("LangGraph checkpointer ready (pool size=%s)", pool.max_size)
        yield
    finally:
        await pool.close()


app = FastAPI(lifespan=lifespan, title="health-agent")


def _ensure_user_row(user_id: str) -> None:
    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        stmt = (
            pg_insert(User)
            .values(clerk_user_id=user_id)
            .on_conflict_do_nothing(index_elements=[User.clerk_user_id])
        )
        session.execute(stmt)
        session.commit()


def clerk_user(request: Request) -> str:
    user_id = require_clerk_user(request)
    _ensure_user_row(user_id)
    return user_id


@app.get("/info")
async def info() -> dict[str, Any]:
    return {"ok": True, "graph_id": "agent"}


@app.get("/ok")
async def healthcheck() -> dict[str, bool]:
    return {"ok": True}


@app.post("/threads")
async def create_thread(request: Request, user_id: str = Depends(clerk_user)) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    metadata = body.get("metadata") if isinstance(body, dict) else None
    title = ""
    if isinstance(metadata, dict):
        title = _truncate(str(metadata.get("title") or ""), THREAD_TITLE_LIMIT)

    thread_id = str(uuid.uuid4())
    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = Thread(thread_id=thread_id, user_id=user_id, title=title)
        session.add(row)
        session.commit()

    return JSONResponse(
        {
            "thread_id": thread_id,
            "metadata": {"user_id": user_id, "title": title},
            "created_at": None,
            "updated_at": None,
        }
    )


@app.post("/threads/search")
async def search_threads(request: Request, user_id: str = Depends(clerk_user)) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}

    limit = 50
    if isinstance(body, dict):
        raw_limit = body.get("limit")
        if isinstance(raw_limit, int) and 0 < raw_limit <= 200:
            limit = raw_limit

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        rows = (
            session.execute(
                select(Thread)
                .where(Thread.user_id == user_id)
                .order_by(desc(Thread.updated_at))
                .limit(limit)
            )
            .scalars()
            .all()
        )
        payload = [
            {
                "thread_id": row.thread_id,
                "metadata": {"user_id": row.user_id, "title": row.title},
                "created_at": row.created_at.isoformat(),
                "updated_at": row.updated_at.isoformat(),
            }
            for row in rows
        ]

    return JSONResponse(payload)


async def _load_thread(thread_id: str, user_id: str) -> Thread:
    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = session.get(Thread, thread_id)
        if row is None or row.user_id != user_id:
            raise HTTPException(status_code=404, detail="not found")
        return row


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str, user_id: str = Depends(clerk_user)) -> JSONResponse:
    row = await _load_thread(thread_id, user_id)
    return JSONResponse(
        {
            "thread_id": row.thread_id,
            "metadata": {"user_id": row.user_id, "title": row.title},
            "created_at": row.created_at.isoformat(),
            "updated_at": row.updated_at.isoformat(),
        }
    )


def _owns_thread(thread_id: str, user_id: str) -> bool:
    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = session.get(Thread, thread_id)
        return row is not None and row.user_id == user_id


@app.get("/threads/{thread_id}/state")
async def get_thread_state(
    request: Request,
    thread_id: str,
    user_id: str = Depends(clerk_user),
) -> JSONResponse:
    if not _owns_thread(thread_id, user_id):
        raise HTTPException(status_code=404, detail="not found")

    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    messages = _serialize_messages(values.get("messages", []))

    return JSONResponse(
        {
            "values": {"messages": messages},
            "next": list(snapshot.next or []),
            "checkpoint_id": (
                snapshot.config.get("configurable", {}).get("checkpoint_id")
                if snapshot.config
                else None
            ),
        }
    )


def _bump_thread(thread_id: str, user_id: str, first_message: str | None) -> None:
    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = session.get(Thread, thread_id)
        if row is None:
            title = _truncate(first_message or "", THREAD_TITLE_LIMIT)
            session.add(Thread(thread_id=thread_id, user_id=user_id, title=title))
            session.commit()
            return
        if row.user_id != user_id:
            raise HTTPException(status_code=403, detail="forbidden")
        if not row.title and first_message:
            row.title = _truncate(first_message, THREAD_TITLE_LIMIT)
        session.add(row)
        session.commit()


def _first_text_from_input(inp: Any) -> str | None:
    if not isinstance(inp, dict):
        return None
    messages = inp.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    if isinstance(inp.get("content"), str):
        return inp["content"]
    return None


async def _stream_events(
    graph: Any,
    input_payload: Any,
    config: dict[str, Any],
) -> AsyncIterator[dict[str, str]]:
    try:
        async for mode, chunk in graph.astream(
            input_payload,
            config=config,
            stream_mode=["messages", "updates", "custom"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk
                if not isinstance(message_chunk, BaseMessage):
                    continue
                tags = (metadata or {}).get("tags") or []
                if "nostream" in tags:
                    continue
                content = message_chunk.content
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    text = "".join(text_parts)
                else:
                    text = str(content or "")
                if text:
                    yield {
                        "event": "token",
                        "data": json.dumps(
                            {
                                "content": text,
                                "node": (metadata or {}).get("langgraph_node"),
                            }
                        ),
                    }
            elif mode == "updates":
                yield {
                    "event": "update",
                    "data": json.dumps(
                        {
                            "nodes": list(chunk.keys()) if isinstance(chunk, dict) else [],
                        }
                    ),
                }
            elif mode == "custom":
                if not isinstance(chunk, dict) or chunk.get("kind") != "phase":
                    continue
                phase = chunk.get("phase")
                status = chunk.get("status")
                meta = chunk.get("meta") or {}
                logger.debug(
                    "phase event: phase=%s status=%s meta=%s", phase, status, meta
                )
                yield {
                    "event": "phase",
                    "data": json.dumps(
                        {"phase": phase, "status": status, "meta": meta}
                    ),
                }
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("stream failed")
        yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        return

    snapshot = await graph.aget_state(config)
    values = snapshot.values or {}
    messages = _serialize_messages(values.get("messages", []))
    yield {
        "event": "values",
        "data": json.dumps({"messages": messages}),
    }
    yield {"event": "done", "data": "{}"}


@app.post("/runs/stream")
async def runs_stream(request: Request, user_id: str = Depends(clerk_user)) -> EventSourceResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    thread_id = body.get("thread_id") if isinstance(body, dict) else None
    input_payload = body.get("input") if isinstance(body, dict) else None
    if not isinstance(thread_id, str) or not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    if input_payload is None:
        raise HTTPException(status_code=400, detail="input required")

    first_text = _first_text_from_input(input_payload)
    _bump_thread(thread_id, user_id, first_text)

    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}, "metadata": {"user_id": user_id}}

    return EventSourceResponse(_stream_events(graph, input_payload, config))


# ---- Ported from http_app.py -------------------------------------------------


@app.post("/share")
async def create_share(request: Request, user_id: str = Depends(clerk_user)) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    thread_id = (body.get("thread_id") or "").strip()
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id required")
    if not _owns_thread(thread_id, user_id):
        raise HTTPException(status_code=404, detail="not found")

    title = _truncate(body.get("title") or "", SHARE_TITLE_LIMIT)
    first_message = _truncate(body.get("first_message") or "", SHARE_FIRST_MESSAGE_LIMIT)

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = SharedConversation(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
            first_message=first_message,
        )
        session.add(row)
        session.commit()
        share_id = str(row.share_id)

    return JSONResponse({"share_id": share_id})


@app.get("/share/{share_id}")
async def get_share(share_id: str) -> JSONResponse:
    try:
        parsed = uuid.UUID(share_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="not found")

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = session.get(SharedConversation, parsed)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        return JSONResponse(
            {
                "share_id": str(row.share_id),
                "thread_id": row.thread_id,
                "title": row.title,
                "first_message": row.first_message,
                "created_at": row.created_at.isoformat(),
            }
        )


@app.get("/share/{share_id}/state")
async def get_share_state(request: Request, share_id: str) -> JSONResponse:
    try:
        parsed = uuid.UUID(share_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=404, detail="not found")

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        row = session.get(SharedConversation, parsed)
        if row is None:
            raise HTTPException(status_code=404, detail="not found")
        thread_id = row.thread_id

    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values or {}
    messages = _serialize_messages(values.get("messages", []))
    return JSONResponse({"values": {"messages": messages}})


@app.get("/featured")
async def list_featured_endpoint() -> JSONResponse:
    return JSONResponse(list_featured())


@app.get("/featured/{slug}")
async def get_featured_endpoint(slug: str) -> JSONResponse:
    item = get_featured(slug)
    if item is None:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(item)


@app.get("/products")
async def list_products_endpoint() -> JSONResponse:
    return JSONResponse(list_products())


@app.get("/products/{slug}")
async def get_product_endpoint(slug: str) -> JSONResponse:
    item = get_product(slug)
    if item is None:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(item)


def _primary_email(email_addresses: list) -> str | None:
    if not isinstance(email_addresses, list) or not email_addresses:
        return None
    first = email_addresses[0]
    if isinstance(first, dict):
        return first.get("email_address")
    return None


@app.post("/internal/users/sync")
async def sync_user(request: Request) -> JSONResponse:
    require_webhook_secret(request)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    event_type = body.get("type", "")
    data = body.get("data") or {}
    clerk_user_id = data.get("id")
    if not isinstance(clerk_user_id, str) or not clerk_user_id:
        raise HTTPException(status_code=400, detail="user id required")

    session_factory = get_session_factory(get_settings())
    with session_factory() as session:
        if event_type == "user.deleted":
            session.query(User).filter(User.clerk_user_id == clerk_user_id).delete()
            session.commit()
            return JSONResponse({"status": "deleted"})

        if event_type in ("user.created", "user.updated"):
            email = _primary_email(data.get("email_addresses") or [])
            first_name = data.get("first_name")
            last_name = data.get("last_name")
            stmt = pg_insert(User).values(
                clerk_user_id=clerk_user_id,
                email=email,
                first_name=first_name,
                last_name=last_name,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[User.clerk_user_id],
                set_={
                    "email": stmt.excluded.email,
                    "first_name": stmt.excluded.first_name,
                    "last_name": stmt.excluded.last_name,
                },
            )
            session.execute(stmt)
            session.commit()
            return JSONResponse({"status": "upserted"})

    return JSONResponse({"status": "ignored"})
