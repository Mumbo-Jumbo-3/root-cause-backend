FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first so code-only changes reuse the cached layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Render's pre-deploy step runs the migrations from this image.
COPY alembic.ini ./
COPY alembic ./alembic

EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn health_agent.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
