"""Fill blank ingestible response.md files with generated ingestible guides.

Run from the backend directory:
    uv run --group dev python scripts/fill_blank_ingestible_responses.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx


DEFAULT_MODEL = "grok-4.3"
XAI_RESPONSES_URL = "https://api.x.ai/v1/responses"
DEFAULT_CONTENT_ROOT = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "health_agent"
    / "content"
    / "ingestibles"
)
PROMPT_TEMPLATE = (
    "Research {ingestible} as a nutrient for humans, what is it for? "
    "How is it best ingested in the optimal amounts (food, drinks, supplements, forms, recommended daily intake amount, risks, etc)? "
    "What are the top 3 highest quality/purity {ingestible} products i can buy?"
)
RESEARCH_PROMPT_TEMPLATE = (
    "Research {ingestible} as a nutrient for humans, what is it for? "
    "How is it best ingested in the optimal amounts (food, drinks, supplements, forms, recommended daily intake amount, risks, etc)?"
)
PRODUCT_PROMPT_TEMPLATE = (
    "What are the top 3 highest quality/purity {ingestible} products i can buy?"
)
SYNTHESIS_SYSTEM_PROMPT = """You are writing consumer-facing ingestible guides.
Use the research and ingestion response for nutrient purpose, food/forms, amounts,
and risks. Use the product response for buyable product recommendations.
Write one coherent Markdown guide that answers every part of the original prompt.
Do not mention internal pipelines, model names, LangGraph, Grok, Claude, or prompts."""


@dataclass(frozen=True)
class IngestibleEntry:
    slug: str
    name: str
    response_path: Path


@dataclass(frozen=True)
class FillResult:
    blank_count: int
    written_count: int
    failures: tuple[str, ...]


ModelCaller = Callable[[str, str, str, float], str]
ResponseGenerator = Callable[[str], str]
BranchCaller = Callable[[str], str]
SynthesisCaller = Callable[[str, str, str, str], str]


class MissingRuntimeConfigError(RuntimeError):
    pass


def slug_to_name(slug: str) -> str:
    words = re.sub(r"[-_]+", " ", slug).strip()
    return words.title() if words else slug


def load_ingestible_name(entry_dir: Path) -> str:
    meta_path = entry_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        name = meta.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return slug_to_name(entry_dir.name)


def is_blank_response(response_path: Path) -> bool:
    if not response_path.is_file():
        return True
    return not response_path.read_text(encoding="utf-8").strip()


def find_blank_ingestibles(content_root: Path) -> list[IngestibleEntry]:
    if not content_root.is_dir():
        raise FileNotFoundError(f"ingestibles content root not found: {content_root}")

    blanks: list[IngestibleEntry] = []
    for entry_dir in sorted(content_root.iterdir(), key=lambda path: path.name):
        if not entry_dir.is_dir():
            continue
        response_path = entry_dir / "response.md"
        if is_blank_response(response_path):
            blanks.append(
                IngestibleEntry(
                    slug=entry_dir.name,
                    name=load_ingestible_name(entry_dir),
                    response_path=response_path,
                )
            )
    return blanks


def build_prompt(ingestible_name: str) -> str:
    return PROMPT_TEMPLATE.format(ingestible=ingestible_name)


def build_research_prompt(ingestible_name: str) -> str:
    return RESEARCH_PROMPT_TEMPLATE.format(ingestible=ingestible_name)


def build_product_prompt(ingestible_name: str) -> str:
    return PRODUCT_PROMPT_TEMPLATE.format(ingestible=ingestible_name)


def extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts: list[str] = []
    for output_item in payload.get("output", []):
        if not isinstance(output_item, dict):
            continue
        for content_item in output_item.get("content", []):
            if isinstance(content_item, str):
                parts.append(content_item)
            elif isinstance(content_item, dict):
                text = content_item.get("text")
                if isinstance(text, str):
                    parts.append(text)

    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def call_xai_response(
    api_key: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> str:
    response = httpx.post(
        XAI_RESPONSES_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search"}],
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return extract_response_text(response.json())


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()

    return str(content or "").strip()


def extract_graph_response_text(result: Any) -> str:
    if not isinstance(result, dict):
        return ""
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return extract_message_text(messages[-1])


def build_synthesis_user_prompt(
    ingestible_name: str,
    full_prompt: str,
    research_response: str,
    product_response: str,
) -> str:
    return (
        f"## Ingestible\n{ingestible_name}\n\n"
        f"## Original Prompt\n{full_prompt}\n\n"
        f"## Research and Ingestion Response\n{research_response}\n\n"
        f"## Product Response\n{product_response}\n\n"
        "Synthesize the two responses into a single practical guide. "
        "The guide must answer what the ingestible is for, how it is best ingested, "
        "and the top 3 high-quality/purity products someone can buy."
    )


def load_settings() -> Any:
    from health_agent.config import Settings

    return Settings()


def missing_runtime_config(settings: Any, xai_api_key: str) -> list[str]:
    checks = {
        "XAI_API_KEY": xai_api_key,
        "ANTHROPIC_API_KEY": getattr(settings, "anthropic_api_key", ""),
        "VOYAGE_API_KEY": getattr(settings, "voyage_api_key", ""),
        "DATABASE_URL": getattr(settings, "database_url", ""),
    }
    return [name for name, value in checks.items() if not str(value).strip()]


class IngestibleResponsePipeline:
    def __init__(
        self,
        *,
        xai_api_key: str,
        product_model: str = DEFAULT_MODEL,
        timeout_seconds: float = 300.0,
        settings: Any | None = None,
        research_call: BranchCaller | None = None,
        product_call: BranchCaller | None = None,
        synthesis_call: SynthesisCaller | None = None,
    ) -> None:
        self.xai_api_key = xai_api_key
        self.product_model = product_model
        self.timeout_seconds = timeout_seconds
        self._settings = settings
        self._research_call = research_call
        self._product_call = product_call
        self._synthesis_call = synthesis_call
        self._graph = None
        self._synthesis_model = None

    @property
    def settings(self) -> Any:
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def _langgraph_research(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        from health_agent.graph import build_graph

        if self._graph is None:
            self._graph = build_graph(self.settings)

        result = self._graph.invoke({"messages": [HumanMessage(content=prompt)]})
        return extract_graph_response_text(result)

    def _grok_product_search(self, prompt: str) -> str:
        return call_xai_response(
            self.xai_api_key,
            self.product_model,
            prompt,
            self.timeout_seconds,
        )

    def _claude_synthesis(
        self,
        ingestible_name: str,
        full_prompt: str,
        research_response: str,
        product_response: str,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        from health_agent.models import get_claude_synthesis_model

        if self._synthesis_model is None:
            self._synthesis_model = get_claude_synthesis_model(self.settings)

        response = self._synthesis_model.invoke(
            [
                SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_synthesis_user_prompt(
                        ingestible_name,
                        full_prompt,
                        research_response,
                        product_response,
                    )
                ),
            ]
        )
        return extract_message_text(response)

    def __call__(self, ingestible_name: str) -> str:
        full_prompt = build_prompt(ingestible_name)
        research_prompt = build_research_prompt(ingestible_name)
        product_prompt = build_product_prompt(ingestible_name)
        research_call = self._research_call or self._langgraph_research
        product_call = self._product_call or self._grok_product_search
        synthesis_call = self._synthesis_call or self._claude_synthesis

        with ThreadPoolExecutor(max_workers=2) as executor:
            research_future = executor.submit(research_call, research_prompt)
            product_future = executor.submit(product_call, product_prompt)
            research_response = research_future.result().strip()
            product_response = product_future.result().strip()

        if not research_response:
            raise RuntimeError("research branch returned empty output")
        if not product_response:
            raise RuntimeError("product branch returned empty output")

        synthesized = synthesis_call(
            ingestible_name,
            full_prompt,
            research_response,
            product_response,
        ).strip()
        if not synthesized:
            raise RuntimeError("synthesis returned empty output")
        return synthesized


def fill_blank_ingestibles(
    content_root: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    timeout_seconds: float = 300.0,
    call_model: ModelCaller | None = None,
    generate_response: ResponseGenerator | None = None,
    settings: Any | None = None,
) -> FillResult:
    blanks = find_blank_ingestibles(content_root)
    failures: list[str] = []
    written_count = 0

    if dry_run:
        for entry in blanks:
            print(f"blank: {entry.slug} ({entry.name})")
        return FillResult(blank_count=len(blanks), written_count=0, failures=())

    if not blanks:
        return FillResult(blank_count=0, written_count=0, failures=())

    if call_model is not None and generate_response is not None:
        raise ValueError("Pass call_model or generate_response, not both.")

    if generate_response is None:
        if call_model is not None:
            generate_response = lambda name: call_model(
                api_key,
                model,
                build_prompt(name),
                timeout_seconds,
            )
        else:
            pipeline_settings = settings or load_settings()
            missing = missing_runtime_config(pipeline_settings, api_key)
            if missing:
                raise MissingRuntimeConfigError(
                    "Missing required environment variable(s): " + ", ".join(missing)
                )
            generate_response = IngestibleResponsePipeline(
                xai_api_key=api_key,
                product_model=model,
                timeout_seconds=timeout_seconds,
                settings=pipeline_settings,
            )

    for entry in blanks:
        print(f"filling: {entry.slug} ({entry.name})")
        try:
            generated = generate_response(entry.name).strip()
        except Exception as exc:
            failures.append(f"{entry.slug}: {exc}")
            print(f"failed: {entry.slug}: {exc}", file=sys.stderr)
            continue

        if not generated:
            failures.append(f"{entry.slug}: model returned empty output")
            print(f"failed: {entry.slug}: model returned empty output", file=sys.stderr)
            continue

        entry.response_path.write_text(generated + "\n", encoding="utf-8")
        written_count += 1
        print(f"wrote: {entry.response_path}")

    return FillResult(
        blank_count=len(blanks),
        written_count=written_count,
        failures=tuple(failures),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill blank ingestible response.md files using LangGraph, Grok, and Claude."
    )
    parser.add_argument(
        "--content-root",
        type=Path,
        default=DEFAULT_CONTENT_ROOT,
        help=f"Ingestibles content root. Defaults to {DEFAULT_CONTENT_ROOT}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"xAI model to call. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List blank ingestibles without calling xAI or writing files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds. Defaults to 300.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("XAI_API_KEY", "")

    try:
        result = fill_blank_ingestibles(
            content_root=args.content_root,
            api_key=api_key,
            model=args.model,
            dry_run=args.dry_run,
            timeout_seconds=args.timeout,
        )
    except MissingRuntimeConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.blank_count == 0:
        print("No blank ingestible responses found.")
    elif args.dry_run:
        print(f"Found {result.blank_count} blank ingestible response(s).")
    else:
        print(
            f"Done: wrote {result.written_count}/{result.blank_count} blank "
            "ingestible response(s)."
        )

    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
