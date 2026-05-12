"""Fill blank ingestible response.md files with Grok-generated research.

Run from the backend directory:
    uv run --group dev python scripts/fill_blank_ingestible_responses.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
    "How is it best ingested in the optimal amounts (food, supplements, etc)? "
    "What are the top 3 highest quality/purity {ingestible} products i can buy?"
)


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


def fill_blank_ingestibles(
    content_root: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    timeout_seconds: float = 300.0,
    call_model: ModelCaller = call_xai_response,
) -> FillResult:
    blanks = find_blank_ingestibles(content_root)
    failures: list[str] = []
    written_count = 0

    if dry_run:
        for entry in blanks:
            print(f"blank: {entry.slug} ({entry.name})")
        return FillResult(blank_count=len(blanks), written_count=0, failures=())

    for entry in blanks:
        prompt = build_prompt(entry.name)
        print(f"filling: {entry.slug} ({entry.name})")
        try:
            generated = call_model(api_key, model, prompt, timeout_seconds).strip()
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
        description="Fill blank ingestible response.md files using xAI Grok."
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

    if not args.dry_run and not api_key.strip():
        print("XAI_API_KEY must be set unless --dry-run is used.", file=sys.stderr)
        return 2

    try:
        result = fill_blank_ingestibles(
            content_root=args.content_root,
            api_key=api_key,
            model=args.model,
            dry_run=args.dry_run,
            timeout_seconds=args.timeout,
        )
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
