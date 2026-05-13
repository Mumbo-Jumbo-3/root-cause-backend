import json
import threading
from pathlib import Path

import pytest

from scripts.fill_blank_ingestible_responses import (
    IngestibleResponsePipeline,
    build_product_prompt,
    build_prompt,
    build_research_prompt,
    extract_response_text,
    fill_blank_ingestibles,
    find_blank_ingestibles,
)


def write_ingestible(
    root: Path,
    slug: str,
    name: str | None = None,
    response: str | None = "",
) -> Path:
    entry_dir = root / slug
    entry_dir.mkdir(parents=True)
    meta = {"slug": slug}
    if name is not None:
        meta["name"] = name
    (entry_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if response is not None:
        (entry_dir / "response.md").write_text(response, encoding="utf-8")
    return entry_dir


def test_find_blank_ingestibles_detects_missing_and_whitespace(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    write_ingestible(root, "alpha", "Alpha", "existing response")
    write_ingestible(root, "beta", "Beta", "   \n\t")
    write_ingestible(root, "gamma", "Gamma", None)

    blanks = find_blank_ingestibles(root)

    assert [entry.slug for entry in blanks] == ["beta", "gamma"]
    assert [entry.name for entry in blanks] == ["Beta", "Gamma"]


def test_find_blank_ingestibles_falls_back_to_slug_name(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    write_ingestible(root, "magnesium-glycinate", response="")

    blanks = find_blank_ingestibles(root)

    assert len(blanks) == 1
    assert blanks[0].name == "Magnesium Glycinate"


def test_build_prompt_uses_exact_requested_template():
    assert build_prompt("Magnesium Glycinate") == (
        "Research Magnesium Glycinate as a nutrient for humans, what is it for? "
        "How is it best ingested in the optimal amounts (food, drinks, supplements, forms, recommended daily intake amount, risks, etc)? "
        "What are the top 3 highest quality/purity Magnesium Glycinate products i can buy?"
    )


def test_build_research_and_product_prompts_split_original_prompt():
    assert build_research_prompt("Magnesium Glycinate") == (
        "Research Magnesium Glycinate as a nutrient for humans, what is it for? "
        "How is it best ingested in the optimal amounts (food, drinks, supplements, forms, recommended daily intake amount, risks, etc)?"
    )
    assert build_product_prompt("Magnesium Glycinate") == (
        "What are the top 3 highest quality/purity Magnesium Glycinate products i can buy?"
    )


def test_extract_response_text_prefers_output_text():
    assert extract_response_text({"output_text": "  Answer  "}) == "Answer"


def test_extract_response_text_reads_response_output_blocks():
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "First"},
                    {"type": "output_text", "text": "Second"},
                ],
            }
        ]
    }

    assert extract_response_text(payload) == "First\nSecond"


def test_fill_blank_ingestibles_dry_run_does_not_call_model_or_write(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    entry_dir = write_ingestible(root, "blank", "Blank", "")

    def fail_generate_response(*args):
        raise AssertionError("generate_response should not be called during dry-run")

    result = fill_blank_ingestibles(
        root,
        api_key="",
        dry_run=True,
        generate_response=fail_generate_response,
    )

    assert result.blank_count == 1
    assert result.written_count == 0
    assert (entry_dir / "response.md").read_text(encoding="utf-8") == ""


def test_fill_blank_ingestibles_writes_only_blank_responses(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    blank_dir = write_ingestible(root, "blank", "Blank", "")
    filled_dir = write_ingestible(root, "filled", "Filled", "keep me")
    names: list[str] = []

    def generate_response(name: str) -> str:
        names.append(name)
        return "generated response"

    result = fill_blank_ingestibles(
        root,
        api_key="test-key",
        model="grok-4.3",
        generate_response=generate_response,
    )

    assert result.blank_count == 1
    assert result.written_count == 1
    assert result.failures == ()
    assert (
        blank_dir / "response.md"
    ).read_text(encoding="utf-8") == "generated response\n"
    assert (filled_dir / "response.md").read_text(encoding="utf-8") == "keep me"
    assert names == ["Blank"]


def test_fill_blank_ingestibles_reports_empty_model_output(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    entry_dir = write_ingestible(root, "blank", "Blank", "")

    def generate_response(name: str) -> str:
        return "   "

    result = fill_blank_ingestibles(
        root,
        api_key="test-key",
        generate_response=generate_response,
    )

    assert result.blank_count == 1
    assert result.written_count == 0
    assert result.failures == ("blank: model returned empty output",)
    assert (entry_dir / "response.md").read_text(encoding="utf-8") == ""


def test_fill_blank_ingestibles_reports_pipeline_failure_without_writing(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    entry_dir = write_ingestible(root, "blank", "Blank", "")

    def generate_response(name: str) -> str:
        raise RuntimeError("branch failed")

    result = fill_blank_ingestibles(
        root,
        api_key="test-key",
        generate_response=generate_response,
    )

    assert result.blank_count == 1
    assert result.written_count == 0
    assert result.failures == ("blank: branch failed",)
    assert (entry_dir / "response.md").read_text(encoding="utf-8") == ""


def test_response_pipeline_runs_research_and_product_in_parallel_and_synthesizes():
    research_started = threading.Event()
    product_started = threading.Event()
    research_prompts: list[str] = []
    product_prompts: list[str] = []
    captured: dict[str, str] = {}

    def research_call(prompt: str) -> str:
        research_prompts.append(prompt)
        research_started.set()
        if not product_started.wait(1):
            raise AssertionError("product branch did not start in parallel")
        return "research response"

    def product_call(prompt: str) -> str:
        product_prompts.append(prompt)
        product_started.set()
        if not research_started.wait(1):
            raise AssertionError("research branch did not start in parallel")
        return "product response"

    def synthesis_call(
        ingestible_name: str,
        full_prompt: str,
        research_response: str,
        product_response: str,
    ) -> str:
        captured["ingestible_name"] = ingestible_name
        captured["full_prompt"] = full_prompt
        captured["research_response"] = research_response
        captured["product_response"] = product_response
        return "final guide"

    pipeline = IngestibleResponsePipeline(
        xai_api_key="test-key",
        research_call=research_call,
        product_call=product_call,
        synthesis_call=synthesis_call,
    )

    assert pipeline("Magnesium Glycinate") == "final guide"
    assert research_prompts == [build_research_prompt("Magnesium Glycinate")]
    assert product_prompts == [build_product_prompt("Magnesium Glycinate")]
    assert captured == {
        "ingestible_name": "Magnesium Glycinate",
        "full_prompt": build_prompt("Magnesium Glycinate"),
        "research_response": "research response",
        "product_response": "product response",
    }


def test_response_pipeline_rejects_empty_research_branch():
    pipeline = IngestibleResponsePipeline(
        xai_api_key="test-key",
        research_call=lambda prompt: " ",
        product_call=lambda prompt: "product response",
        synthesis_call=lambda name, full, research, product: "final guide",
    )

    with pytest.raises(RuntimeError, match="research branch returned empty output"):
        pipeline("Blank")


def test_response_pipeline_rejects_empty_product_branch():
    pipeline = IngestibleResponsePipeline(
        xai_api_key="test-key",
        research_call=lambda prompt: "research response",
        product_call=lambda prompt: " ",
        synthesis_call=lambda name, full, research, product: "final guide",
    )

    with pytest.raises(RuntimeError, match="product branch returned empty output"):
        pipeline("Blank")


def test_response_pipeline_rejects_empty_synthesis():
    pipeline = IngestibleResponsePipeline(
        xai_api_key="test-key",
        research_call=lambda prompt: "research response",
        product_call=lambda prompt: "product response",
        synthesis_call=lambda name, full, research, product: " ",
    )

    with pytest.raises(RuntimeError, match="synthesis returned empty output"):
        pipeline("Blank")
