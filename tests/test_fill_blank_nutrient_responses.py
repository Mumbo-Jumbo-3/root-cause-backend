import json
import threading
from pathlib import Path

import pytest

from scripts.fill_blank_nutrient_responses import (
    SECTIONS,
    NutrientResponsePipeline,
    build_product_prompt,
    build_prompt,
    build_research_prompt,
    extract_response_text,
    fill_blank_nutrients,
    find_blank_nutrients,
    find_nutrients,
)


def _all_sections_covered(_nutrient: str, _research: str) -> dict[str, bool]:
    return {section: True for section in SECTIONS}


def write_nutrient(
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


def test_find_blank_nutrients_detects_missing_and_whitespace(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    write_nutrient(root, "alpha", "Alpha", "existing response")
    write_nutrient(root, "beta", "Beta", "   \n\t")
    write_nutrient(root, "gamma", "Gamma", None)

    blanks = find_blank_nutrients(root)

    assert [entry.slug for entry in blanks] == ["beta", "gamma"]
    assert [entry.name for entry in blanks] == ["Beta", "Gamma"]


def test_find_blank_nutrients_falls_back_to_slug_name(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    write_nutrient(root, "magnesium-glycinate", response="")

    blanks = find_blank_nutrients(root)

    assert len(blanks) == 1
    assert blanks[0].name == "Magnesium Glycinate"


def test_find_nutrients_returns_all_when_only_blanks_false(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    write_nutrient(root, "alpha", "Alpha", "existing response")
    write_nutrient(root, "beta", "Beta", "")
    write_nutrient(root, "gamma", "Gamma", "another response")

    all_entries = find_nutrients(root, only_blanks=False)

    assert [entry.slug for entry in all_entries] == ["alpha", "beta", "gamma"]


def test_find_nutrients_defaults_to_blanks_only(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    write_nutrient(root, "alpha", "Alpha", "existing response")
    write_nutrient(root, "beta", "Beta", "")

    entries = find_nutrients(root)

    assert [entry.slug for entry in entries] == ["beta"]


def test_build_prompt_uses_exact_requested_template():
    assert build_prompt("Magnesium Glycinate") == (
        "Research magnesium glycinate. How does it affect human health/nutrition/performance? What is it for? "
        "How is it best ingested in the optimal amounts (food, drinks, supplements, forms, dosing, risks, interactions, etc)? "
        "What are the top 3 highest quality/purity magnesium glycinate products i can buy?"
    )


def test_build_prompt_preserves_single_letter_vitamin_designations():
    assert build_prompt("Vitamin D").startswith("Research vitamin D.")


def test_build_research_prompt_covers_all_four_sections_with_lowercase_nutrient():
    prompt = build_research_prompt("Magnesium Glycinate")
    assert "magnesium glycinate" in prompt
    assert "Magnesium Glycinate" not in prompt  # _prompt_form lowercases multi-char words
    assert "1. PURPOSE" in prompt
    assert "2. INGESTION" in prompt
    assert "3. DOSING" in prompt
    assert "4. RISKS" in prompt


def test_build_product_prompt_excludes_listed_additives():
    assert build_product_prompt("Magnesium Glycinate").startswith(
        "What are the top 3 highest quality/purity magnesium glycinate products i can buy?"
    )
    assert "titanium dioxide" in build_product_prompt("Magnesium Glycinate")


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


def test_fill_blank_nutrients_dry_run_does_not_call_model_or_write(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    entry_dir = write_nutrient(root, "blank", "Blank", "")

    def fail_generate_response(*args):
        raise AssertionError("generate_response should not be called during dry-run")

    result = fill_blank_nutrients(
        root,
        api_key="",
        dry_run=True,
        generate_response=fail_generate_response,
    )

    assert result.blank_count == 1
    assert result.written_count == 0
    assert (entry_dir / "response.md").read_text(encoding="utf-8") == ""


def test_fill_blank_nutrients_writes_only_blank_responses(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    blank_dir = write_nutrient(root, "blank", "Blank", "")
    filled_dir = write_nutrient(root, "filled", "Filled", "keep me")
    names: list[str] = []

    def generate_response(name: str) -> str:
        names.append(name)
        return "generated response"

    result = fill_blank_nutrients(
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


def test_fill_blank_nutrients_refresh_all_overwrites_existing(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    blank_dir = write_nutrient(root, "blank", "Blank", "")
    filled_dir = write_nutrient(root, "filled", "Filled", "old content")
    names: list[str] = []

    def generate_response(name: str) -> str:
        names.append(name)
        return "regenerated"

    result = fill_blank_nutrients(
        root,
        api_key="test-key",
        generate_response=generate_response,
        refresh_all=True,
    )

    assert result.blank_count == 2
    assert result.written_count == 2
    assert result.failures == ()
    assert (blank_dir / "response.md").read_text(encoding="utf-8") == "regenerated\n"
    assert (filled_dir / "response.md").read_text(encoding="utf-8") == "regenerated\n"
    assert sorted(names) == ["Blank", "Filled"]


def test_fill_blank_nutrients_refresh_all_with_only_targets_non_blank(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    write_nutrient(root, "blank", "Blank", "")
    filled_dir = write_nutrient(root, "filled", "Filled", "old content")

    def generate_response(name: str) -> str:
        return "regenerated"

    result = fill_blank_nutrients(
        root,
        api_key="test-key",
        generate_response=generate_response,
        only="filled",
        refresh_all=True,
    )

    assert result.blank_count == 1
    assert result.written_count == 1
    assert (filled_dir / "response.md").read_text(encoding="utf-8") == "regenerated\n"


def test_fill_blank_nutrients_reports_empty_model_output(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    entry_dir = write_nutrient(root, "blank", "Blank", "")

    def generate_response(name: str) -> str:
        return "   "

    result = fill_blank_nutrients(
        root,
        api_key="test-key",
        generate_response=generate_response,
    )

    assert result.blank_count == 1
    assert result.written_count == 0
    assert result.failures == ("blank: model returned empty output",)
    assert (entry_dir / "response.md").read_text(encoding="utf-8") == ""


def test_fill_blank_nutrients_reports_pipeline_failure_without_writing(tmp_path: Path):
    root = tmp_path / "nutrients"
    root.mkdir()
    entry_dir = write_nutrient(root, "blank", "Blank", "")

    def generate_response(name: str) -> str:
        raise RuntimeError("branch failed")

    result = fill_blank_nutrients(
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
        nutrient_name: str,
        full_prompt: str,
        research_response: str,
        product_response: str,
    ) -> str:
        captured["nutrient_name"] = nutrient_name
        captured["full_prompt"] = full_prompt
        captured["research_response"] = research_response
        captured["product_response"] = product_response
        return "final guide"

    pipeline = NutrientResponsePipeline(
        xai_api_key="test-key",
        research_call=research_call,
        product_call=product_call,
        synthesis_call=synthesis_call,
        coverage_call=_all_sections_covered,
    )

    assert pipeline("Magnesium Glycinate") == "final guide"
    assert research_prompts == [build_research_prompt("Magnesium Glycinate")]
    assert product_prompts == [build_product_prompt("Magnesium Glycinate")]
    assert captured == {
        "nutrient_name": "Magnesium Glycinate",
        "full_prompt": build_prompt("Magnesium Glycinate"),
        "research_response": "research response",
        "product_response": "product response",
    }


def test_response_pipeline_rejects_empty_research_branch():
    pipeline = NutrientResponsePipeline(
        xai_api_key="test-key",
        research_call=lambda prompt: " ",
        product_call=lambda prompt: "product response",
        synthesis_call=lambda name, full, research, product: "final guide",
    )

    with pytest.raises(RuntimeError, match="research branch returned empty output"):
        pipeline("Blank")


def test_response_pipeline_rejects_empty_product_branch():
    pipeline = NutrientResponsePipeline(
        xai_api_key="test-key",
        research_call=lambda prompt: "research response",
        product_call=lambda prompt: " ",
        synthesis_call=lambda name, full, research, product: "final guide",
    )

    with pytest.raises(RuntimeError, match="product branch returned empty output"):
        pipeline("Blank")


def test_response_pipeline_rejects_empty_synthesis():
    pipeline = NutrientResponsePipeline(
        xai_api_key="test-key",
        research_call=lambda prompt: "research response",
        product_call=lambda prompt: "product response",
        synthesis_call=lambda name, full, research, product: " ",
        coverage_call=_all_sections_covered,
    )

    with pytest.raises(RuntimeError, match="synthesis returned empty output"):
        pipeline("Blank")
