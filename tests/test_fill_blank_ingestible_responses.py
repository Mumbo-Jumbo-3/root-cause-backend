import json
from pathlib import Path

from scripts.fill_blank_ingestible_responses import (
    build_prompt,
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
        "How is it best ingested in the optimal amounts (food, supplements, etc)? "
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

    def fail_call_model(*args):
        raise AssertionError("call_model should not be called during dry-run")

    result = fill_blank_ingestibles(
        root,
        api_key="",
        dry_run=True,
        call_model=fail_call_model,
    )

    assert result.blank_count == 1
    assert result.written_count == 0
    assert (entry_dir / "response.md").read_text(encoding="utf-8") == ""


def test_fill_blank_ingestibles_writes_only_blank_responses(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    blank_dir = write_ingestible(root, "blank", "Blank", "")
    filled_dir = write_ingestible(root, "filled", "Filled", "keep me")
    prompts: list[str] = []

    def call_model(api_key: str, model: str, prompt: str, timeout_seconds: float) -> str:
        prompts.append(prompt)
        return "generated response"

    result = fill_blank_ingestibles(
        root,
        api_key="test-key",
        model="grok-4.3",
        call_model=call_model,
    )

    assert result.blank_count == 1
    assert result.written_count == 1
    assert result.failures == ()
    assert (
        blank_dir / "response.md"
    ).read_text(encoding="utf-8") == "generated response\n"
    assert (filled_dir / "response.md").read_text(encoding="utf-8") == "keep me"
    assert prompts == [build_prompt("Blank")]


def test_fill_blank_ingestibles_reports_empty_model_output(tmp_path: Path):
    root = tmp_path / "ingestibles"
    root.mkdir()
    entry_dir = write_ingestible(root, "blank", "Blank", "")

    def call_model(api_key: str, model: str, prompt: str, timeout_seconds: float) -> str:
        return "   "

    result = fill_blank_ingestibles(
        root,
        api_key="test-key",
        call_model=call_model,
    )

    assert result.blank_count == 1
    assert result.written_count == 0
    assert result.failures == ("blank: model returned empty output",)
    assert (entry_dir / "response.md").read_text(encoding="utf-8") == ""
