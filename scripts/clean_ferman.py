"""
Clean the ferman_*.md resources extracted from George Ferman's PDF programs.

The PDFs open with a medical disclaimer and a "This program was created by..."
byline, and every page ends with a "(lesson N) written by ... Sold to" footer.
Extraction kept all of it, so ingest titled documents after the disclaimer and
chunks carried the footers. This script:

- removes the opening disclaimer and byline lines,
- removes page footers, rejoining sentences they split,
- replaces the lesson line with an H1 title naming the program and lesson,
- turns wrapped prose and reference URLs that extraction made into headings
  back into plain text.

In-body "Disclaimer: ..." notes (dosing and lab-test cautions) are content and
are kept. Safe to re-run: cleaned files come out unchanged.

    uv run python scripts/clean_ferman.py
"""

import re
import sys
from pathlib import Path

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"

PROGRAMS = {"gut": "Gut Health Handbook", "system": "The System"}

LESSON_TITLES = {
    "gut": {
        1: "Resetting the Digestive System",
        2: "Understanding Mast Cells",
        3: "A More In-Depth Discussion About Nutrition",
        4: "Learning How to Properly Use Supplements",
        5: "Understanding the Effects of Toxins",
        6: "A Discussion About Genes",
    },
    "system": {
        1: "Light and the Circadian Rhythm",
        2: "Enter Non-Native Electromagnetic Fields (nnEMFs)",
        3: "Understanding the Effects of Toxins",
        4: "Stress and Its Effects",
        5: "Understanding Nutrition",
        6: "Resetting the Digestive System",
        7: "Understanding Mast Cells",
        8: "Understanding Neurotransmitters",
        9: "Learning How to Properly Use Supplements",
        10: "Understanding Your Metabolism",
        11: "Understanding and Improving the Function of Key Hormone Receptors",
        12: "A Discussion About Genes",
    },
}

FRONT_MATTER_LINE = re.compile(
    r"^[ \t]*(?:#+[ \t]*)?(?:"
    r"DISCLAIMER: This PDF does not provide medical advice\."
    r"|No content on this PDF file is intended to substitute for professional"
    r"|medical advice, diagnosis, or treatment\."
    r"|This program was created by George @Helios_movement\."
    r")[ \t]*\n",
    re.IGNORECASE | re.MULTILINE,
)

SOLD_TO = r"[ \t]*(?:\*\*)?Sold to(?:\*\*)?[ \t]*"
PAGE_FOOTER = re.compile(
    r"\s*^[ \t]*(?:The system(?: V\d\.\d)?|Gut health hand?book) \(lesson \d+\) "
    r"written by George @Helios_Movement(?:" + SOLD_TO + r")?[ \t]*$"
    r"(?:\s*^" + SOLD_TO + r"$)?\s*",
    re.IGNORECASE | re.MULTILINE,
)
STRAY_SOLD_TO = re.compile(r"^" + SOLD_TO + r"\n", re.MULTILINE)

# "**Lesson 2: Understanding mast cells (aside...)**" or "## The system lesson 4: ..."
LESSON_LINE = re.compile(
    r"\A\s*(?:#+[ \t]*)?(\*\*)?(?:The system )?lesson (\d+):(.*?)(?(1)\*\*|$)[ \t]*\n",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

HEADING = re.compile(r"^(#+)[ \t]+(.*?)[ \t]*$")
SENTENCE_END = tuple('.!?:;)"”’*')


def _join_footer_gap(match: re.Match[str]) -> str:
    """Replace a page footer with the break the text needs on either side of it."""
    before = match.string[: match.start()].rstrip()
    after = match.string[match.end() :]
    if not before or not after:
        return "\n"
    if not before.endswith(SENTENCE_END) and after[0].islower():
        return " "  # the footer split a sentence
    return "\n\n"


def _demote_broken_headings(text: str) -> str:
    """Turn headings that are really wrapped prose or reference URLs into text."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        heading = HEADING.match(lines[i])
        if not heading:
            out.append(lines[i])
            i += 1
            continue

        body = heading.group(2)
        if " " not in body and re.search(r"[/=]", body):
            out.append(body)  # a reference URL or URL fragment
            i += 1
            continue

        # A long heading that stops mid-sentence and runs on into lowercase text.
        j = i
        while len(body) >= 40 and not body.endswith(SENTENCE_END):
            k = j + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k == len(lines):
                break
            nxt = lines[k].strip()
            next_heading = HEADING.match(nxt)
            nxt_text = next_heading.group(2) if next_heading else nxt
            if not nxt_text[:1].islower():
                break
            body = f"{body} {nxt_text}"
            j = k
        out.append(body if j > i else lines[i])
        i = j + 1
    return "\n".join(out)


def clean(text: str, program: str, lesson: int) -> str:
    text = FRONT_MATTER_LINE.sub("", text)
    text = PAGE_FOOTER.sub(_join_footer_gap, text)
    text = STRAY_SOLD_TO.sub("", text)

    title = f"# {PROGRAMS[program]}, Lesson {lesson}: {LESSON_TITLES[program][lesson]}"
    text = text.lstrip()
    if not text.startswith(title):
        lesson_line = LESSON_LINE.match(text)
        aside = ""
        if lesson_line:
            text = text[lesson_line.end() :]
            # Keep reading notes like "(do not apply anything in this file for the first 2 months)".
            note = re.search(r"\(([^()]{20,})\)\s*\.?\s*$", " ".join(lesson_line.group(3).split()))
            if note:
                aside = f"({note.group(1)})\n\n"
        text = f"{title}\n\n{aside}{text.lstrip()}"

    text = _demote_broken_headings(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip() + "\n"


def main() -> int:
    for path in sorted(RESOURCES_DIR.glob("ferman_*.md")):
        match = re.fullmatch(r"ferman_(gut|system)_(\d+)\.md", path.name)
        if not match:
            print(f"skipping {path.name}: unexpected name", file=sys.stderr)
            continue
        original = path.read_text(encoding="utf-8")
        cleaned = clean(original, match.group(1), int(match.group(2)))
        if cleaned != original:
            path.write_text(cleaned, encoding="utf-8")
            print(f"cleaned {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
