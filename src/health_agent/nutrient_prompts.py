"""Canonical prompt template for nutrient guides.

Single source of truth for the user-visible prompt attached to each nutrient.
Imported by both the API layer (`content.py`) and the offline fill script
(`scripts/fill_blank_nutrient_responses.py`), so changing the wording here
propagates everywhere — no need to edit per-nutrient `meta.json` files.
"""

from __future__ import annotations


PROMPT_TEMPLATE = (
    "Research {nutrient}. How does it affect human health/nutrition/performance? What is it for? "
    "How is it best ingested in the optimal amounts (food, drinks, supplements, forms, dosing, risks, interactions, etc)? "
    "What are the top 3 highest quality/purity {nutrient} products i can buy?"
)


def _prompt_form(nutrient_name: str) -> str:
    return " ".join(w.lower() if len(w) > 1 else w for w in nutrient_name.split())


def build_prompt(nutrient_name: str) -> str:
    return PROMPT_TEMPLATE.format(nutrient=_prompt_form(nutrient_name))
