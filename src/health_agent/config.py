from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {}

    voyage_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    database_url: str = ""

    trusted_xai_model: str = "grok-4.3"
    unrestricted_xai_model: str = "grok-4.3"
    trusted_xai_reasoning_effort: str = "medium"
    unrestricted_xai_reasoning_effort: str = "medium"
    anthropic_synthesis_model: str = "claude-sonnet-4-6"
    anthropic_judge_model: str = "claude-sonnet-4-6"
    anthropic_classifier_model: str = "claude-haiku-4-5-20251001"

    embedding_model: str = "voyage-4-large"
    embedding_dimensions: int = 1024

    reranker_model: str = "rerank-2.5"
    reranker_top_k: int = 12
    retrieval_strategy: Literal["legacy", "hybrid_v2"] = "legacy"
    retrieval_k: int = 10
    keyword_k: int = 30
    retrieval_fetch_k: int = 80
    keyword_fetch_k: int = 80
    rrf_k: int = 60
    keyword_weight: float = 0.4
    vector_weight: float = 0.6
    reranker_score_threshold: float = 0.3

    trusted_x_accounts: list[str] = [
        "helios_movement",
        "grimhood",
        "aestheticprimal",
        "hubermanlab",
        "foundmyfitness",
        "outdoctrination",
        "AJA_Cortes",
        "ChrisMasterjohn",
        "BradStanfieldMD",
        "theholisticnick",
    ]

    womens_health_x_accounts: list[str] = [
        "BioavailableNd",
        "iam_preethi",
        "LilyNicholsRDN",
        "celestialbe1ing",
    ]

    resources_dir: Path = Path("resources")
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_settings() -> Settings:
    return Settings()
