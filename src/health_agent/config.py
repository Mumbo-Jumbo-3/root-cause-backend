from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {}

    voyage_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    database_url: str = ""

    trusted_xai_model: str = "grok-4.3"
    unrestricted_xai_model: str = "grok-4.3"
    trusted_xai_reasoning_effort: str = "low"
    unrestricted_xai_reasoning_effort: str = "low"
    anthropic_synthesis_model: str = "claude-sonnet-4-20250514"
    anthropic_judge_model: str = "claude-sonnet-4-20250514"

    embedding_model: str = "voyage-3.5"
    embedding_dimensions: int = 1024

    reranker_model: str = "rerank-2.5"
    reranker_top_k: int = 12
    retrieval_k: int = 10
    keyword_k: int = 30
    retrieval_fetch_k: int = 80
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
    ]

    resources_dir: Path = Path("resources")
    chunk_size: int = 1000
    chunk_overlap: int = 200


def get_settings() -> Settings:
    return Settings()
