from functools import lru_cache

import voyageai
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI

from health_agent.config import Settings


def get_trusted_grok_model(settings: Settings) -> ChatXAI:
    return ChatXAI(
        model=settings.trusted_xai_model,
        api_key=settings.xai_api_key,
        disable_streaming="tool_calling",
        extra_body={"reasoning_effort": settings.trusted_xai_reasoning_effort},
    )


def get_unrestricted_grok_model(settings: Settings) -> ChatXAI:
    return ChatXAI(
        model=settings.unrestricted_xai_model,
        api_key=settings.xai_api_key,
        disable_streaming="tool_calling",
        extra_body={"reasoning_effort": settings.unrestricted_xai_reasoning_effort},
    )


def get_claude_synthesis_model(settings: Settings) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_synthesis_model,
        api_key=settings.anthropic_api_key,
    )


def get_claude_judge_model(settings: Settings) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_judge_model,
        api_key=settings.anthropic_api_key,
    )


def get_claude_classifier_model(settings: Settings) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_classifier_model,
        api_key=settings.anthropic_api_key,
    )


@lru_cache(maxsize=1)
def voyage_client(api_key: str) -> voyageai.Client:
    # One client per process. Each client caches its own tokenizer, so building one per call leaks memory.
    # Retries ride out a transient 429 or 5xx, which would otherwise fail a whole deploy's ingest.
    return voyageai.Client(api_key=api_key, max_retries=3)
