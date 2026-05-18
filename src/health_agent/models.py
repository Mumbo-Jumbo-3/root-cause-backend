from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIEmbeddings
from langchain_xai import ChatXAI

from health_agent.config import Settings


def get_trusted_grok_model(settings: Settings) -> ChatXAI:
    return ChatXAI(
        model=settings.trusted_xai_model,
        api_key=settings.xai_api_key,
        extra_body={"reasoning_effort": settings.trusted_xai_reasoning_effort},
    )


def get_trusted_grok_x_search_model(
    settings: Settings, x_handles: list[str]
) -> ChatXAI:
    return ChatXAI(
        model=settings.trusted_xai_model,
        api_key=settings.xai_api_key,
        extra_body={"reasoning_effort": settings.trusted_xai_reasoning_effort},
        search_parameters={
            "mode": "on",
            "sources": [{"type": "x", "x_handles": x_handles}],
        },
    )


def get_unrestricted_grok_model(settings: Settings) -> ChatXAI:
    return ChatXAI(
        model=settings.unrestricted_xai_model,
        api_key=settings.xai_api_key,
        extra_body={"reasoning_effort": settings.unrestricted_xai_reasoning_effort},
    )


def get_unrestricted_grok_x_search_model(settings: Settings) -> ChatXAI:
    return ChatXAI(
        model=settings.unrestricted_xai_model,
        api_key=settings.xai_api_key,
        extra_body={"reasoning_effort": settings.unrestricted_xai_reasoning_effort},
        search_parameters={
            "mode": "on",
            "sources": [{"type": "x"}],
        },
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


def get_embeddings_model(settings: Settings) -> VoyageAIEmbeddings:
    return VoyageAIEmbeddings(
        model=settings.embedding_model,
        voyage_api_key=settings.voyage_api_key,
    )
