from health_agent.config import Settings
from health_agent.models import (
    get_claude_synthesis_model,
    get_embeddings_model,
    get_trusted_grok_model,
    get_unrestricted_grok_model,
)


def test_trusted_grok_model():
    settings = Settings(xai_api_key="test-key")
    model = get_trusted_grok_model(settings)
    from langchain_xai import ChatXAI

    assert isinstance(model, ChatXAI)
    assert model.disable_streaming == "tool_calling"
    assert model.extra_body == {"reasoning_effort": "high"}


def test_unrestricted_grok_model():
    settings = Settings(xai_api_key="test-key")
    model = get_unrestricted_grok_model(settings)
    from langchain_xai import ChatXAI

    assert isinstance(model, ChatXAI)
    assert model.disable_streaming == "tool_calling"
    assert model.extra_body == {"reasoning_effort": "high"}


def test_claude_synthesis_model():
    settings = Settings(anthropic_api_key="test-key")
    model = get_claude_synthesis_model(settings)
    from langchain_anthropic import ChatAnthropic

    assert isinstance(model, ChatAnthropic)


def test_embeddings_model():
    settings = Settings(voyage_api_key="test-key")
    model = get_embeddings_model(settings)
    from langchain_voyageai import VoyageAIEmbeddings

    assert isinstance(model, VoyageAIEmbeddings)
