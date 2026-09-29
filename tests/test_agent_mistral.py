import asyncio
import types

import pytest

from AI import agent
from experiments.integration_eval import run_integration_eval


def _fake_response(content):
    """Minimal stand-in for an OpenAI chat.completions response object."""
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content),
            logprobs=None,
        )],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _fake_client(on_create):
    """Build an object exposing .chat.completions.create(**kwargs) -> awaitable."""
    class _Completions:
        async def create(self, **kwargs):
            return on_create(kwargs)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))


def test_medium_is_default_response_model_on_scaleway():
    assert agent.DEFAULT_RESPONSE_MODEL_KEY == "MISTRAL_MEDIUM"
    assert agent.MODELS[agent.DEFAULT_RESPONSE_MODEL_KEY].name == "mistral-medium-3.5-128b"
    assert agent.MODELS[agent.DEFAULT_RESPONSE_MODEL_KEY].provider == "scaleway"


def test_small_is_default_routing_model_on_scaleway():
    assert agent.DEFAULT_ROUTING_MODEL_KEY == "MISTRAL_SMALL"
    assert agent.MODELS[agent.DEFAULT_ROUTING_MODEL_KEY].name == "mistral-small-3.2-24b-instruct-2506"
    assert agent.MODELS[agent.DEFAULT_ROUTING_MODEL_KEY].provider == "scaleway"


def test_fallback_models_are_direct_mistral():
    assert agent.MODELS["MISTRAL_LARGE"].name == "mistral-large-2512"
    assert agent.MODELS["MISTRAL_LARGE"].provider == "mistral"
    assert agent.MODELS["MINISTRAL_8B"].name == "ministral-8b-2512"
    assert agent.MODELS["MINISTRAL_8B"].provider == "mistral"
    assert agent.MODELS["MISTRAL_SMALL_DIRECT"].name == "mistral-small-2603"
    assert agent.MODELS["MISTRAL_SMALL_DIRECT"].provider == "mistral"
    assert "MISTRAL_NEMO" in agent.MODELS


def test_integration_eval_uses_production_response_model(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, model_config, system_prompt, temperature=0.5, max_tokens=1024, fallback_model_config=None):
            captured["model_config"] = model_config
            captured["system_prompt"] = system_prompt
            captured["temperature"] = temperature
            captured["max_tokens"] = max_tokens
            captured["fallback_model_config"] = fallback_model_config

        async def generate(self, prompt):
            return "ok", {"input": 1, "output": 1, "total": 2}, None

    monkeypatch.setattr(run_integration_eval, "Agent", FakeAgent)

    response, usage, latency_ms = asyncio.run(
        run_integration_eval.generate_one("system prompt", "full prompt")
    )

    assert response == "ok"
    assert usage == {"input": 1, "output": 1, "total": 2}
    assert latency_ms >= 0
    assert captured["model_config"] == agent.MODELS[agent.DEFAULT_RESPONSE_MODEL_KEY]
    assert captured["fallback_model_config"] == agent.MODELS[agent.RESPONSE_FALLBACK_MODEL_KEY]
    assert captured["temperature"] == 0.1


def test_mistral_provider_uses_direct_api(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setattr(agent, "AsyncOpenAI", FakeAsyncOpenAI)

    llm = agent.Agent(agent.MODELS["MISTRAL_LARGE"], "system prompt")

    assert llm.model == "mistral-large-2512"
    assert llm.provider == "mistral"
    assert llm.fallback_client is None
    assert captured == {
        "api_key": "test-mistral-key",
        "base_url": "https://api.mistral.ai/v1",
    }


def test_mistral_provider_requires_mistral_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        agent.Agent(agent.MODELS["MISTRAL_LARGE"], "system prompt")


def test_agent_builds_primary_and_fallback_clients(monkeypatch):
    base_urls = []

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            base_urls.append(kwargs.get("base_url"))

    monkeypatch.setenv("SCALEWAY_API_KEY", "sk")
    monkeypatch.setenv("MISTRAL_API_KEY", "mk")
    monkeypatch.setattr(agent, "AsyncOpenAI", FakeAsyncOpenAI)

    llm = agent.Agent(
        agent.MODELS["MISTRAL_SMALL"],
        "system prompt",
        temperature=0,
        fallback_model_config=agent.MODELS["MINISTRAL_8B"],
    )

    assert llm.model == "mistral-small-3.2-24b-instruct-2506"
    assert llm.provider == "scaleway"
    assert llm.fallback_model == "ministral-8b-2512"
    assert llm.fallback_provider == "mistral"
    assert llm.fallback_client is not None
    assert "https://api.scaleway.ai/v1" in base_urls
    assert "https://api.mistral.ai/v1" in base_urls


def test_generate_falls_back_on_primary_error(monkeypatch):
    monkeypatch.setenv("SCALEWAY_API_KEY", "sk")
    monkeypatch.setenv("MISTRAL_API_KEY", "mk")

    llm = agent.Agent(
        agent.MODELS["MISTRAL_SMALL"],
        "system prompt",
        fallback_model_config=agent.MODELS["MINISTRAL_8B"],
    )

    fallback_calls = []

    def primary_create(kwargs):
        raise RuntimeError("primary down")

    def fallback_create(kwargs):
        fallback_calls.append(kwargs)
        return _fake_response("FALLBACK_OK")

    llm.client = _fake_client(primary_create)
    llm.fallback_client = _fake_client(fallback_create)

    content, usage, logprobs = asyncio.run(llm.generate("hello"))

    assert content == "FALLBACK_OK"
    assert usage == {"input": 1, "output": 1, "total": 2}
    assert logprobs is None
    assert fallback_calls and fallback_calls[0]["model"] == "ministral-8b-2512"


def test_generate_raises_when_no_fallback(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "mk")

    llm = agent.Agent(agent.MODELS["MISTRAL_LARGE"], "system prompt")

    def primary_create(kwargs):
        raise RuntimeError("primary down")

    llm.client = _fake_client(primary_create)

    with pytest.raises(RuntimeError, match="primary down"):
        asyncio.run(llm.generate("hello"))
