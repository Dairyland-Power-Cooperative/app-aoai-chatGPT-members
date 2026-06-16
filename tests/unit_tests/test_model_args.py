import os
from importlib import import_module, reload
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="function")
def app_module(monkeypatch):
    """Reload backend.settings + app against the gpt-chat-latest dotenv fixture.

    app.py binds app_settings at import time, so settings must be reloaded
    BEFORE app. MS_DEFENDER_ENABLED is forced off so prepare_model_args does
    not try to build a Defender user-security context. monkeypatch restores
    both env vars after each test so the changes don't leak across the suite.
    """
    dotenv_path = os.path.join(
        os.path.dirname(__file__), "dotenv_data", "dotenv_gpt_chat_latest"
    )
    monkeypatch.setenv("DOTENV_PATH", dotenv_path)
    monkeypatch.setenv("MS_DEFENDER_ENABLED", "false")
    reload(import_module("backend.settings"))
    module = reload(import_module("app"))
    yield module


def test_prepare_model_args_uses_chat_contract(app_module):
    model_args = app_module.prepare_model_args(
        {"messages": [{"role": "user", "content": "hello"}]},
        {},
    )

    # Non-reasoning chat model uses the classic "system" role, not "developer".
    assert model_args["messages"][0]["role"] == "system"
    # Chat-model parameter contract.
    assert model_args["max_completion_tokens"] == 1000
    assert "max_tokens" not in model_args
    assert "reasoning_effort" not in model_args
    assert model_args["temperature"] == 0
    assert "top_p" in model_args
    assert "stop" in model_args
    assert model_args["model"] == "gpt-chat-latest"


@pytest.mark.asyncio
async def test_generate_title_uses_chat_contract(app_module, monkeypatch):
    create_mock = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content="A Concise Title"))]
        )
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = create_mock

    async def fake_init_openai_client():
        return mock_client

    monkeypatch.setattr(app_module, "init_openai_client", fake_init_openai_client)

    title = await app_module.generate_title(
        [{"role": "user", "content": "hello there"}]
    )

    assert title == "A Concise Title"
    _, kwargs = create_mock.call_args
    assert kwargs["max_completion_tokens"] == 64
    assert kwargs["temperature"] == 1
    assert "reasoning_effort" not in kwargs
    assert "max_tokens" not in kwargs
