# gpt-chat-latest Model Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app send exactly the parameters the non-reasoning `gpt-chat-latest` chat model accepts, and delete the brittle name-prefix model-family guessing.

**Architecture:** Remove the three `is_*_model` prefix properties and the `reasoning_effort` field from `_AzureOpenAISettings`, then flatten the two request builders (`prepare_model_args`, `generate_title`) in `app.py` to a single hardcoded chat contract: `system` role, `temperature`/`top_p`/`stop`, `max_completion_tokens`, and no `reasoning_effort`. Lock the new contract in with unit tests against the model-arg construction, which is currently untested.

**Tech Stack:** Python, Quart, OpenAI Python SDK (`>=2.11.0`), Pydantic settings, pytest + pytest-asyncio.

**Spec:** [docs/superpowers/specs/2026-06-16-gpt-chat-latest-model-compat-design.md](../specs/2026-06-16-gpt-chat-latest-model-compat-design.md)

---

## File Structure

- **Modify** [backend/settings.py](../../../backend/settings.py) — remove `reasoning_effort` field and the `is_o_series_model` / `is_legacy_model` / `is_gpt5_series_model` properties from `_AzureOpenAISettings`. Keep `model_name` and `protected_namespaces=('settings_',)`.
- **Modify** [app.py](../../../app.py) — `prepare_model_args` (~L241-303) and `generate_title` (~L1051-1080): drop branching, hardcode the chat contract.
- **Modify** [.env.sample](../../../.env.sample) — remove `AZURE_OPENAI_REASONING_EFFORT`, set model example to `gpt-chat-latest`.
- **Create** `tests/unit_tests/dotenv_data/dotenv_gpt_chat_latest` — env fixture for the new tests.
- **Create** [tests/unit_tests/test_model_args.py](../../../tests/unit_tests/test_model_args.py) — unit tests for the two request builders.

---

## Task 1: Replace model-family branching with the chat contract (TDD)

**Files:**
- Create: `tests/unit_tests/dotenv_data/dotenv_gpt_chat_latest`
- Create: `tests/unit_tests/test_model_args.py`
- Modify: `backend/settings.py` (remove `reasoning_effort` field + 3 properties)
- Modify: `app.py` (`prepare_model_args`, `generate_title`)

- [ ] **Step 1: Add the dotenv fixture**

Create `tests/unit_tests/dotenv_data/dotenv_gpt_chat_latest` with exactly:

```
AZURE_OPENAI_MODEL=gpt-chat-latest
AZURE_OPENAI_MODEL_NAME=gpt-chat-latest
AZURE_OPENAI_KEY=dummy
AZURE_OPENAI_TEMPERATURE=0
AZURE_OPENAI_TOP_P=1.0
AZURE_OPENAI_MAX_TOKENS=1000
AZURE_OPENAI_STOP_SEQUENCE=
AZURE_OPENAI_SYSTEM_MESSAGE=You are an AI assistant that helps people find information.
AZURE_OPENAI_PREVIEW_API_VERSION=2025-04-01-preview
AZURE_OPENAI_STREAM=False
AZURE_OPENAI_ENDPOINT=https://dummy.openai.azure.com/
```

- [ ] **Step 2: Write the tests (one red, one regression guard)**

Create `tests/unit_tests/test_model_args.py`:

```python
import os
from importlib import import_module, reload
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="function")
def app_module():
    """Reload backend.settings + app against the gpt-chat-latest dotenv fixture.

    app.py binds app_settings at import time, so settings must be reloaded
    BEFORE app. MS_DEFENDER_ENABLED is forced off so prepare_model_args does
    not try to build a Defender user-security context.
    """
    dotenv_path = os.path.join(
        os.path.dirname(__file__), "dotenv_data", "dotenv_gpt_chat_latest"
    )
    os.environ["DOTENV_PATH"] = dotenv_path
    os.environ["MS_DEFENDER_ENABLED"] = "false"
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
```

- [ ] **Step 3: Run the tests to confirm the red**

Run: `python -m pytest tests/unit_tests/test_model_args.py -v`
Expected: `test_prepare_model_args_uses_chat_contract` **FAILS** on the role assertion (`assert 'developer' == 'system'`). `test_generate_title_uses_chat_contract` may already PASS (the target model's title params are already correct) — that is fine; it is a regression guard.

- [ ] **Step 4: Remove the `reasoning_effort` field and the 3 prefix properties from `backend/settings.py`**

In `_AzureOpenAISettings`, delete the `reasoning_effort` field line:

```python
    reasoning_effort: str = "none"
```

And delete these three properties in full:

```python
    @property
    def is_o_series_model(self) -> bool:
        name = (self.model_name or self.model or "").lower()
        return len(name) >= 2 and name[0] == "o" and name[1].isdigit()

    @property
    def is_legacy_model(self) -> bool:
        name = (self.model_name or self.model or "").lower()
        return name.startswith(("gpt-35", "gpt-4"))

    @property
    def is_gpt5_series_model(self) -> bool:
        name = (self.model_name or self.model or "").lower()
        return name.startswith("gpt-5")
```

Leave `model_name: Optional[str] = None` and `protected_namespaces=('settings_',)` in place (pydantic requires the latter to allow the `model_name` field).

- [ ] **Step 5: Flatten `prepare_model_args` in `app.py`**

Replace the `system_role` line + system-message block. Old:

```python
    system_role = "system" if app_settings.azure_openai.is_legacy_model else "developer"
    if not app_settings.datasource:
        messages = [
            {
                "role": system_role,
                "content": app_settings.azure_openai.system_message
            }
        ]
```

New:

```python
    if not app_settings.datasource:
        messages = [
            {
                "role": "system",
                "content": app_settings.azure_openai.system_message
            }
        ]
```

Then replace the `model_args` dict and the branching that follows it. Old:

```python
    model_args = {
        "messages": messages,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,
    }

    if app_settings.azure_openai.is_legacy_model:
        model_args["max_tokens"] = app_settings.azure_openai.max_tokens
    else:
        model_args["max_completion_tokens"] = app_settings.azure_openai.max_tokens

    if app_settings.azure_openai.is_o_series_model:
        model_args["reasoning_effort"] = app_settings.azure_openai.reasoning_effort
    else:
        model_args["temperature"] = app_settings.azure_openai.temperature
        model_args["top_p"] = app_settings.azure_openai.top_p
        model_args["stop"] = app_settings.azure_openai.stop_sequence
        if app_settings.azure_openai.is_gpt5_series_model:
            model_args["reasoning_effort"] = app_settings.azure_openai.reasoning_effort
```

New:

```python
    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_completion_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,
    }
```

Leave everything after this (tools, `extra_body` / `data_sources`, secret redaction, `user_security_context`, logging, `return model_args`) unchanged.

- [ ] **Step 6: Flatten `generate_title` in `app.py`**

Replace the `title_kwargs` block. Old:

```python
        title_kwargs = {"model": app_settings.azure_openai.model, "messages": messages}
        if app_settings.azure_openai.is_legacy_model:
            title_kwargs["max_tokens"] = 64
        else:
            title_kwargs["max_completion_tokens"] = 64
        if app_settings.azure_openai.is_o_series_model:
            title_kwargs["reasoning_effort"] = "none"
        else:
            title_kwargs["temperature"] = 1
            if app_settings.azure_openai.is_gpt5_series_model:
                title_kwargs["reasoning_effort"] = "none"
        response = await azure_openai_client.chat.completions.create(**title_kwargs)
```

New:

```python
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model,
            messages=messages,
            temperature=1,
            max_completion_tokens=64,
        )
```

- [ ] **Step 7: Run the tests to confirm green**

Run: `python -m pytest tests/unit_tests/test_model_args.py -v`
Expected: both tests **PASS**.

- [ ] **Step 8: Commit**

```bash
git add backend/settings.py app.py tests/unit_tests/test_model_args.py tests/unit_tests/dotenv_data/dotenv_gpt_chat_latest
git commit -m "Replace model-family branching with gpt-chat-latest chat contract

System message uses the classic 'system' role, always send
temperature/top_p/stop + max_completion_tokens, never reasoning_effort.
Remove the brittle is_legacy/is_o_series/is_gpt5 prefix properties and
the reasoning_effort setting. Add unit tests for the request builders.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Update `.env.sample`

**Files:**
- Modify: `.env.sample`

- [ ] **Step 1: Remove the reasoning-effort var and update the model example**

Delete this line entirely:

```
AZURE_OPENAI_REASONING_EFFORT=none
```

Change the model-name example from:

```
AZURE_OPENAI_MODEL_NAME=gpt-35-turbo-16k
```

to:

```
AZURE_OPENAI_MODEL_NAME=gpt-chat-latest
```

Leave `AZURE_OPENAI_PREVIEW_API_VERSION=2025-04-01-preview` unchanged.

- [ ] **Step 2: Verify the edits**

Run: `grep -n "REASONING_EFFORT\|MODEL_NAME" .env.sample`
Expected: no `REASONING_EFFORT` line; `AZURE_OPENAI_MODEL_NAME=gpt-chat-latest` present.

- [ ] **Step 3: Commit**

```bash
git add .env.sample
git commit -m "Update .env.sample for gpt-chat-latest (drop reasoning_effort)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Full-suite verification + dead-reference sweep

**Files:** none (verification only)

- [ ] **Step 1: Confirm no lingering references to the removed symbols**

Run: `grep -rn "is_legacy_model\|is_o_series_model\|is_gpt5_series_model\|reasoning_effort" app.py backend/`
Expected: **no matches**.

- [ ] **Step 2: Run the full unit suite**

Run: `python -m pytest tests/unit_tests -v`
Expected: all tests PASS (existing settings/utils tests + the two new model-arg tests). Removing the `reasoning_effort` field and the properties must not break `test_settings.py`.

- [ ] **Step 3 (manual, outside this environment): Live deployment verification**

These require the real Azure `gpt-chat-latest` deployment and cannot be run here:
1. Send a chat message end-to-end — confirm a 200 (no "Unrecognized request argument" / role errors) with `system` role + `max_completion_tokens`.
2. If the deployment rejects `max_completion_tokens`, switch both call sites to `max_tokens` (documented form for the chat family) and re-run Task 1 tests with `max_completion_tokens` swapped for `max_tokens` in the assertions.
3. If the **On Your Data** datasource feature is in use, confirm `data_sources` grounding works with this model.

---

## Self-Review

**1. Spec coverage:**
- Remove 3 prefix properties + `reasoning_effort` field → Task 1 Step 4. ✓
- `prepare_model_args` system role + flat chat contract → Task 1 Step 5. ✓
- `generate_title` simplification → Task 1 Step 6. ✓
- `.env.sample` updates → Task 2. ✓
- Unit tests for both builders → Task 1 Steps 1-2, 7. ✓
- Full-suite regression + live-verification notes → Task 3. ✓

**2. Placeholder scan:** No TBD/TODO; all code and commands are concrete. Task 3 Step 3 is explicitly a manual, environment-dependent step (not a placeholder), matching the spec's Verification section. ✓

**3. Type/name consistency:** Fixture/test names (`app_module`, `dotenv_gpt_chat_latest`, `test_prepare_model_args_uses_chat_contract`, `test_generate_title_uses_chat_contract`) are used consistently. The `app_module` fixture reloads `backend.settings` then `app` (correct order, since `app` imports `app_settings` by value). `prepare_model_args(request_body, request_headers)` is called with both positional args. ✓
