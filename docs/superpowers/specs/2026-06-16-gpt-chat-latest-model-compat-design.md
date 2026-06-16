# Design: `gpt-chat-latest` model compatibility

- **Date:** 2026-06-16
- **Branch:** `gpt-chat-latest-compat`
- **Status:** Approved (pending spec review)

## Problem

The app is switching its underlying Azure AI Foundry model to **`gpt-chat-latest`**, a
non-reasoning chat model. Today the request parameters are chosen by guessing the model
family from the deployment/model **name prefix** ([backend/settings.py:135-148](../../../backend/settings.py#L135-L148)):

- `is_legacy_model` → name starts with `gpt-35` / `gpt-4`
- `is_o_series_model` → name starts with `o` + digit (`o1`, `o3`, …)
- `is_gpt5_series_model` → name starts with `gpt-5`

`gpt-chat-latest` matches **none** of these, so it falls into the "modern reasoning-style"
default branch and is sent parameters it does not accept.

### Concrete incompatibilities

| Param sent today | `gpt-chat-latest` expectation | Verdict |
|---|---|---|
| system message `role: "developer"` | classic `role: "system"` | ❌ **primary bug** |
| `max_completion_tokens` | accepted on `2025-04-01-preview` | ✅ (verify on deployment) |
| `temperature`, `top_p`, `stop` | supported (chat, not reasoning) | ✅ |
| `reasoning_effort` | **not supported** — 400 "Unrecognized request argument" | ✅ not sent, but only by luck |

Two problems:

1. **The `developer` role breaks the chat model.** `developer` is OpenAI's *reasoning*-model
   interface (o-series / GPT-5 reasoning). `gpt-chat-latest` is the non-reasoning chat
   variant and uses the standard schema, where the instruction message is `system`.
2. **The current safety is accidental.** `reasoning_effort` is omitted only because the name
   does not start with `gpt-5`. A deployment named `gpt-5-chat-latest` (the same model on
   some surfaces) would be classified as gpt-5-series and sent `reasoning_effort`, causing a
   400. The name-prefix guessing is fragile.

### Assumption

`gpt-chat-latest` follows the non-reasoning chat-completions contract (same family as Azure's
`gpt-5-chat` / `gpt-5-chat-latest`): `system` role, supports `temperature`/`top_p`/`stop`, and
**rejects** `reasoning_effort`. This is the basis of the design. It must be confirmed against
the live deployment (see Verification) since it cannot be exercised from the dev environment.

## Goal

Make `gpt-chat-latest` work correctly by sending exactly the parameters it accepts, and remove
the brittle name-prefix guessing.

## Non-goals

- Supporting o-series / GPT-5 *reasoning* models (explicitly dropped — single-model target).
- Changing the On Your Data, function-calling, MS Defender, or streaming flows beyond the
  shared parameter construction.
- Touching the unused `seed` / `presence_penalty` / `frequency_penalty` / `logit_bias` /
  `choices_count` / `user` settings fields (already not sent by `prepare_model_args`).

## Decision

**Approach A — Clean replace.** Delete the prefix-based classification and hardcode the
non-reasoning chat contract. Confirmed choices:

- Token limit parameter: **`max_completion_tokens`** (forward-compatible; keep current key).
- **Fully remove** the three prefix properties and the `reasoning_effort` field/env.

Rejected alternatives: (B) keep branching and only fix the chat case — retains the fragile
prefix logic and is not a clean replace; (C) introduce an explicit `AZURE_OPENAI_MODEL_FAMILY`
config — more robust long-term but over-engineered for a single-model target.

## Detailed design

### 1. `backend/settings.py`

In `_AzureOpenAISettings`:

- **Remove** the `reasoning_effort: str = "none"` field.
- **Remove** the `is_o_series_model`, `is_legacy_model`, and `is_gpt5_series_model` properties.
- **Keep** `model_name: Optional[str] = None` and `protected_namespaces=('settings_',)` (the
  latter is required by pydantic so the `model_name` field is allowed).

### 2. `app.py` — `prepare_model_args` ([app.py:241-303](../../../app.py#L241-L303))

- System message role becomes the literal `"system"` (remove the `system_role` variable and
  the `is_legacy_model` ternary).
- Collapse the token / param branching into one flat `model_args` dict:

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

- **No** `reasoning_effort` key under any condition.
- Everything downstream (tools, `extra_body.data_sources`, `user_security_context`, secret
  redaction, logging) is unchanged.

### 3. `app.py` — `generate_title` ([app.py:1051-1080](../../../app.py#L1051-L1080))

Replace the `title_kwargs` branching with a direct call:

```python
response = await azure_openai_client.chat.completions.create(
    model=app_settings.azure_openai.model,
    messages=messages,
    temperature=1,
    max_completion_tokens=64,
)
```

### 4. `.env.sample`

- **Remove** `AZURE_OPENAI_REASONING_EFFORT=none`.
- Set the model example to the new target, e.g. `AZURE_OPENAI_MODEL_NAME=gpt-chat-latest`.
- Keep `AZURE_OPENAI_PREVIEW_API_VERSION=2025-04-01-preview`.

## Testing strategy

The model-arg construction is currently **untested**. Add `tests/unit_tests` coverage that
locks in the new contract:

- `prepare_model_args` (no datasource): the injected instruction message has `role == "system"`;
  `model_args` contains `max_completion_tokens` and **not** `max_tokens`; contains
  `temperature`, `top_p`, `stop`; does **not** contain `reasoning_effort`.
- `generate_title`: the `chat.completions.create` call is made with `max_completion_tokens=64`,
  `temperature=1`, and no `reasoning_effort` (assert via a mocked async client).

Run the existing suite (`pytest tests/unit_tests`) to confirm settings/datasource tests still
pass after removing the properties and `reasoning_effort` field.

## Verification (cannot be done from dev env)

A real request must be made against the actual `gpt-chat-latest` deployment to confirm:

1. The chat completion succeeds with `system` role + `max_completion_tokens` (no 400).
2. If the deployment rejects `max_completion_tokens`, fall back to `max_tokens` (documented
   form for the chat family) — this is the one residual uncertainty.
3. **On Your Data:** if the datasource feature is used, confirm `data_sources` works with this
   model — there are reports of GPT-5-family limitations with Azure AI Search grounding. This
   is an Azure-side capability check, not affected by the parameter changes.

## Risks / trade-offs

- Dropping the branching means pointing the app at an o-series / reasoning model later requires
  a code change. Accepted — the app is standardizing on `gpt-chat-latest`.
- Diverges further from the upstream Microsoft `sample-app-aoai-chatGPT`, increasing future
  merge friction. Accepted.
- `max_completion_tokens` acceptance on the specific deployment is unconfirmed until live
  verification; `max_tokens` is the documented fallback.
