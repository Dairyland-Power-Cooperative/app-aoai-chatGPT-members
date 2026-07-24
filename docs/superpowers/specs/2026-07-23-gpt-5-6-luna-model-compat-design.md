# Design: `gpt-5.6-luna` (reasoning model) compatibility

- **Date:** 2026-07-23
- **Branch:** `gpt-chat-latest-compat`
- **Status:** Implemented (live-verified against the deployment)

## Problem

The app is switching its Azure OpenAI model to **`gpt-5.6-luna`** (OpenAI's GPT-5.6
family, released 2026-07-09). Luna is a **reasoning** model. The immediately prior work
([2026-06-16-gpt-chat-latest-model-compat-design.md](2026-06-16-gpt-chat-latest-model-compat-design.md))
hardcoded a **non-reasoning** chat contract (always send `temperature`/`top_p`/`stop`,
never `reasoning_effort`). That contract is rejected by Luna.

### Live-verified contract (deployment `gpt-5.6-luna`, api-version `2025-04-01-preview`)

Verified with real API calls against `dpc-members-aillm`:

| Parameter | Result |
|---|---|
| `system` role | ✅ works (so does `developer`) — no role change needed |
| `max_completion_tokens` | ✅ required (`max_tokens` → 400 "use max_completion_tokens") |
| `temperature` | ❌ only the default `1` allowed; any other value → 400 |
| `top_p` | ❌ "not supported with this model" → 400 |
| `stop` | ❌ "not supported with this model" → 400 |
| `reasoning_effort` | ✅ supported; enum: `none, low, medium, high, xhigh` (not `minimal`/`max`) |
| `stream` | ✅ works |

The prior branch sends `temperature`/`top_p`/`stop`, so **every chat request 400s** against
Luna until fixed.

## Goal

Make the app send exactly the parameters Luna accepts, **and** keep it working with the
non-reasoning models the org still deploys (`gpt-chat-latest`, `gpt-5.4`, `gpt-5.5`, …) —
selected by explicit config, not name-prefix guessing.

## Decision

**Config-driven contract switch.** Add one explicit boolean setting,
`AZURE_OPENAI_REASONING`, that selects between two request contracts. This is the
"explicit config" option the prior design rejected as over-engineered for a *single* model
target — now justified because two contract types are genuinely in play. It deliberately
avoids the brittle model-name-prefix classification that the prior branch removed.

Rejected: (A) re-hardcode to the reasoning contract (loses the non-reasoning path, needs a
code change to switch back); (B) auto-detect by name prefix (the fragile approach already
removed).

## Detailed design

### 1. `backend/settings.py` — `_AzureOpenAISettings`

Add two fields:

```python
reasoning: bool = False              # AZURE_OPENAI_REASONING
reasoning_effort: str = "medium"     # AZURE_OPENAI_REASONING_EFFORT (luna: none|low|medium|high|xhigh)
```

`reasoning_effort` is a free string (not enum-validated) because the valid set varies by
model; the documented Luna values are `none, low, medium, high, xhigh`.

### 2. `app.py` — `prepare_model_args`

Base dict is contract-independent; branch on `reasoning`. System role stays `"system"`
(verified to work on Luna and on the chat models):

```python
model_args = {
    "messages": messages,
    "max_completion_tokens": app_settings.azure_openai.max_tokens,
    "stream": app_settings.azure_openai.stream,
    "model": app_settings.azure_openai.model,
}
if app_settings.azure_openai.reasoning:
    model_args["reasoning_effort"] = app_settings.azure_openai.reasoning_effort
else:
    model_args["temperature"] = app_settings.azure_openai.temperature
    model_args["top_p"] = app_settings.azure_openai.top_p
    model_args["stop"] = app_settings.azure_openai.stop_sequence
```

Everything downstream (tools, `extra_body`/`data_sources`, secret redaction,
`user_security_context`, logging) is unchanged.

### 3. `app.py` — `generate_title`

Same switch. Reasoning path sets `reasoning_effort="none"` so the 64-token budget goes to
the title rather than reasoning (verified: returns a real title, `finish_reason=stop`):

```python
title_args = {"model": ..., "messages": messages, "max_completion_tokens": 64}
if app_settings.azure_openai.reasoning:
    title_args["reasoning_effort"] = "none"
else:
    title_args["temperature"] = 1
```

### 4. `.env.sample`

Point at Luna and document the switch: `AZURE_OPENAI_MODEL=gpt-5.6-luna`,
`AZURE_OPENAI_MODEL_NAME=gpt-5.6-luna`, `AZURE_OPENAI_REASONING=true`,
`AZURE_OPENAI_REASONING_EFFORT=medium`, with a comment to set `REASONING=false` for
non-reasoning chat models.

### 5. `infra/` (Bicep)

- `main.bicep`: default `openAIModel`/`openAIModelName` → `gpt-5.6-luna`, version
  `2026-07-09`; add `openAIReasoning`/`openAIReasoningEffort` params + the matching
  `AZURE_OPENAI_REASONING` / `AZURE_OPENAI_REASONING_EFFORT` app settings; add
  `openAIDeploymentSkuName` (`GlobalStandard`) / `openAIDeploymentCapacity` (`250`) so IaC
  reproduces the provisioned deployment.
- `cognitiveservices.bicep`: read the deployment SKU from the deployment object
  (`contains(deployment, 'sku') ? deployment.sku : 'Standard'`) — Luna only offers
  `GlobalStandard`/`DataZoneStandard`, not the previously hardcoded `Standard`. Embedding
  deployments (no `sku` key) still default to `Standard`.

## Testing

`tests/unit_tests/test_model_args.py`:

- Existing `dotenv_gpt_chat_latest` tests retained — they now guard the `reasoning=false`
  branch (system role, `temperature`/`top_p`/`stop` present, no `reasoning_effort`).
- New `dotenv_gpt_5_6_luna` fixture (`AZURE_OPENAI_REASONING=true`, with
  `TEMPERATURE`/`TOP_P` set to prove they are dropped) + tests asserting the reasoning
  branch: `reasoning_effort` present, `temperature`/`top_p`/`stop` **absent**,
  `max_completion_tokens` present; `generate_title` sends `reasoning_effort="none"` and no
  `temperature`.

Full `pytest tests/unit_tests` passes (11 tests). End-to-end verified with live calls to the
deployment for both the main-chat and title-generation request shapes (both `200`,
`finish_reason=stop`).

## Deployment / activation

1. Provision the deployment (done): `gpt-5.6-luna`, model version `2026-07-09`,
   `GlobalStandard`, capacity 250, in `dpc-members-aillm` / `rg-members-oai-eus2`.
2. The new code must be deployed to the App Service **before** the app settings are flipped
   — the App Service deploys via GitHub Actions on push to `main`. Flipping settings against
   the old code would send `temperature`/`top_p`/`stop` and 400.
3. After deploy, set on `dpc-members-voltwrite-app`: `AZURE_OPENAI_MODEL=gpt-5.6-luna`,
   `AZURE_OPENAI_MODEL_NAME=gpt-5.6-luna`, `AZURE_OPENAI_REASONING=true`,
   `AZURE_OPENAI_REASONING_EFFORT=<none|low|medium|high|xhigh>`; restart; verify.

## Risks / trade-offs

- `reasoning_effort` is not enum-validated in settings; an unsupported value for a given
  model surfaces as a 400 at request time. Documented Luna values are noted in `.env.sample`.
- Reasoning consumes `max_completion_tokens`; keep the budget generous for main chat and use
  `reasoning_effort="none"` for the short title call.
