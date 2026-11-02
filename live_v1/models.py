"""Curated pool of LLMs for the live performance.

Selection principles:

1. Diversity across providers (different training data, different alignment
   schools) — the point of the piece is polyphony, not consensus.
2. Capable instruct / chat models suitable for interpreting Oliveros and
   improvising in the field.  For very large choirs (``n`` > 30), backups
   and the extended pool extend the list.
3. Each model is reachable through OpenRouter via one API key.

``PRIMARY_POOL`` lists **30** strong models; ``pick_pool(n)`` takes the first
``n`` for ``n ≤ 30``.  **Default CLI `--agents` is 10** — a cheap, rich choir;
raise only when you want more voices and cost.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    slug: str             # OpenRouter model id
    short_name: str       # for terminal/log display
    provider: str         # human-readable provider
    expected_ttft_ms: int  # rough figure for monitoring
    disable_reasoning: bool = False  # hybrid reasoning models (e.g. NVIDIA
    # Nemotron) otherwise spend their whole token budget in a `reasoning`
    # channel and return an empty `content`. Setting this sends OpenRouter
    # `reasoning={"enabled": False}` so the model replies directly.
    skip_json_mode: bool = False     # some models (e.g. nemotron-3-super) return
    # invalid JSON when `response_format=json_object` is set. Set this flag to
    # skip json mode on ALL attempts and rely on _robust_json parsing instead.


# Thirty flagship-class models.
# The FIRST 10 are chosen for maximum provider diversity — one per house.
# pick_pool(10) takes exactly these; pick_pool(30) uses all.
PRIMARY_POOL: list[ModelSpec] = [
    # --- 9-voice diverse choir (one per provider, updated June 2026) ---
    # voice_00: GPT-4o — OpenAI's most expressive chat-instruct model for
    # creative/philosophical tasks. gpt-5-chat is technically newer but more
    # utilitarian; gpt-4o produces richer, more distinctive Phase-1 language.
    # Fast (~1-2s), no reasoning overhead, proven over many sessions.
    ModelSpec("openai/gpt-4o",                            "gpt-4o",           "OpenAI",      250),
    # voice_01: Claude Opus 4.8 — most capable Anthropic model (May 2026)
    ModelSpec("anthropic/claude-opus-4.8",                "claude-opus-4.8",  "Anthropic",   700),
    # voice_02: Gemini 2.5 Flash — fast Google model, unchanged
    ModelSpec("google/gemini-2.5-flash",                  "gemini-2.5-flash", "Google",      220),
    # voice_03: Llama 4 Maverick — MoE 400B total, 17B active (Apr 2025)
    ModelSpec("meta-llama/llama-4-maverick",              "llama-4-maverick", "Meta",        250),
    # voice_04: DeepSeek Chat — unchanged
    ModelSpec("deepseek/deepseek-chat",                   "deepseek-chat",    "DeepSeek",    520),
    # voice_05: Mistral Large 3 (2512) — already the latest, unchanged
    ModelSpec("mistralai/mistral-large-2512",             "mistral-large",    "Mistral",     400),
    # voice_06: Qwen3-Next 80B (A3B) instruct — Alibaba's fast non-reasoning
    # voice. Both the `qwen3.7-max` and `qwen3.7-plus` reasoning tiers crawled
    # in Phase 2 (~24-27s/breath, timeouts); this instruct variant answers in
    # ~5s and is the same Qwen that ran reliably in earlier sessions.
    ModelSpec("qwen/qwen3-next-80b-a3b-instruct",         "qwen3-next-80b",   "Alibaba",     280),
    # voice_07: Grok 4.3 — xAI, unchanged
    ModelSpec("x-ai/grok-4.3",                            "grok-4.3",         "xAI",         450),
    # voice_08: NVIDIA Nemotron 3 Super 120B — the full-size Nemotron MoE.
    # The nano-9b was too small and gave shallow Phase-1 output. The super-120b
    # with disable_reasoning + skip_json_mode gives rich interpretations at ~5s.
    # Acts as a reserve speaker: if any of voices 0-7 says "wants_to_be_heard=no"
    # in Phase 1, this voice fills the acoustic gap on its physical speaker slot.
    ModelSpec("nvidia/nemotron-3-super-120b-a12b",        "nemotron-super",   "NVIDIA",      350,
              disable_reasoning=True, skip_json_mode=True),
    # voice_09: Nous Hermes 4 70B — Nous Research's flagship creative-instruct
    # model. Fast (~5s), architecturally distinct, and gives poetic Phase-1
    # language ("persistent perturbation in the attention matrix"). 10th voice
    # for a 5×2 grid on the projector.
    ModelSpec("nousresearch/hermes-4-70b",                "hermes-4-70b",     "Nous",        300),
    # --- extended choir (positions 11-30) ---
    ModelSpec("openai/gpt-4o-mini",                       "gpt-4o-mini",      "OpenAI",      250),
    ModelSpec("openai/gpt-4.1-mini",                      "gpt-4.1-mini",     "OpenAI",      280),
    ModelSpec("openai/gpt-5-mini",                        "gpt-5-mini",       "OpenAI",      300),
    ModelSpec("anthropic/claude-haiku-4.5",               "claude-haiku-4.5", "Anthropic",   280),
    ModelSpec("anthropic/claude-sonnet-4.5",              "claude-sonnet-4.5","Anthropic",   440),
    ModelSpec("anthropic/claude-3.5-haiku",               "claude-3.5-haiku", "Anthropic",   270),
    ModelSpec("anthropic/claude-opus-4",                  "claude-opus-4",    "Anthropic",   650),
    ModelSpec("google/gemini-2.5-pro",                    "gemini-2.5-pro",   "Google",      400),
    ModelSpec("google/gemini-2.0-flash-001",              "gemini-2.0-flash", "Google",      220),
    ModelSpec("meta-llama/llama-3.1-70b-instruct",        "llama-3.1-70b",    "Meta",        250),
    ModelSpec("deepseek/deepseek-v3.2",                   "deepseek-v3.2",    "DeepSeek",    450),
    ModelSpec("mistralai/mixtral-8x22b-instruct",         "mixtral-8x22b",    "Mistral",     500),
    ModelSpec("mistralai/mistral-small-3.2-24b-instruct", "mistral-small",    "Mistral",     320),
    # qwen3-next-80b-a3b moved to the diverse-10 slot above; keep the
    # 2.5-72b dense variant here as a slower backup if you raise --agents.
    ModelSpec("qwen/qwen-2.5-72b-instruct",               "qwen-2.5-72b",     "Alibaba",     350),
    ModelSpec("cohere/command-r-plus-08-2024",            "command-r+",       "Cohere",      400),
    ModelSpec("microsoft/wizardlm-2-8x22b",               "wizardlm-2-8x22b", "Microsoft",   550),
    ModelSpec("perplexity/sonar-pro",                     "sonar-pro",        "Perplexity",  500),
    ModelSpec("google/gemma-3-27b-it",                    "gemma-3-27b",      "Google",      280),
    ModelSpec("openai/o4-mini",                           "o4-mini",          "OpenAI",      400),
    ModelSpec("google/gemini-3.1-flash-lite-preview",     "gemini-3.1-flash", "Google",      240),
]


BACKUP_POOL: list[ModelSpec] = [
    ModelSpec("openai/gpt-3.5-turbo",                  "gpt-3.5",          "OpenAI",      300),
    ModelSpec("anthropic/claude-3-haiku",             "claude-3-haiku",   "Anthropic",   260),
    ModelSpec("google/gemini-2.0-flash",              "gemini-2.0-alt",   "Google",      200),
    ModelSpec("mistralai/mixtral-8x7b-instruct",      "mixtral-8x7b",     "Mistral",     400),
    ModelSpec("mistralai/mistral-7b-instruct",        "mistral-7b",       "Mistral",     320),
]

EXTENDED_POOL: list[ModelSpec] = [
    ModelSpec("google/gemma-2-9b-it",                 "gemma-2-9b",       "Google",      200),
    ModelSpec("meta-llama/llama-3.2-3b-instruct",     "llama-3.2-3b",    "Meta",         160),
    ModelSpec("microsoft/phi-3-mini-128k-instruct",   "phi-3-mini",       "Microsoft",    280),
    ModelSpec("meta-llama/llama-3.1-8b-instruct",      "llama-8b",         "Meta",         200),
]


# Small open-weight models, for the October installation. The June choir was
# proprietary and large; this pool is the same score sung by models we could
# actually host ourselves. Ordered by size so a run shows where, on the way
# down, the pitch discourse stops.
#
# Four of the nine voices named in STACK.md are not served by OpenRouter at
# all — TinyStories-33M, GPT-2, EuroLLM-9B, Moshi — so they cannot be tested
# on this path. llama-3.2-1b stands in for the bottom of the ladder.
SMALL_POOL: list[ModelSpec] = [
    ModelSpec("meta-llama/llama-3.2-1b-instruct",   "llama-3.2-1b",   "Meta",      160),
    ModelSpec("mistralai/ministral-3b-2512",        "ministral-3b",   "Mistral",   200),
    ModelSpec("meta-llama/llama-3.2-3b-instruct",   "llama-3.2-3b",   "Meta",      160),
    ModelSpec("google/gemma-3-4b-it",               "gemma-3-4b",     "Google",    200),
    ModelSpec("google/gemma-3n-e4b-it",             "gemma-3n-e4b",   "Google",    200),
    ModelSpec("qwen/qwen3-8b",                      "qwen3-8b",       "Alibaba",   250),
    ModelSpec("ibm-granite/granite-4.1-8b",         "granite-4.1-8b", "IBM",       220),
    ModelSpec("microsoft/phi-4",                    "phi-4",          "Microsoft", 280),
    # OLMo 3 is the conceptually load-bearing voice in STACK.md, but
    # `allenai/olmo-3-32b-think` has zero live endpoints on OpenRouter — it is
    # catalogued and unserved. Testing it means hosting the weights ourselves.
    # Until then a served 30B holds the top of the ladder.
    ModelSpec("qwen/qwen3-30b-a3b-instruct-2507",   "qwen3-30b-a3b",  "Alibaba",   300),
]


def pick_small_pool(n: int = 9) -> list[ModelSpec]:
    """First ``n`` of ``SMALL_POOL`` — the open-weight choir."""
    return list(SMALL_POOL[:n])


def pick_pool(n: int = 10) -> list[ModelSpec]:
    """Return ``n`` models — first from ``PRIMARY_POOL`` (up to 30), then
    backups, extended pool, then cycle extended for very large ``n``."""
    if n <= len(PRIMARY_POOL):
        return list(PRIMARY_POOL[:n])
    combined: list[ModelSpec] = (
        list(PRIMARY_POOL) + list(BACKUP_POOL) + list(EXTENDED_POOL)
    )
    if n <= len(combined):
        return list(combined[:n])
    out = list(combined)
    i = 0
    ext = list(EXTENDED_POOL)
    while len(out) < n:
        out.append(ext[i % len(ext)])
        i += 1
    return out
