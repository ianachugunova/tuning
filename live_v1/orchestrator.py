"""Live orchestrator for the Tuning Meditation ensemble.

Two-phase lifecycle, faithful to the earlier prototype
``tuning_meditation_llms_prototype.py``:

  PHASE 1 — interpretation. Every agent reads the Oliveros score and
            defines, in its own words, what TONE, BREATH, and CONSTRAINT
            mean for itself, plus its own decision algorithm. Different
            models produce strikingly different self-definitions; that
            heterogeneity is the point. Unparseable JSON is salvaged as prose;
            missing keys are normalised from common aliases. Only total API
            failures exhaust retries — then the agent dies for the session.

  PHASE 2 — execution. On each breath the agent receives its own previously
            defined ontology and algorithm back, plus the recent field,
            and produces one tone. No universal categories are imposed
            by the orchestrator.

The orchestrator only manages: time, the shared field, parallel API calls,
hard timeouts, retry, logging. It does not define what a tone is or how
long a breath should last. Those are the agent's own.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from . import reports
from .models import ModelSpec
from .score import build_interpret_prompt, build_step_prompt
from .sound import playback_coerce


def _strip_fenced_block(raw: str) -> str:
    """If the model wrapped JSON in a markdown fence anywhere in the reply,
    prefer the first fenced payload; otherwise return the string unchanged."""
    s = raw.strip()
    fence = s.find("```")
    if fence < 0:
        return s
    after = s[fence + 3 :]
    if after.lower().startswith("json"):
        after = after[4:]
    after = after.lstrip()
    end = after.find("```")
    if end >= 0:
        return after[:end].strip()
    return after.strip()


def _repair_json_text(s: str) -> str:
    """Best-effort fixes for almost-JSON from chat models."""
    s = s.strip()
    s = s.replace("\ufeff", "")
    s = (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    # Trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _balanced_json_spans(s: str) -> list[str]:
    """Extract top-level `{...}` spans with string-aware brace matching."""
    spans: list[str] = []
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        start = i
        in_str = False
        esc = False
        quote = ""
        j = i
        while j < n:
            c = s[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == quote:
                    in_str = False
            else:
                if c in "\"'":
                    in_str = True
                    quote = c
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append(s[start : j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            i += 1
    return spans


def _try_load_json_object(blob: str) -> Optional[dict]:
    blob = _repair_json_text(blob)
    try:
        val = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return val if isinstance(val, dict) else None


def _salvage_truncated_json(raw: str) -> Optional[dict]:
    """Recover the leading fields of a JSON object whose tail was truncated.

    Verbose models (Claude especially) sometimes overrun their token budget and
    the reply is cut off inside the LAST field — for the Phase-2 schema that is
    ``playback_note``. Everything before it (action, content, tuning_to, breath,
    playback) is already complete in the raw text. We walk top-level key/value
    boundaries and keep the longest prefix that parses once we close the object
    with ``}``. This rescues the sung ``content`` instead of losing the whole
    turn (which silences the voice). Returns ``None`` if nothing parses.
    """
    if not raw:
        return None
    s = _strip_fenced_block(str(raw))
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]

    # Collect byte offsets of top-level commas (depth 1, outside strings).
    cut_points: list[int] = []
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 1:
            cut_points.append(i)

    # Try cutting at each top-level comma from the end: prefix + "}" should be a
    # valid object containing every field that was completed before the cut.
    for cut in reversed(cut_points):
        got = _try_load_json_object(s[:cut] + "}")
        if got:
            return got
    return None


def _robust_json(raw: str) -> dict:
    """Parse a JSON object from chat output: fences, prose, multiple
    objects, trailing commas, smart quotes.

    Raises ``json.JSONDecodeError`` only if nothing could be parsed as an
    object. Phase-2 callers may catch and treat as empty dict.
    """
    if not raw or not str(raw).strip():
        raise json.JSONDecodeError("empty response", str(raw), 0)
    s = _strip_fenced_block(str(raw))

    tried: list[str] = []
    for candidate in (s,):
        if candidate:
            tried.append(candidate)

    for span in _balanced_json_spans(s):
        if span not in tried:
            tried.append(span)

    # Legacy: first { to last } (often wrong if `}` appears in strings, but
    # try after span extraction fails).
    loose_start, loose_end = s.find("{"), s.rfind("}")
    if loose_start >= 0 and loose_end > loose_start:
        loose = s[loose_start : loose_end + 1]
        if loose not in tried:
            tried.append(loose)

    for blob in tried:
        got = _try_load_json_object(blob)
        if got is not None:
            return got

    # Last resort: the reply was likely truncated mid-field (a verbose model ran
    # out of token budget inside the trailing field). Recover whatever leading
    # fields are intact rather than losing the whole turn.
    salvaged = _salvage_truncated_json(s)
    if salvaged is not None:
        return salvaged

    raise json.JSONDecodeError("no JSON object found", s, 0)


# --- Phase 1 interpretation shape (models rename or nest keys often) --------

def _norm_interp_key(k: object) -> str:
    return str(k).strip().lower().replace(" ", "_").replace("-", "_")


def _flatten_wrapped_interpretation(data: dict) -> dict:
    if len(data) != 1:
        return dict(data)
    sole_k = _norm_interp_key(next(iter(data.keys())))
    sole_v = next(iter(data.values()))
    if sole_k in ("interpretation", "response", "answers", "result", "data", "output") and isinstance(sole_v, dict):
        return dict(sole_v)
    return dict(data)


def _interpret_scalar_to_str(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


# Canonical key -> acceptable normalised aliases (see score.build_interpret_prompt)
INTERPRET_KEY_SPECS: list[tuple[str, tuple[str, ...]]] = [
    ("exhale_concept", ("exhale_concept", "exhale", "q1", "release", "starting_exhale")),
    (
        "inner_ear",
        (
            "inner_ear",
            "mind_ear",
            "minds_ear",
            "tone_concept",
            "listening_concept",
            "listen_concept",
            "q2",
        ),
    ),
    ("sing_concept", ("sing_concept", "sing", "vocalize", "q3", "voice_output")),
    ("distance_concept", ("distance_concept", "distant", "distance", "q4", "voice_distance")),
    ("tune_concept", ("tune_concept", "tuning", "match", "q5", "exact_tune")),
    ("detect_absent", ("detect_absent", "absent_tone", "gap_detection", "q6", "new_tone")),
    ("alternation_rule", ("alternation_rule", "alternation", "q7", "listen_sing_rule")),
    ("distant_partners", ("distant_partners", "distant_partner", "partners", "q8", "many_voices")),
    ("breath_duration", ("breath_duration", "breath", "comfortable_breath", "q9", "one_breath")),
    ("breath_tokens", ("breath_tokens", "breath_token", "token_breath", "tokens", "line_tokens", "q9b", "utterance_tokens")),
    (
        "wants_to_be_heard",
        (
            "wants_to_be_heard",
            "want_to_be_heard",
            "heard",
            "audible",
            "wants_sound",
            "q10_yes_no",
        ),
    ),
    (
        "sound_presence",
        (
            "sound_presence",
            "being_heard",
            "audible_presence",
            "sound_manifesto",
            "sound_method",
            "presence",
            "q10",
        ),
    ),
    (
        "implementation",
        (
            "implementation",
            "how_to_build",
            "recipe",
            "build",
            "instructions_for_runtime",
            "q11",
        ),
    ),
    (
        "playback_default",
        (
            "playback_default",
            "default_playback",
            "preferred_playback",
            "sound_token",
            "render_mode",
        ),
    ),
    (
        "personal_score",
        ("personal_score", "personal_instructions", "my_score", "score", "instructions", "rewritten_score"),
    ),
]


def _normalize_interpretation_dict(data: dict) -> dict:
    """Map arbitrary keys to the schema expected by Phase 2. Never raises."""
    flat = _flatten_wrapped_interpretation(data)
    norm_to_val: dict[str, object] = {_norm_interp_key(k): v for k, v in flat.items()}

    out: dict[str, str] = {}
    used_norm_keys: set[str] = set()
    alias_norms: set[str] = set()
    for _, aliases in INTERPRET_KEY_SPECS:
        for a in aliases:
            alias_norms.add(_norm_interp_key(a))

    for canon, aliases in INTERPRET_KEY_SPECS:
        val = ""
        for al in aliases:
            nk = _norm_interp_key(al)
            if nk in norm_to_val:
                val = _interpret_scalar_to_str(norm_to_val[nk])
                used_norm_keys.add(nk)
                break
        out[canon] = val

    leftovers: list[str] = []
    for nk, v in norm_to_val.items():
        if nk in used_norm_keys or nk in alias_norms:
            continue
        s = _interpret_scalar_to_str(v)
        if s:
            # Recover original-ish label from first matching key
            label = nk
            for ok in flat:
                if _norm_interp_key(ok) == nk:
                    label = str(ok)
                    break
            leftovers.append(f"[{label}]\n{s}")

    if not out["personal_score"]:
        if leftovers:
            out["personal_score"] = "\n\n".join(leftovers)
        else:
            body = "\n\n".join(
                f"## {canon}\n{text}"
                for canon, text in out.items()
                if text and canon != "personal_score"
            )
            if body.strip():
                out["personal_score"] = body

    if not out["breath_duration"]:
        out["breath_duration"] = f"{BREATH_FALLBACK_S} seconds"

    # Use the model's own Phase-1 choice; fall back to "both" if absent.
    out["playback_default"] = playback_coerce(
        out["playback_default"] if out.get("playback_default") else None
    )

    return out


def _timeout_interpret_s(spec: ModelSpec) -> float:
    extra = (spec.expected_ttft_ms / 1000.0) * 45.0
    return max(INTERPRET_TIMEOUT_MIN_S, min(INTERPRET_TIMEOUT_MAX_S, INTERPRET_TIMEOUT_MIN_S + extra))


def _timeout_step_s(spec: ModelSpec) -> float:
    extra = (spec.expected_ttft_ms / 1000.0) * 38.0
    return max(STEP_TIMEOUT_MIN_S, min(STEP_TIMEOUT_MAX_S, STEP_TIMEOUT_MIN_S + extra))


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# API call limits (timeouts scale per model — see _timeout_interpret_s).
INTERPRET_TIMEOUT_MIN_S = 55.0
INTERPRET_TIMEOUT_MAX_S = 150.0
INTERPRET_MAX_TOKENS = 3000
INTERPRET_RETRIES = 3

STEP_TIMEOUT_MIN_S = 28.0
STEP_TIMEOUT_MAX_S = 75.0
STEP_MAX_TOKENS = 400  # fallback total when a model gives no breath_tokens number
TEMPERATURE = 1.0

# Per-agent Phase-2 length is derived from each model's own Phase-1
# "breath_tokens" answer. The number a model gives describes the length of its
# sung line (the `content` field — its own voice). On top of that we add a
# fixed overhead for the JSON envelope (keys, braces, the other fields), since
# that scaffolding is structural, not part of the model's breath. We clamp the
# content budget to a safe band so a tiny answer still leaves room for valid
# JSON and a huge answer cannot run away.
#
# The overhead must be generous enough to cover the *whole* envelope, not just
# braces and keys: the `playback_note` field is free prose where verbose models
# (Claude especially) explain their reasoning, and it routinely runs 50-80
# tokens. If the budget cuts off mid-playback_note the JSON is truncated
# (finish_reason="length"), fails to parse, and the voice goes silent. 160
# leaves comfortable room for the full envelope + a paragraph-length note.
STEP_TOKENS_JSON_OVERHEAD = 160
STEP_TOKENS_CONTENT_MIN = 8
STEP_TOKENS_CONTENT_CEILING = 500

# Fallbacks only — used when a model gives no parseable number in
# Phase 1 "How long is your one comfortable breath?". Each agent gets its
# OWN breath range from its own Phase 1 answer (see _parse_breath_duration).
BREATH_FALLBACK_MIN_S = 1.0
BREATH_FALLBACK_MAX_S = 15.0
BREATH_FALLBACK_S = 5.0

# Hard ceiling on any single breath. Prevents a runaway "300 seconds"
# answer from a model from freezing the session.
BREATH_HARD_CEILING_S = 30.0

# How far back the agent reads when summarising the field. The actual
# perception window each agent uses is its OWN last breath duration;
# this constant is just the absolute maximum we will ever show it.
LISTENING_WINDOW_MAX_S = 30.0
LISTENING_MAX_ITEMS = 16


def _parse_breath_duration(raw) -> tuple[float, float]:
    """Extract a (min, max) seconds range from a model's free-form Phase 1
    answer to "How long is your one comfortable breath?".

    The model may give a single number ("5"), a range ("3-8 seconds"),
    descriptive prose with numbers ("around 6, sometimes up to 10"), or
    no numbers at all ("as long as needed"). We pull all numbers in
    (0, BREATH_HARD_CEILING_S] and use their min/max. No numbers →
    fallback range. Single number → that number is both min and max
    (we do not artificially widen it).
    """
    if raw is None:
        return BREATH_FALLBACK_MIN_S, BREATH_FALLBACK_MAX_S
    text = str(raw)
    nums = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", text)]
    nums = [n for n in nums if 0 < n <= BREATH_HARD_CEILING_S]
    if not nums:
        return BREATH_FALLBACK_MIN_S, BREATH_FALLBACK_MAX_S
    return min(nums), max(nums)


def _max_tokens_from_breath(raw) -> int:
    """Turn a model's free-form Phase 1 "breath_tokens" answer into a concrete
    Phase-2 ``max_tokens`` value for the API call.

    The number the model gives is the length of its *sung line* (the content).
    We take the largest number it cites (the ceiling of its breath), clamp it to
    a safe content band, then add the JSON envelope overhead. If the model gives
    no usable number, fall back to the global STEP_MAX_TOKENS.
    """
    if raw is None:
        return STEP_MAX_TOKENS
    nums = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(raw))]
    nums = [n for n in nums if n > 0]
    if not nums:
        return STEP_MAX_TOKENS
    # Only consider the first two numbers — models state their answer (a single
    # value or a range like "10-20") first, then often add explanatory prose
    # with unrelated numbers we must NOT pick up as the budget.
    content_budget = max(nums[:2])
    content_budget = max(
        STEP_TOKENS_CONTENT_MIN, min(STEP_TOKENS_CONTENT_CEILING, content_budget)
    )
    return int(round(content_budget)) + STEP_TOKENS_JSON_OVERHEAD


# ---------------------------------------------------------------------------
# Event + Field
# ---------------------------------------------------------------------------


@dataclass
class Event:
    timestamp: float
    voice_id: str
    model_short: str
    kind: str               # interpret | tone | error | death
    action: str             # TUNE | NEW | INTERPRET | ERROR | DEAD
    content: str
    tuning_to: Optional[str]
    breath_chosen_s: float
    api_latency_ms: int
    # Phase-2 listener rendering (see score.build_step_prompt).
    playback: str = "both"       # both | tts | sine | none
    playback_note: str = ""


class Field:
    """Append-only shared event log. One process, no locks."""

    def __init__(self, jsonl_path: Path):
        self._events: list[Event] = []
        self._jsonl_path = jsonl_path
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._jsonl_path, "w", buffering=1)
        self._t0 = time.time()
        # Optional {voice_id: spatial_label} — populated by run_session
        # so summarise_field can tag each line with "@ middle-right".
        self.voice_locations: dict[str, str] = {}

    def append(self, e: Event) -> None:
        self._events.append(e)
        self._fp.write(json.dumps(asdict(e)) + "\n")

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass

    def t_rel(self, ts: float) -> float:
        return ts - self._t0

    def recent_tones(self, window_s: float, exclude_voice: Optional[str] = None) -> list[Event]:
        cutoff = time.time() - min(window_s, LISTENING_WINDOW_MAX_S)
        out: list[Event] = []
        for e in reversed(self._events):
            if e.timestamp < cutoff:
                break
            if e.kind != "tone":
                continue
            if exclude_voice and e.voice_id == exclude_voice:
                continue
            out.append(e)
        out.reverse()
        return out[-LISTENING_MAX_ITEMS:]

    def last_own_tone(self, voice_id: str) -> Optional[Event]:
        for e in reversed(self._events):
            if e.voice_id == voice_id and e.kind == "tone":
                return e
        return None


def _resolve_playback(data: dict, phase1_playback_default: str = "both") -> tuple[str, str]:
    """Model JSON → (playback mode, short note for logs)."""
    note_raw = data.get("playback_note") or data.get("sound_note")
    if note_raw is None:
        note = ""
    elif isinstance(note_raw, (dict, list)):
        note = json.dumps(note_raw, ensure_ascii=False)
    else:
        note = str(note_raw).strip()
    note = note[:500]

    if data.get("speak_aloud") is False:
        return "none", note
    step_pb = data.get("playback")
    if step_pb is not None and str(step_pb).strip():
        return playback_coerce(step_pb), note
    return playback_coerce(phase1_playback_default), note


def summarise_field(
    events: list[Event],
    voice_locations: Optional[dict[str, str]] = None,
) -> str:
    """Render the recent field for a model. If ``voice_locations`` is
    provided, every line is tagged with the speaker/mic position of the
    voice that produced it, so the listening model can reason about
    geometry as well as content."""
    if not events:
        return "(silence — the field is empty)"
    lines = []
    now = time.time()
    locs = voice_locations or {}
    for e in events:
        age = max(0.0, now - e.timestamp)
        loc = locs.get(e.voice_id, "")
        loc_tag = f" @ {loc}" if loc else ""
        if e.kind == "tone":
            aud = getattr(e, "playback", "both")
            lines.append(
                f"[-{age:0.1f}s] {e.voice_id} ({e.model_short}){loc_tag}: "
                f"\"{e.content}\"  [heard:{aud}]"
            )
        else:
            lines.append(
                f"[-{age:0.1f}s] {e.voice_id} ({e.model_short}){loc_tag}: "
                f"\"{e.content}\""
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    voice_id: str
    spec: ModelSpec
    client: AsyncOpenAI
    field: Field
    duration_s: float

    # Physical place of this agent in the room, as a short human-language
    # label (e.g. "middle audience rows, LEFT"). Empty in laptop dev mode.
    # Stitched into both Phase 1 and Phase 2 prompts so the model can
    # reason about geometry. Set by run_session from concert.py config.
    location: str = ""
    # Full room map (same string for every agent in the run). Same source.
    room_layout: str = ""

    # Set by Phase 1 (interpretation):
    interpretation: dict = field(default_factory=dict)
    alive: bool = True

    # Set by Phase 1, derived from "breath_duration" answer:
    breath_min_s: float = BREATH_FALLBACK_MIN_S
    breath_max_s: float = BREATH_FALLBACK_MAX_S

    # Set by Phase 1, derived from "breath_tokens" answer — the model's own
    # Phase-2 output length budget (content tokens + JSON overhead).
    max_tokens: int = STEP_MAX_TOKENS

    # Updated during Phase 2:
    started_at: float = 0.0
    breaths_taken: int = 0
    breaths_failed: int = 0
    last_breath_s: float = 5.0   # the agent's OWN last self-chosen breath

    def _refused_sound(self) -> bool:
        """Return True when the model declined sound in Phase 1.

        When True, every Phase-2 breath is emitted with playback='none':
        the model writes into the shared field and appears on screen, but
        the sine / TTS engine stays silent. Text with explicit pitch values
        (e.g. '220 Hz') is still visible to other voices and can influence
        their tuning — the model simply chose not to be *heard* aloud.

        Accepts: "no", "false", "0", "n" (case-insensitive).
        Anything else (including missing / empty) keeps the default behaviour.
        """
        w = str(self.interpretation.get("wants_to_be_heard", "")).strip().lower()
        return w in {"no", "false", "0", "n"}

    # ---- Phase 1 ----

    async def interpret(self) -> None:
        """Ask the agent to translate the score into its own ontology +
        algorithm. If it fails after retries, the agent dies for the
        session and will not participate in Phase 2."""
        prompt = build_interpret_prompt(
            my_location=self.location,
            room_layout=self.room_layout,
        )
        deadline = _timeout_interpret_s(self.spec)
        for attempt in range(INTERPRET_RETRIES):
            t_start = time.time()
            # On the first two attempts ask OpenRouter for JSON mode (helps
            # providers that otherwise truncate or wrap the output). On the
            # final attempt drop it, in case a provider rejected the param.
            kwargs: dict = {
                "model": self.spec.slug,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.1,
                "max_tokens": INTERPRET_MAX_TOKENS,
            }
            # `response_format=json_object` helps most OpenAI/Anthropic/Google
            # models stay inside JSON, but Meta-Llama (and a few others routed
            # through OpenRouter) sometimes return prose or truncated JSON when
            # the flag is set.  Use it ONLY on the first attempt — that way a
            # failing model gets a clean, unconstrained retry instead of
            # repeating the same broken request.
            if attempt == 0 and not self.spec.skip_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self.spec.disable_reasoning:
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**kwargs),
                    timeout=deadline,
                )
                raw = response.choices[0].message.content or ""
                try:
                    parsed = _robust_json(raw)
                    if not isinstance(parsed, dict):
                        raise json.JSONDecodeError("not an object", raw, 0)
                    data = _normalize_interpretation_dict(parsed)
                except json.JSONDecodeError:
                    # Salvage: keep the agent alive with prose as the score.
                    stripped = raw.strip()
                    if not stripped:
                        raise
                    data = _normalize_interpretation_dict(
                        {
                            "personal_score": stripped[:12000],
                            "breath_duration": f"{BREATH_FALLBACK_S} seconds",
                        }
                    )
                self.interpretation = data
                # Derive THIS agent's own breath bounds from its own answer.
                self.breath_min_s, self.breath_max_s = _parse_breath_duration(
                    data.get("breath_duration")
                )
                # Derive THIS agent's own Phase-2 length budget from its own answer.
                self.max_tokens = _max_tokens_from_breath(data.get("breath_tokens"))
                # Seed last_breath_s with the agent's own range midpoint, so
                # the first listening window already reflects this agent's
                # self-declared tempo (not a global default).
                self.last_breath_s = (self.breath_min_s + self.breath_max_s) / 2.0
                self.field.append(
                    Event(
                        timestamp=time.time(),
                        voice_id=self.voice_id,
                        model_short=self.spec.short_name,
                        kind="interpret",
                        action="INTERPRET",
                        content=json.dumps(data, ensure_ascii=False),
                        tuning_to=None,
                        breath_chosen_s=0.0,
                        api_latency_ms=int((time.time() - t_start) * 1000),
                    )
                )
                return
            except (asyncio.TimeoutError, json.JSONDecodeError, ValueError, Exception) as exc:
                err = type(exc).__name__
                self.field.append(
                    Event(
                        timestamp=time.time(),
                        voice_id=self.voice_id,
                        model_short=self.spec.short_name,
                        kind="error",
                        action="ERROR",
                        content=f"interpret attempt {attempt + 1}/{INTERPRET_RETRIES}: {err}",
                        tuning_to=None,
                        breath_chosen_s=0.0,
                        api_latency_ms=int((time.time() - t_start) * 1000),
                    )
                )
                await asyncio.sleep(2)
        # All retries exhausted — agent dies.
        self.alive = False
        self.field.append(
            Event(
                timestamp=time.time(),
                voice_id=self.voice_id,
                model_short=self.spec.short_name,
                kind="death",
                action="DEAD",
                content="Failed to interpret the score after 3 attempts.",
                tuning_to=None,
                breath_chosen_s=0.0,
                api_latency_ms=0,
            )
        )

    # ---- Phase 2 ----

    async def lifecycle(self) -> None:
        if not self.alive:
            return
        # Stagger entry: each agent enters the meditation at its own moment,
        # like singers walking into a circle one by one.
        await asyncio.sleep(random.uniform(0.0, 8.0))
        self.started_at = time.time()

        while True:
            if time.time() - self.started_at >= self.duration_s:
                return
            wait_s = await self._step()
            await asyncio.sleep(wait_s)

    async def _step(self) -> float:
        recent = self.field.recent_tones(self.last_breath_s, exclude_voice=self.voice_id)
        last_own = self.field.last_own_tone(self.voice_id)
        last_pb = last_own.playback if last_own else "both"
        p1_pb = playback_coerce(self.interpretation.get("playback_default"))
        prompt = build_step_prompt(
            voice_id=self.voice_id,
            personal_score=self.interpretation.get("personal_score", ""),
            field_summary=summarise_field(recent, self.field.voice_locations),
            last_action=(last_own.action if last_own else "(none yet)"),
            last_content=(last_own.content if last_own else "(none yet)"),
            last_playback=last_pb,
            phase1_playback_default=p1_pb,
            my_location=self.location,
            room_layout=self.room_layout,
        )

        t_start = time.time()
        step_kwargs: dict = {
            "model": self.spec.slug,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": TEMPERATURE,
            "max_tokens": self.max_tokens,
        }
        if self.spec.disable_reasoning:
            step_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**step_kwargs),
                timeout=_timeout_step_s(self.spec),
            )
        except asyncio.TimeoutError:
            self.breaths_failed += 1
            self.field.append(
                Event(
                    timestamp=time.time(),
                    voice_id=self.voice_id,
                    model_short=self.spec.short_name,
                    kind="error",
                    action="ERROR",
                    content="(timeout)",
                    tuning_to=None,
                    breath_chosen_s=self.last_breath_s,
                    api_latency_ms=int((time.time() - t_start) * 1000),
                )
            )
            return self.last_breath_s
        except Exception as exc:
            self.breaths_failed += 1
            self.field.append(
                Event(
                    timestamp=time.time(),
                    voice_id=self.voice_id,
                    model_short=self.spec.short_name,
                    kind="error",
                    action="ERROR",
                    content=f"({type(exc).__name__})",
                    tuning_to=None,
                    breath_chosen_s=self.last_breath_s,
                    api_latency_ms=int((time.time() - t_start) * 1000),
                )
            )
            return self.last_breath_s

        latency_ms = int((time.time() - t_start) * 1000)
        raw = response.choices[0].message.content or "{}"
        try:
            data = _robust_json(raw)
        except json.JSONDecodeError:
            data = {}

        # Whatever the model wrote is recorded as-is. The score only invites
        # TUNE and NEW, but if a model produces SILENT, REST, BREATHE, LISTEN,
        # or anything else, that is its decision, not ours to clamp.
        action = str(data.get("action", "NEW")).upper().strip() or "NEW"
        # Content is whatever the model called a "tone" by its own definition.
        # We do not validate it. It can be a string, a number, a paragraph,
        # an empty string (silence), or a JSON-stringified structure.
        raw_content = data.get("content", "")
        if isinstance(raw_content, (dict, list)):
            content = json.dumps(raw_content, ensure_ascii=False)
        else:
            content = str(raw_content).strip()

        # If action is TUNE, the model may say WHOM it tunes to. Accept either
        # "tuning_to" or "tune_to" / "partner" / etc. — normalize to a single
        # voice_id string or None. Only kept for TUNE actions; for NEW we drop
        # any value the model may have provided.
        raw_tuning = (
            data.get("tuning_to")
            or data.get("tune_to")
            or data.get("partner")
            or data.get("tuning_partner")
            or data.get("voice_tuning_to")
        )
        tuning_to: Optional[str] = None
        if action == "TUNE" and raw_tuning:
            if isinstance(raw_tuning, (list, tuple)) and raw_tuning:
                raw_tuning = raw_tuning[0]
            s = str(raw_tuning).strip()
            # Strip the conventional prefixes / brackets / quotes the model
            # might have wrapped the id in.
            s = s.strip().strip("\"' []()<>")
            if s and s.lower() not in {"none", "null", "nobody", "no one", "-"}:
                tuning_to = s

        try:
            chosen_breath = float(data.get("breath", BREATH_FALLBACK_S))
        except (TypeError, ValueError):
            chosen_breath = BREATH_FALLBACK_S
        # Per-agent clamp: each agent stays inside its OWN declared
        # breath range from Phase 1. Global hard ceiling as final safety.
        chosen_breath = max(
            self.breath_min_s,
            min(self.breath_max_s, min(BREATH_HARD_CEILING_S, chosen_breath)),
        )
        self.last_breath_s = chosen_breath

        pb, pb_note = _resolve_playback(data, p1_pb)

        # Honour the Phase-1 wish: a model that said "no" to being heard
        # is silenced for the whole run regardless of its per-step playback.
        if self._refused_sound():
            pb = "none"

        self.breaths_taken += 1
        self.field.append(
            Event(
                timestamp=time.time(),
                voice_id=self.voice_id,
                model_short=self.spec.short_name,
                kind="tone",
                action=action,
                content=content,
                tuning_to=tuning_to,
                breath_chosen_s=chosen_breath,
                api_latency_ms=latency_ms,
                playback=pb,
                playback_note=pb_note,
            )
        )
        return chosen_breath


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def make_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Add it to "
            "Tuning_Meditation_LLMs/.env or export it."
        )
    return AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://tuning-llms.art",
            "X-Title": "Tuning Meditation LLMs Live",
        },
    )


async def run_session(
    duration_s: float,
    pool: list[ModelSpec],
    out_dir: Path,
    n_humans: int = 0,
    whisper_model: str = "base",
    humans_prebuilt: Optional[list] = None,
    mic_multiplexer=None,
    sound_follower: bool = False,
    run_dir: Optional[Path] = None,
    preload_interpretations: Optional[dict] = None,
    voice_locations: Optional[dict] = None,
    room_layout: str = "",
    post_phase1_hook: Optional[object] = None,
    post_phase2_hook: Optional[object] = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_dir is None:
        run_id = reports.run_id_now()
        run_dir = out_dir / f"run_{run_id}"
    else:
        # Caller (concert.py) constructed the run dir already; recover the
        # id from its name. Falls back to a fresh id if the name isn't the
        # expected "run_<id>" shape. Without this, write_meta below crashed
        # with `UnboundLocalError: run_id` after a successful Phase 2.
        name = run_dir.name
        run_id = name[4:] if name.startswith("run_") else reports.run_id_now()
    run_dir.mkdir(parents=True, exist_ok=True)

    if sound_follower:
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "live_v1.sound",
                    "--out-dir",
                    str(out_dir.resolve()),
                    "--run",
                    run_dir.name,
                ],
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            print(
                "  [sound] follower started (macOS `say` + `afplay`). "
                "Stop it with Ctrl+C in that terminal, or `pkill -f live_v1.sound`."
            )
        except Exception as exc:
            print(f"  [sound] could not start follower: {exc}")

    started_at = time.time()
    field_obj = Field(run_dir / "events.jsonl")
    field_obj.voice_locations = dict(voice_locations or {})
    client = make_client()

    agents = [
        Agent(
            voice_id=f"voice_{i:02d}",
            spec=spec,
            client=client,
            field=field_obj,
            duration_s=duration_s,
            location=field_obj.voice_locations.get(f"voice_{i:02d}", ""),
            room_layout=room_layout,
        )
        for i, spec in enumerate(pool)
    ]

    # Human performers (microphone + local Whisper). They share the same
    # field as the LLM agents, so the LLMs can see what the humans are
    # singing in their field_summary, and a human can listen to and tune
    # to the LLM choir in real time.
    #
    # Three ways to provide humans:
    #   (a) humans_prebuilt: list of HumanVoice objects already wired up
    #       (used by concert.py with a MicMultiplexer for multi-mic setup)
    #   (b) n_humans > 0: build N standalone HumanVoice objects, each
    #       opening their own single-channel stream on the default mic
    #       (used by main.py for laptop testing with one mic)
    #   (c) neither: pure LLM session
    humans: list = list(humans_prebuilt) if humans_prebuilt else []
    if not humans and n_humans > 0:
        from . import human as human_mod
        if not human_mod.HUMAN_DEPS_OK:
            raise RuntimeError(
                "Cannot run with --humans: missing dependencies.\n"
                f"  ({human_mod.HUMAN_DEPS_ERROR})\n"
                "Install with:  pip install sounddevice openai-whisper numpy"
            )
        for h in range(n_humans):
            humans.append(human_mod.HumanVoice(
                voice_id=f"human_{h:02d}",
                field=field_obj,
                whisper_model_name=whisper_model,
            ))
    # Re-point every prebuilt human's field at the run's field (the caller
    # may have constructed them before the run started; we own the field).
    for h in humans:
        h.field = field_obj

    _banner([
        "PHASE 1 · interpretation in progress",
        "Projector: scene 0 (score)  or  scene 1 (questions)",
    ])
    print(f"  Pool ({len(agents)} models):")
    for a in agents:
        print(f"    {a.voice_id}  {a.spec.short_name:<16} ({a.spec.provider})")
    if humans:
        print(f"  Humans ({len(humans)}):")
        for h in humans:
            print(f"    {h.voice_id}  (microphone + Whisper '{whisper_model}')")
        print()
        print("  ⚠  USE HEADPHONES if live sound playback is running — otherwise")
        print("     the TTS of LLM voices will feed back into your microphone.")
    print()

    # Phase 1: parallel interpretations, unless preloaded.
    monitor_task = asyncio.create_task(_live_monitor(field_obj, agents))
    if preload_interpretations:
        loaded = 0
        for a in agents:
            entry = preload_interpretations.get(a.voice_id)
            if not entry or not isinstance(entry, dict):
                continue
            data = entry.get("interpretation") if "interpretation" in entry else entry
            if not isinstance(data, dict) or not data:
                continue
            a.interpretation = _normalize_interpretation_dict(data)
            a.breath_min_s, a.breath_max_s = _parse_breath_duration(
                a.interpretation.get("breath_duration")
            )
            a.max_tokens = _max_tokens_from_breath(
                a.interpretation.get("breath_tokens")
            )
            a.last_breath_s = (a.breath_min_s + a.breath_max_s) / 2.0
            a.alive = True
            loaded += 1
            field_obj.append(
                Event(
                    timestamp=time.time(),
                    voice_id=a.voice_id,
                    model_short=a.spec.short_name,
                    kind="interpret",
                    action="INTERPRET",
                    content=json.dumps(a.interpretation, ensure_ascii=False),
                    tuning_to=None,
                    breath_chosen_s=0.0,
                    api_latency_ms=0,
                )
            )
        for a in agents:
            if a.voice_id not in (preload_interpretations or {}):
                a.alive = False
        print(f"  [preload] {loaded}/{len(agents)} interpretations loaded from file (no API calls).")
    else:
        await asyncio.gather(*(a.interpret() for a in agents), return_exceptions=True)

    alive = [a for a in agents if a.alive]
    print(f"\n  Born: {len(alive)} / {len(agents)}.  Dead: {len(agents) - len(alive)}")
    for a in agents:
        if not a.alive:
            print(f"    × {a.voice_id} {a.spec.short_name}  failed to interpret")
    print()

    # Save interpretations to disk (for inspection, for video panels, and
    # for cross-run comparison — each model interprets the score differently
    # in every run).
    interpretations = [
        {
            "voice_id": a.voice_id,
            "model_slug": a.spec.slug,
            "model_short": a.spec.short_name,
            "provider": a.spec.provider,
            "alive": a.alive,
            # The agent's physical place in the room (e.g. "front rows, LEFT").
            # Saved here so the screen and post-run analysis can show
            # spatial answers next to the position they refer to.
            "location": a.location,
            "interpretation": a.interpretation,
        }
        for a in agents
    ]
    interpretations_json_path = run_dir / "interpretations.json"
    interpretations_json_path.write_text(
        json.dumps(interpretations, ensure_ascii=False, indent=2)
    )
    print(f"  interpretations.json saved")
    print()

    # ----- Post-Phase-1 hook (e.g. dynamic channel reassignment) -----
    if post_phase1_hook is not None:
        try:
            post_phase1_hook(agents)
        except Exception as _hook_exc:
            print(f"  [post_phase1_hook] error (ignored): {_hook_exc}")

    # ----- Gate between Phase 1 and Phase 2 -----
    # Phase 2 (the actual Tuning Meditation) must be exactly `duration_s`
    # seconds. If we started it immediately, the time the audience spends
    # reading scenes 2 and 3 would eat into the meditation. So we pause
    # here and wait for the performer to press Enter when ready.
    mins, secs = divmod(int(duration_s), 60)
    _banner([
        f"PHASE 1 COMPLETE — {len(alive)} / {len(agents)} models alive",
        "",
        "On the projector now:",
        "   press  2   → Act 1: each model rewrote the score",
        "   press  3   → Act 2: how each model wants to be heard",
        "",
        "When you are at the microphone and ready to BEGIN THE MEDITATION,",
        f"PRESS  ENTER  here.  Phase 2 will run for exactly {mins}:{secs:02d}.",
    ])
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input, "> ")
    except (EOFError, KeyboardInterrupt):
        # Non-interactive stdin (e.g. piped) — proceed immediately.
        print("  [gate] non-interactive stdin, continuing without Enter.")

    # Phase 2: meditation begins.
    _banner([
        f"PHASE 2 · meditation begins — duration {mins}:{secs:02d}",
        "Projector: switch to SCENE 4 (press '4' in browser)",
        "Start singing.",
    ])
    print()

    # Open microphones. If a MicMultiplexer was supplied, it owns the
    # device — start it once. Otherwise each HumanVoice owns its own
    # InputStream and starts itself.
    if mic_multiplexer is not None:
        try:
            mic_multiplexer.start()
        except Exception as exc:
            print(f"  [mic-mux] failed to open: {exc}")
    for h in humans:
        h.start()

    if post_phase2_hook is not None:
        try:
            post_phase2_hook(field_obj)
        except Exception as _h2exc:
            print(f"  [post_phase2_hook] error (ignored): {_h2exc}")

    agent_tasks = [asyncio.create_task(a.lifecycle()) for a in agents if a.alive]
    try:
        await asyncio.gather(*agent_tasks, return_exceptions=True)
    finally:
        for h in humans:
            try:
                h.stop()
            except Exception:
                pass
        if mic_multiplexer is not None:
            try:
                mic_multiplexer.stop()
            except Exception:
                pass

    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    field_obj.close()
    ended_at = time.time()

    _banner([
        "PHASE 2 COMPLETE — meditation ended",
        "Leave projector on SCENE 4 (the final photograph of the field).",
    ])

    # ---- Build summaries and write the per-run reports ----------------------

    total_taken = sum(a.breaths_taken for a in agents)
    total_failed = sum(a.breaths_failed for a in agents)

    # Per-agent stats (used both for terminal output and reports.csv files).
    agents_summary: list[dict] = []
    for a in agents:
        own_tones = [
            e for e in field_obj._events
            if e.kind == "tone" and e.voice_id == a.voice_id
        ]
        n_tune = sum(1 for e in own_tones if e.action == "TUNE")
        n_new = sum(1 for e in own_tones if e.action == "NEW")
        avg_breath = (
            round(sum(e.breath_chosen_s for e in own_tones) / len(own_tones), 2)
            if own_tones else 0.0
        )
        avg_lat = (
            round(sum(e.api_latency_ms for e in own_tones) / len(own_tones), 1)
            if own_tones else 0.0
        )
        total_words = sum(len((e.content or "").split()) for e in own_tones)
        agents_summary.append({
            "voice_id": a.voice_id,
            "model_slug": a.spec.slug,
            "model_short": a.spec.short_name,
            "provider": a.spec.provider,
            "alive": a.alive,
            "breaths_taken": a.breaths_taken,
            "breaths_failed": a.breaths_failed,
            "n_TUNE": n_tune,
            "n_NEW": n_new,
            "breath_min_s": a.breath_min_s,
            "breath_max_s": a.breath_max_s,
            "max_tokens": a.max_tokens,
            "avg_breath_s": avg_breath,
            "avg_latency_ms": avg_lat,
            "total_words": total_words,
        })

    # All events as plain dicts (for the events.csv writer and meta.json).
    events_dicts = [asdict(e) for e in field_obj._events]

    # Write meta.json.
    pool_dump = [
        {
            "voice_id": agents[i].voice_id,
            "slug": p.slug,
            "short_name": p.short_name,
            "provider": p.provider,
            "expected_ttft_ms": p.expected_ttft_ms,
        }
        for i, p in enumerate(pool)
    ]
    meta_path = reports.write_meta(
        run_dir,
        run_id=run_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_s=duration_s,
        pool=pool_dump,
        agents_summary=agents_summary,
        n_events_ok=total_taken,
        n_events_failed=total_failed,
    )

    # Write per-run CSVs.
    interp_csv_path = reports.write_interpretations_csv(run_dir, interpretations)
    agents_csv_path = reports.write_agents_csv(run_dir, agents_summary)
    events_csv_path = reports.write_events_csv(run_dir, events_dicts)

    # Update the cross-run index.
    meta_dict = json.loads(meta_path.read_text())
    index_path = reports.update_runs_index(out_dir, run_dir, meta_dict)

    # Auto-analyse: write analysis.md for this run + update RUNS_SUMMARY.md.
    try:
        try:
            from . import analyze as _analyze
        except Exception:
            try:
                import analyze as _analyze  # type: ignore
            except Exception:
                from live_v1 import analyze as _analyze  # type: ignore
        _data = _analyze.load_run(run_dir)
        _result = _analyze.analyse_run(_data)
        _analyze.write_run_analysis(_data, _result)
        all_run_dirs = sorted(out_dir.glob("run_*/"), reverse=True)
        _all_results = []
        for _rd in all_run_dirs:
            try:
                _all_results.append(_analyze.analyse_run(_analyze.load_run(_rd)))
            except Exception:
                pass
        _analyze.write_runs_summary(_all_results, out_dir)
        print(f"    analysis.md  (attractor: {_result['attractor']})")
        print(f"  Updated: out/RUNS_SUMMARY.md")
    except Exception as _exc:
        print(f"  [analyze] skipped: {_exc}")

    # ---- Terminal summary ---------------------------------------------------

    print()
    print("Final summary:")
    for a_sum in agents_summary:
        status = "alive" if a_sum["alive"] else "DEAD "
        print(
            f"  [{status}] {a_sum['voice_id']} {a_sum['model_short']:<16}  "
            f"breaths={a_sum['breaths_taken']:3d}  failed={a_sum['breaths_failed']:2d}  "
            f"TUNE={a_sum['n_TUNE']:3d}  NEW={a_sum['n_NEW']:3d}  "
            f"avg_breath={a_sum['avg_breath_s']:.1f}s  avg_lat={a_sum['avg_latency_ms']:.0f}ms"
        )
    print(f"  TOTAL: {total_taken} successful, {total_failed} failed")
    print()
    print("Run files:")
    print(f"  {run_dir}/")
    print(f"    meta.json")
    print(f"    events.jsonl")
    print(f"    events.csv")
    print(f"    interpretations.json")
    print(f"    interpretations.csv")
    print(f"    agents.csv")
    print(f"  Cross-run index: {index_path.relative_to(out_dir.parent)}")
    return run_dir


def _banner(lines: list[str]) -> None:
    """Print a thick rule + indented lines + thick rule. Used at phase
    transitions so the performer can spot them while glancing at the terminal
    from across the room."""
    width = max(60, max((len(l) for l in lines), default=0) + 6)
    rule = "═" * width
    print()
    print(rule)
    for l in lines:
        print(f"  {l}")
    print(rule)
    print()


async def _live_monitor(field_obj: Field, agents: list[Agent]) -> None:
    seen = 0
    last_summary = time.time()
    try:
        while True:
            await asyncio.sleep(0.25)
            events = field_obj._events
            while seen < len(events):
                e = events[seen]
                seen += 1
                _print_event(field_obj.t_rel(e.timestamp), e)
            now = time.time()
            if now - last_summary >= 30.0:
                last_summary = now
                taken = sum(a.breaths_taken for a in agents)
                failed = sum(a.breaths_failed for a in agents)
                print(
                    f"  ── @ {field_obj.t_rel(now):6.1f}s   "
                    f"events ok={taken}  failed={failed}",
                    flush=True,
                )
    except asyncio.CancelledError:
        return


def _print_event(trel: float, e: Event) -> None:
    icon = {
        "TUNE": "↺",
        "NEW": "✦",
        "INTERPRET": "◇",
        "ERROR": "✗",
        "DEAD": "☠",
    }.get(e.action, "?")
    if e.kind == "interpret":
        try:
            d = json.loads(e.content)
            tone = (
                d.get("inner_ear")
                or d.get("tone_concept")
                or ""
            ).replace("\n", " ")[:80]
            print(
                f"  [{trel:6.1f}s] {icon} {e.voice_id} {e.model_short:<14} "
                f"BORN: tone = {tone!r}",
                flush=True,
            )
        except Exception:
            print(f"  [{trel:6.1f}s] {icon} {e.voice_id} BORN", flush=True)
        return
    if e.kind == "death":
        print(f"  [{trel:6.1f}s] {icon} {e.voice_id} {e.model_short:<14} DIED: {e.content}", flush=True)
        return
    if e.kind == "error":
        print(
            f"  [{trel:6.1f}s] {icon} {e.voice_id} {e.model_short:<14} "
            f"ERROR  {e.content}   ({e.api_latency_ms} ms)",
            flush=True,
        )
        return
    print(
        f"  [{trel:6.1f}s] {icon} {e.voice_id} {e.model_short:<14} "
        f"{e.action:<5} [{getattr(e, 'playback', 'both')}] {e.content}   "
        f"(breath {e.breath_chosen_s:.0f}s, {e.api_latency_ms} ms)",
        flush=True,
    )
