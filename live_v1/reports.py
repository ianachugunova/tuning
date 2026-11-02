"""Per-run reports + cross-run index.

Every live session writes a self-contained folder, e.g.

    out/run_2026-05-06_12-34-56/
        meta.json              — full run config + summary stats
        events.jsonl           — every event in raw form (one JSON per line)
        interpretations.json   — each agent's self-definition (Phase 1 output)
        interpretations.csv    — same, tabular, easy to open in a spreadsheet
        agents.csv             — per-agent summary stats (breaths, latencies, ...)
        events.csv             — every tone event in tabular form

Across all runs we also maintain:

    out/runs_index.csv         — one row per run with overall stats

This module is pure side-effect at the end of a session. No async, no LLM
calls. It only formats what already happened.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_id_now() -> str:
    """Filesystem-safe ISO-like timestamp for naming this run's folder."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


def _stringify(value: Any) -> str:
    """Coerce arbitrary JSON value into a single CSV-safe string. Lists or
    dicts (e.g. mistral-large returning an array of step objects) are
    JSON-encoded so the structure is preserved in the cell."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-run report
# ---------------------------------------------------------------------------


def write_meta(
    run_dir: Path,
    *,
    run_id: str,
    started_at: float,
    ended_at: float,
    duration_s: float,
    pool: list[dict],
    agents_summary: list[dict],
    n_events_ok: int,
    n_events_failed: int,
) -> Path:
    meta = {
        "run_id": run_id,
        "started_at_unix": started_at,
        "started_at_iso": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "ended_at_unix": ended_at,
        "ended_at_iso": datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
        "wallclock_duration_s": round(ended_at - started_at, 2),
        "configured_duration_s": duration_s,
        "n_agents_requested": len(pool),
        "n_alive_after_phase1": sum(1 for a in agents_summary if a["alive"]),
        "n_dead_after_phase1": sum(1 for a in agents_summary if not a["alive"]),
        "n_events_ok": n_events_ok,
        "n_events_failed": n_events_failed,
        "pool": pool,
        "agents": agents_summary,
    }
    path = run_dir / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return path


def write_interpretations_csv(run_dir: Path, interpretations: list[dict]) -> Path:
    path = run_dir / "interpretations.csv"
    fields = [
        "voice_id",
        "model_slug",
        "model_short",
        "provider",
        "alive",
        "exhale_concept",
        "inner_ear",
        "sing_concept",
        "distance_concept",
        "tune_concept",
        "detect_absent",
        "alternation_rule",
        "distant_partners",
        "breath_duration",
        "sound_presence",
        "implementation",
        "personal_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for entry in interpretations:
            interp = entry.get("interpretation") or {}
            w.writerow({
                "voice_id": entry["voice_id"],
                "model_slug": entry["model_slug"],
                "model_short": entry["model_short"],
                "provider": entry["provider"],
                "alive": entry["alive"],
                "exhale_concept":    _stringify(interp.get("exhale_concept")),
                "inner_ear":         _stringify(interp.get("inner_ear")),
                "sing_concept":      _stringify(interp.get("sing_concept")),
                "distance_concept":  _stringify(interp.get("distance_concept")),
                "tune_concept":      _stringify(interp.get("tune_concept")),
                "detect_absent":     _stringify(interp.get("detect_absent")),
                "alternation_rule":  _stringify(interp.get("alternation_rule")),
                "distant_partners":  _stringify(interp.get("distant_partners")),
                "breath_duration":   _stringify(interp.get("breath_duration")),
                "sound_presence":    _stringify(interp.get("sound_presence")),
                "implementation":    _stringify(interp.get("implementation")),
                "personal_score":    _stringify(interp.get("personal_score")),
            })
    return path


def write_agents_csv(run_dir: Path, agents_summary: list[dict]) -> Path:
    path = run_dir / "agents.csv"
    fields = [
        "voice_id",
        "model_slug",
        "model_short",
        "provider",
        "alive",
        "breaths_taken",
        "breaths_failed",
        "n_TUNE",
        "n_NEW",
        "breath_min_s",
        "breath_max_s",
        "avg_breath_s",
        "avg_latency_ms",
        "total_words",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for a in agents_summary:
            w.writerow({k: a.get(k, "") for k in fields})
    return path


def write_events_csv(run_dir: Path, events: list) -> Path:
    """events: list[Event-like dict]. Tones only (we skip interpret/error/death
    in this table — the JSONL has the full record)."""
    path = run_dir / "events.csv"
    fields = [
        "t_rel_s",
        "timestamp_unix",
        "voice_id",
        "model_short",
        "action",
        "playback",
        "playback_note",
        "content",
        "breath_chosen_s",
        "api_latency_ms",
    ]
    if not events:
        with open(path, "w", newline="", encoding="utf-8") as fp:
            csv.writer(fp).writerow(fields)
        return path
    t0 = min(e["timestamp"] for e in events)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for e in events:
            if e.get("kind") != "tone":
                continue
            w.writerow({
                "t_rel_s": round(e["timestamp"] - t0, 2),
                "timestamp_unix": e["timestamp"],
                "voice_id": e["voice_id"],
                "model_short": e["model_short"],
                "action": e["action"],
                "playback": e.get("playback", "both"),
                "playback_note": e.get("playback_note", ""),
                "content": e["content"],
                "breath_chosen_s": e["breath_chosen_s"],
                "api_latency_ms": e["api_latency_ms"],
            })
    return path


# ---------------------------------------------------------------------------
# Cross-run index
# ---------------------------------------------------------------------------


INDEX_FIELDS = [
    "run_id",
    "started_at_iso",
    "configured_duration_s",
    "wallclock_duration_s",
    "n_agents_requested",
    "n_alive",
    "n_dead",
    "n_events_ok",
    "n_events_failed",
    "models_used",
    "run_dir",
]


def update_runs_index(out_dir: Path, run_dir: Path, meta: dict) -> Path:
    """Append (or create) the cross-run index CSV under ``out_dir``."""
    path = out_dir / "runs_index.csv"
    new_row = {
        "run_id": meta["run_id"],
        "started_at_iso": meta["started_at_iso"],
        "configured_duration_s": meta["configured_duration_s"],
        "wallclock_duration_s": meta["wallclock_duration_s"],
        "n_agents_requested": meta["n_agents_requested"],
        "n_alive": meta["n_alive_after_phase1"],
        "n_dead": meta["n_dead_after_phase1"],
        "n_events_ok": meta["n_events_ok"],
        "n_events_failed": meta["n_events_failed"],
        "models_used": ",".join(p["short_name"] for p in meta["pool"]),
        "run_dir": str(run_dir.relative_to(out_dir)),
    }
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=INDEX_FIELDS, quoting=csv.QUOTE_MINIMAL)
        if write_header:
            w.writeheader()
        w.writerow(new_row)
    return path
