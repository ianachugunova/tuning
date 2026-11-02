#!/usr/bin/env python3
"""
live_v1/analyze.py — Post-run analysis and cross-run summary.

Per run:
  - detects dominant attractor (vowels / frequencies / prose / mixed)
  - extracts top vocabulary
  - profiles each agent
  - writes  run_<id>/analysis.md

Cross-run:
  - writes  out/RUNS_SUMMARY.md  comparing all runs

Usage:
    python3 -m live_v1.analyze                         # latest run
    python3 -m live_v1.analyze --run run_2026-...      # specific run
    python3 -m live_v1.analyze --all                   # all runs + summary
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Optional

RUNS_PARENT = Path(__file__).parent / "out"

# ---------------------------------------------------------------------------
# Attractor detection
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a","an","the","and","or","of","to","in","is","are","for","that","this",
    "with","as","by","at","from","it","its","i","my","me","we","our","you",
    "your","be","been","have","has","had","do","does","did","will","would",
    "can","could","may","might","not","no","on","so","but","if","than","then",
    "into","out","up","their","they","them","which","who","what","how","when",
    "where","there","here","about","also","each","more","some","such","other",
    "than","very","just","one","two","three","new","tone","tones","voice",
}

VOWEL_RE = re.compile(r"^[aeiouAEIOU]+$")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:hz|khz|mhz|ghz|nm|ms|bpm)?\b", re.I)


def detect_attractor(contents: list[str]) -> str:
    """Return a short label for what the run converged on."""
    if not contents:
        return "silence"

    word_counts = [len(c.split()) for c in contents]
    avg_words = sum(word_counts) / len(word_counts)

    # purely one-word outputs → vowel sounds or single tokens
    if avg_words < 2:
        vowel_frac = sum(1 for c in contents if VOWEL_RE.match(c.strip())) / len(contents)
        if vowel_frac > 0.5:
            return "vowel sounds"
        return "single tokens"

    # lots of numbers / unit strings → frequencies / numerical
    all_text = " ".join(contents)
    num_matches = len(NUMBER_RE.findall(all_text))
    if num_matches > len(contents) * 0.8:
        return "numerical / frequencies"

    # long prose
    if avg_words > 40:
        return "long prose"
    if avg_words > 10:
        return "short prose"

    return "mixed"


def top_words(contents: list[str], n: int = 15) -> list[tuple[str, int]]:
    words = []
    for c in contents:
        for w in re.findall(r"[a-zA-Z']+", c.lower()):
            if w not in STOPWORDS and len(w) > 2:
                words.append(w)
    return Counter(words).most_common(n)


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------


def load_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    interps_path = run_dir / "interpretations.json"
    interps = json.loads(interps_path.read_text(encoding="utf-8")) if interps_path.exists() else []
    events = [
        json.loads(l)
        for l in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    return {"meta": meta, "interps": interps, "events": events, "run_dir": run_dir}


def analyse_run(data: dict) -> dict:
    meta = data["meta"]
    interps = data["interps"]
    events = data["events"]

    tones = [e for e in events if e.get("kind") == "tone"]
    contents = [e["content"] for e in tones if e.get("content")]

    attractor = detect_attractor(contents)
    vocab = top_words(contents)

    # per-agent profile
    interp_map = {e["voice_id"]: e for e in interps}
    agent_profiles = []
    for entry in meta.get("agents", []):
        vid = entry["voice_id"]
        short = entry.get("short_name", vid)
        alive = entry.get("alive", False)
        agent_tones = [e for e in tones if e["voice_id"] == vid]
        lengths = [len(e["content"].split()) for e in agent_tones if e.get("content")]
        avg_len = round(sum(lengths) / len(lengths), 1) if lengths else 0
        n_tune = sum(1 for e in agent_tones if e.get("action") == "TUNE")
        n_new  = sum(1 for e in agent_tones if e.get("action") == "NEW")
        interp = (interp_map.get(vid) or {}).get("interpretation") or {}
        # New schema (post-2026-05-12_20-XX): line-by-line translation fields.
        # Old schema fallback to tone_concept for older runs.
        inner_ear        = interp.get("inner_ear",        interp.get("tone_concept",   "—"))
        distance_concept = interp.get("distance_concept", "—")
        detect_absent    = interp.get("detect_absent",    "—")
        alternation_rule = interp.get("alternation_rule", "—")
        personal_score   = interp.get("personal_score",   interp.get("my_algorithm", "—"))
        breath_dur       = interp.get("breath_duration",  "—")
        # pick one interesting sample (longest content)
        sample = max(agent_tones, key=lambda e: len(e.get("content", "")), default=None)
        agent_profiles.append({
            "voice_id": vid,
            "model_short": short,
            "alive": alive,
            "n_tones": len(agent_tones),
            "avg_words": avg_len,
            "n_tune": n_tune,
            "n_new": n_new,
            "inner_ear": inner_ear,
            "distance_concept": distance_concept,
            "detect_absent": detect_absent,
            "alternation_rule": alternation_rule,
            "personal_score": personal_score,
            "breath_duration": breath_dur,
            "best_sample": sample["content"] if sample else "",
        })

    return {
        "run_id": meta["run_id"],
        "duration_s": meta.get("wallclock_duration_s", 0),
        "n_alive": meta.get("n_alive_after_phase1", 0),
        "n_agents": meta.get("n_agents_requested", 0),
        "n_events": meta.get("n_events_ok", 0),
        "attractor": attractor,
        "vocab": vocab,
        "agent_profiles": agent_profiles,
        "contents": contents,
    }


def write_run_analysis(data: dict, result: dict) -> Path:
    run_dir = data["run_dir"]
    r = result
    lines = []
    a = lines.append

    a(f"# Run analysis: {r['run_id']}\n")
    a(f"**Duration:** {r['duration_s']:.0f}s  |  "
      f"**Alive:** {r['n_alive']}/{r['n_agents']}  |  "
      f"**Events:** {r['n_events']}\n")
    a(f"## Attractor\n")
    a(f"**`{r['attractor']}`**\n")

    if r["vocab"]:
        vocab_str = "  ".join(f"`{w}` ×{n}" for w, n in r["vocab"][:12])
        a(f"**Top words:** {vocab_str}\n")

    a(f"## Agent profiles\n")
    a("| voice | model | alive | tones | avg words | TUNE/NEW | breath |")
    a("|---|---|---|---|---|---|---|")
    for p in r["agent_profiles"]:
        alive_icon = "✓" if p["alive"] else "✗"
        bd = str(p["breath_duration"])[:40].replace("|", "/").replace("\n", " ")
        a(f"| {p['voice_id']} | {p['model_short']} | {alive_icon} | "
          f"{p['n_tones']} | {p['avg_words']} | {p['n_tune']}/{p['n_new']} | {bd} |")

    a(f"\n## Personal scores\n")
    a("How each model rewrote the Oliveros score for itself:\n")
    for p in r["agent_profiles"]:
        if p["alive"] and p["personal_score"] != "—":
            a(f"\n**{p['voice_id']} ({p['model_short']}):**")
            a(f"> {p['personal_score']}")

    a(f"\n## What each agent interpreted as \"tone\" / listening\n")
    for p in r["agent_profiles"]:
        if p["alive"] and p["inner_ear"] != "—":
            a(f"**{p['voice_id']} ({p['model_short']}):** {p['inner_ear']}\n")

    a(f"\n## Distance concept — how each agent measures \"distant voice\"\n")
    for p in r["agent_profiles"]:
        if p["alive"] and p["distance_concept"] != "—":
            a(f"**{p['voice_id']} ({p['model_short']}):** {p['distance_concept']}\n")

    a(f"\n## Detecting absent — how each agent finds what is missing in the field\n")
    for p in r["agent_profiles"]:
        if p["alive"] and p["detect_absent"] != "—":
            a(f"**{p['voice_id']} ({p['model_short']}):** {p['detect_absent']}\n")

    a(f"\n## Alternation rule — how each agent chooses between own tone and tuning\n")
    for p in r["agent_profiles"]:
        if p["alive"] and p["alternation_rule"] != "—":
            a(f"**{p['voice_id']} ({p['model_short']}):** {p['alternation_rule']}\n")

    a(f"\n## Notable outputs (longest per agent)\n")
    for p in r["agent_profiles"]:
        if p["alive"] and p["best_sample"]:
            sample = textwrap.fill(p["best_sample"], width=100)
            a(f"**{p['voice_id']} ({p['model_short']}):**")
            a(f"```")
            a(sample[:600])
            a(f"```\n")

    path = run_dir / "analysis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Cross-run summary
# ---------------------------------------------------------------------------


def write_runs_summary(all_results: list[dict], out_dir: Path) -> Path:
    # Load hand-written observations if present
    obs_path = out_dir / "observations.json"
    observations: dict = {}
    if obs_path.exists():
        try:
            observations = json.loads(obs_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    lines = []
    a = lines.append

    a("# Tuning Meditation — Runs Summary\n")
    a(f"Total runs: {len(all_results)}\n")

    a("## Attractor per run\n")
    a("| run | duration | alive | events | attractor | top words | observations |")
    a("|---|---|---|---|---|---|---|")
    for r in all_results:
        vocab_str = "  ".join(w for w, _ in r["vocab"][:6])
        obs = (observations.get(r["run_id"]) or {}).get("notes", "")
        obs_cell = obs[:120].replace("|", "/").replace("\n", " ")
        a(f"| {r['run_id']} | {r['duration_s']:.0f}s | "
          f"{r['n_alive']}/{r['n_agents']} | {r['n_events']} | "
          f"**{r['attractor']}** | {vocab_str} | {obs_cell} |")

    a(f"\n## tone/listening convergence across runs\n")
    a("How each model defined 'tone' / 'inner ear' in each run:\n")
    all_voices = []
    seen = set()
    for r in all_results:
        for p in r["agent_profiles"]:
            key = (p["voice_id"], p["model_short"])
            if key not in seen:
                all_voices.append(key)
                seen.add(key)

    for vid, short in sorted(all_voices):
        a(f"\n**{vid} ({short}):**")
        for r in all_results:
            profile = next((p for p in r["agent_profiles"] if p["voice_id"] == vid), None)
            if profile and profile["alive"] and profile["inner_ear"] != "—":
                tc = profile["inner_ear"][:120]
                a(f"- `{r['run_id']}`: {tc}")

    path = out_dir / "RUNS_SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_latest_run(runs_dir: Path) -> Path:
    runs = sorted(runs_dir.glob("run_*/"), reverse=True)
    if not runs:
        raise SystemExit(f"No runs found in {runs_dir}")
    return runs[0]


def resolve_run_dir(run_arg: Optional[str], runs_dir: Path) -> Path:
    if run_arg is None:
        return find_latest_run(runs_dir)
    p = Path(run_arg)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    candidate = runs_dir / p
    if candidate.exists():
        return candidate.resolve()
    raise SystemExit(f"Run not found: {run_arg!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse Tuning Meditation run(s)")
    ap.add_argument("--run", default=None, help="Run name or path (default: latest)")
    ap.add_argument("--out-dir", default=str(RUNS_PARENT))
    ap.add_argument("--all", action="store_true", help="Analyse all runs + write RUNS_SUMMARY.md")
    args = ap.parse_args()

    runs_dir = Path(args.out_dir).resolve()

    if args.all:
        run_dirs = sorted(runs_dir.glob("run_*/"), reverse=True)
        if not run_dirs:
            raise SystemExit(f"No runs in {runs_dir}")
        all_results = []
        for rd in run_dirs:
            try:
                data = load_run(rd)
                result = analyse_run(data)
                path = write_run_analysis(data, result)
                print(f"  ✓ {rd.name}  attractor={result['attractor']}  → {path.name}")
                all_results.append(result)
            except Exception as exc:
                print(f"  ✗ {rd.name}  {exc}")
        if all_results:
            summary_path = write_runs_summary(all_results, runs_dir)
            print(f"\n→ {summary_path}")
    else:
        run_dir = resolve_run_dir(args.run, runs_dir)
        data = load_run(run_dir)
        result = analyse_run(data)
        path = write_run_analysis(data, result)
        print(f"attractor : {result['attractor']}")
        print(f"top words : {' '.join(w for w,_ in result['vocab'][:10])}")
        print(f"→ {path}")


if __name__ == "__main__":
    main()
