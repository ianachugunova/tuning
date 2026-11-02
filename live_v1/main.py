"""CLI entry-point for the live Tuning Meditation orchestrator.

Usage:
    python3 -m Tuning_Meditation_LLMs.live_v1.main --duration 180 --agents 10
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

# Load the .env file that lives next to the older prototype, where the
# OpenRouter key already is.
load_dotenv(REPO_ROOT / "Tuning_Meditation_LLMs" / ".env")

if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))
    from Tuning_Meditation_LLMs.live_v1.models import pick_pool, pick_small_pool  # type: ignore
    from Tuning_Meditation_LLMs.live_v1.orchestrator import run_session  # type: ignore
else:
    from .models import pick_pool, pick_small_pool
    from .orchestrator import run_session


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Tuning Meditation orchestrator")
    p.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Total session duration in seconds (default: 180 = 3 min)",
    )
    p.add_argument(
        "--agents",
        type=int,
        default=10,
        help="Number of LLM agents (default: 10). Use e.g. 30 for a large choir — "
             "cost and rate limits scale roughly linearly; see models.pick_pool.",
    )
    p.add_argument(
        "--pool",
        choices=["primary", "small"],
        default="primary",
        help="Which cast to run: 'primary' = the June choir of large proprietary "
             "models; 'small' = the open-weight models proposed for October.",
    )
    p.add_argument(
        "--humans",
        type=int,
        default=0,
        help="Number of human performers (microphone + Whisper). Default 0. "
             "USE HEADPHONES if combining with --humans > 0 and live sound playback.",
    )
    p.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size: tiny / base / small / medium / large-v3. "
             "Default 'base'. 'tiny' is fastest but may garble singing.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "out",
        help="Where to write the JSONL log",
    )
    p.add_argument(
        "--sound",
        action="store_true",
        help="Spawn `python3 -m live_v1.sound` for this run (macOS: TTS for each "
        "tone event). Requires `say` and `afplay`. Use headphones if --humans > 0.",
    )
    return p.parse_args()


async def _main() -> None:
    args = parse_args()
    pool = (
        pick_small_pool(args.agents)
        if args.pool == "small"
        else pick_pool(args.agents)
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    await run_session(
        duration_s=args.duration,
        pool=pool,
        out_dir=args.out_dir,
        n_humans=args.humans,
        whisper_model=args.whisper_model,
        sound_follower=args.sound,
    )


if __name__ == "__main__":
    asyncio.run(_main())
