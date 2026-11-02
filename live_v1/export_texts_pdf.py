#!/usr/bin/env python3
"""Export a clean, typeset PDF transcript of all texts the LLM voices produced
during a "Tuning Meditation for LLMs" concert run.

Reads the run's events.jsonl + meta.json and produces a keepsake PDF with:
  1. a title page,
  2. a legend (voice_id -> model),
  3. a chronological "libretto" of every produced line, and
  4. a per-model section so a reader can follow one voice's arc.

Texts that were generated but not played aloud (playback == "none") are
included and tagged "[unvoiced]".

Usage:
    python export_texts_pdf.py --run run_2026-06-07_03-11-12
    python export_texts_pdf.py --run run_2026-06-07_03-11-12 --out /path/out.pdf

This script is self-contained and read-only with respect to the run data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# --- Fonts -----------------------------------------------------------------
# The texts contain IPA, combining diacritics, math symbols and em-dashes, so
# we register a broad-coverage Unicode TTF. macOS ships Arial Unicode MS, which
# covers the vast majority of glyphs the models emitted.
BODY_FONT = "BodyUnicode"
LABEL_FONT = "Helvetica-Bold"  # crisp ASCII for time/model labels

_UNICODE_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode MS.ttf",
]


def register_fonts() -> None:
    global BODY_FONT
    for path in _UNICODE_FONT_CANDIDATES:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(BODY_FONT, path))
            return
    # Fallback: use Helvetica (limited unicode coverage) so we still produce a PDF.
    print(
        "WARNING: no broad-coverage Unicode font found; falling back to Helvetica "
        "(some glyphs may not render).",
        file=sys.stderr,
    )
    BODY_FONT = "Helvetica"


# --- Data loading ----------------------------------------------------------

def fmt_offset(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def load_run(run_dir: str):
    meta_path = os.path.join(run_dir, "meta.json")
    events_path = os.path.join(run_dir, "events.jsonl")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    started = float(meta.get("started_at_unix", 0.0))

    # voice_id -> info, preserving pool order. LLM voices come from meta.json's
    # pool; human voices (if any) are discovered from the event stream below.
    pool = OrderedDict()
    for entry in meta.get("pool", []):
        pool[entry["voice_id"]] = {
            "short_name": entry.get("short_name", entry["voice_id"]),
            "provider": entry.get("provider", ""),
            "slug": entry.get("slug", ""),
            "is_human": False,
        }

    human_ids = set()  # human_* voice_ids actually present in this run
    entries = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("kind") != "tone":
                continue
            voice_id = ev.get("voice_id", "")
            is_human = voice_id.startswith("human")
            if not (voice_id.startswith("voice_") or is_human):
                continue  # ignore anything that isn't a recognised voice
            content = (ev.get("content") or "").strip()
            if not content:
                continue
            if is_human:
                human_ids.add(voice_id)
            ts = float(ev.get("timestamp", started))
            entries.append(
                {
                    "timestamp": ts,
                    "offset": ts - started,
                    "voice_id": voice_id,
                    "model_short": ev.get("model_short", pool.get(voice_id, {}).get("short_name", voice_id)),
                    "content": content,
                    "playback": ev.get("playback", ""),
                    "unvoiced": ev.get("playback") == "none",
                    "is_human": is_human,
                    "action": ev.get("action", ""),
                    "tuning_to": ev.get("tuning_to") or None,
                    "breath_chosen_s": ev.get("breath_chosen_s"),
                }
            )

    # Register discovered human voices after the LLM pool so the legend and the
    # by-voice section list them in a stable, sensible order.
    for hid in sorted(human_ids):
        pool[hid] = {
            "short_name": "Human voice",
            "provider": "in the room",
            "slug": "",
            "is_human": True,
        }

    entries.sort(key=lambda e: e["timestamp"])
    return meta, pool, entries


def voice_short(vid, pool):
    """Human-readable model/voice name for a voice_id."""
    info = pool.get(vid)
    if info:
        return info["short_name"]
    if str(vid).startswith("human"):
        return "Human voice"
    return vid


def harmonic_phrase(e, pool):
    """Compact description of what the voice was doing harmonically."""
    action = (e.get("action") or "").upper()
    target = e.get("tuning_to")
    if e.get("is_human") or action == "HUMAN":
        phrase = "sung pitch (human)"
    elif target:
        phrase = f"tuning to {target} ({voice_short(target, pool)})"
    elif action == "NEW":
        phrase = "new pitch"
    elif action == "TUNE":
        phrase = "tuning (open)"
    else:
        phrase = "new pitch"
    breath = e.get("breath_chosen_s")
    if isinstance(breath, (int, float)) and breath > 0:
        phrase += f" &#183; held {breath:g}s"
    return phrase


# --- PDF building ----------------------------------------------------------

def make_styles():
    styles = {}
    styles["title"] = ParagraphStyle(
        "title", fontName=LABEL_FONT, fontSize=26, leading=32,
        alignment=TA_CENTER, spaceAfter=18,
    )
    styles["subtitle"] = ParagraphStyle(
        "subtitle", fontName=BODY_FONT, fontSize=13, leading=18,
        alignment=TA_CENTER, textColor=colors.HexColor("#333333"), spaceAfter=10,
    )
    styles["note"] = ParagraphStyle(
        "note", fontName=BODY_FONT, fontSize=10.5, leading=15,
        alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
    )
    styles["section"] = ParagraphStyle(
        "section", fontName=LABEL_FONT, fontSize=16, leading=20,
        spaceBefore=6, spaceAfter=14, textColor=colors.HexColor("#1a1a1a"),
    )
    styles["legend"] = ParagraphStyle(
        "legend", fontName=BODY_FONT, fontSize=10.5, leading=15,
    )
    styles["entry"] = ParagraphStyle(
        "entry", fontName=BODY_FONT, fontSize=10.5, leading=15, spaceAfter=10,
    )
    styles["model_head"] = ParagraphStyle(
        "model_head", fontName=LABEL_FONT, fontSize=13, leading=17,
        spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#1a1a1a"),
    )
    return styles


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#999999"))
    page_num = canvas.getPageNumber()
    if page_num > 1:  # skip the title page
        canvas.drawString(doc.leftMargin, 0.5 * inch, "Tuning Meditation for LLMs")
        canvas.drawRightString(
            doc.pagesize[0] - doc.rightMargin, 0.5 * inch, f"{page_num}"
        )
    canvas.restoreState()


def entry_paragraph(e, pool, styles, show_voice=True):
    display_name = "Human" if e.get("is_human") else e["model_short"]
    label_bits = [f"[{fmt_offset(e['offset'])}]", escape(display_name)]
    if show_voice:
        label_bits.append(f"({e['voice_id'].replace('voice_', 'v')})")
    label = "  ".join(label_bits)
    tag = ' <font color="#999999">[unvoiced]</font>' if e["unvoiced"] else ""
    harmonic = (
        f'<br/><font color="#7a7a7a">&#8594; {harmonic_phrase(e, pool)}</font>'
    )
    text = escape(e["content"])
    markup = (
        f'<font name="{LABEL_FONT}" color="#1a1a1a">{label}</font>{tag}'
        f'{harmonic}<br/>{text}'
    )
    return Paragraph(markup, styles["entry"])


def build_pdf(meta, pool, entries, out_path):
    styles = make_styles()

    doc = BaseDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="Tuning Meditation for LLMs",
        author="Tuning Meditation for LLMs",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height, id="main",
    )
    doc.addPageTemplates([
        PageTemplate(id="all", frames=[frame], onPage=header_footer),
    ])

    story = []

    # --- Title page ---
    story.append(Spacer(1, 2.2 * inch))
    story.append(Paragraph("Tuning Meditation for LLMs", styles["title"]))
    story.append(Paragraph(
        "A choir of ten large language models<br/>"
        "Center for New Music, San Francisco &#183; June 6, 2026",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 0.4 * inch))
    n_unvoiced = sum(1 for e in entries if e["unvoiced"])
    n_human = sum(1 for e in entries if e.get("is_human"))
    human_clause = (
        f" Human voices in the room are included too ({n_human} lines)."
        if n_human else ""
    )
    story.append(Paragraph(
        "This is a transcript of every text the models produced during the "
        "~10-minute performance &mdash; the lines they sang into the room and "
        "those they generated in silence (marked <i>[unvoiced]</i>)." + human_clause,
        styles["note"],
    ))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(
        "Under each line, <font color=\"#7a7a7a\">&#8594;</font> notes what the voice "
        "was doing harmonically: <i>tuning to</i> another named voice, or sounding a "
        "<i>new pitch</i>.",
        styles["note"],
    ))
    story.append(Paragraph(
        f"{len(entries)} entries &#183; {len(entries) - n_unvoiced} voiced &#183; "
        f"{n_unvoiced} unvoiced",
        styles["note"],
    ))
    story.append(PageBreak())

    # --- Legend ---
    story.append(Paragraph("The Voices", styles["section"]))
    legend_rows = []
    for vid, info in pool.items():
        provider = f" ({info['provider']})" if info["provider"] else ""
        legend_rows.append([
            Paragraph(f'<font name="{LABEL_FONT}">{escape(vid)}</font>', styles["legend"]),
            Paragraph(f'{escape(info["short_name"])}{escape(provider)}', styles["legend"]),
        ])
    legend_table = Table(legend_rows, colWidths=[1.4 * inch, doc.width - 1.4 * inch])
    legend_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#e2e2e2")),
    ]))
    story.append(legend_table)
    story.append(PageBreak())

    # --- Primary: chronological libretto ---
    story.append(Paragraph("Libretto &#8212; in performance order", styles["section"]))
    for e in entries:
        story.append(entry_paragraph(e, pool, styles, show_voice=True))
    story.append(PageBreak())

    # --- Secondary: grouped by model ---
    story.append(Paragraph("By Voice &#8212; each model's arc", styles["section"]))
    for vid, info in pool.items():
        model_entries = [e for e in entries if e["voice_id"] == vid]
        if not model_entries:
            continue
        provider = f" ({info['provider']})" if info["provider"] else ""
        story.append(Paragraph(
            f"{escape(info['short_name'])}{escape(provider)} "
            f"&#183; {escape(vid)} &#183; {len(model_entries)} lines",
            styles["model_head"],
        ))
        for e in model_entries:
            story.append(entry_paragraph(e, pool, styles, show_voice=False))

    doc.build(story)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", default="run_2026-06-07_03-11-12",
        help="Run id or path to a run directory under live_v1/out/.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output PDF path (default: <run_dir>/llm_texts.pdf).",
    )
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    run_arg = args.run
    if os.path.isdir(run_arg):
        run_dir = run_arg
    else:
        run_dir = os.path.join(here, "out", run_arg)
    if not os.path.isdir(run_dir):
        sys.exit(f"Run directory not found: {run_dir}")

    out_path = args.out or os.path.join(run_dir, "llm_texts.pdf")

    register_fonts()
    meta, pool, entries = load_run(run_dir)

    build_pdf(meta, pool, entries, out_path)

    # --- Report ---
    per_model = Counter(e["model_short"] for e in entries)
    n_unvoiced = sum(1 for e in entries if e["unvoiced"])
    n_human = sum(1 for e in entries if e.get("is_human"))
    n_tuning = sum(1 for e in entries if e.get("tuning_to"))
    n_new = sum(1 for e in entries if not e.get("tuning_to") and not e.get("is_human"))
    size = os.path.getsize(out_path)
    print(f"Wrote {out_path} ({size:,} bytes)")
    print(f"Total entries: {len(entries)}  (voiced {len(entries) - n_unvoiced}, unvoiced {n_unvoiced})")
    print(f"Harmonic: {n_tuning} tuning-to-another, {n_new} new/independent pitch")
    print(f"Human-voice lines: {n_human}")
    print("Per-voice counts:")
    for model, count in sorted(per_model.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {model:>20s}: {count}")


if __name__ == "__main__":
    main()
