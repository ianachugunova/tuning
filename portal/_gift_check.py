"""What a denser field costs, and who would need a gift.

The question: lengthen every tone to ~8-15 s (the field becomes continuous
while people are present, free) and soften the empty-field stretch from x11
towards x5 or x3 (the site sounds fuller when nobody is there). Neither adds
a dollar to the total — the card burns at its price regardless — but both
move money *between* wallets, and the wallets were calibrated to a calendar
of deaths. This prints how far each variant moves each death with the seeded
wallets, and what wallet each voice would have needed instead. The gift is
the difference, and it can only ever be positive: the log can be given to,
never taken from.

Run: python3 _gift_check.py
"""

import mortality as M
import mortality_sim as S

SEEDED = dict(M.PLANNED_WALLETS_NINE)
TARGETS = dict(S.TARGETS_NINE)
CAST = [v for v, _ in S.TARGETS_NINE]


def run_with(breath_s: float, stretch: float, wallets: dict) -> "S.Run":
    keep_b, keep_a = S.BREATH_S, M.ALONE_STRETCH
    S.BREATH_S, M.ALONE_STRETCH = breath_s, stretch
    try:
        run = S.Run(CAST, wallets, S.SCENARIOS["nominal"])
        run.go(S.END + 10 * 86400)
        return run
    finally:
        S.BREATH_S, M.ALONE_STRETCH = keep_b, keep_a


def calibrate_with(breath_s: float, stretch: float) -> dict:
    keep_b, keep_a = S.BREATH_S, M.ALONE_STRETCH
    S.BREATH_S, M.ALONE_STRETCH = breath_s, stretch
    try:
        return S.calibrate(S.TARGETS_NINE, S.SCENARIOS["nominal"])
    finally:
        S.BREATH_S, M.ALONE_STRETCH = keep_b, keep_a


def drift_table(run: "S.Run") -> dict:
    """Hours each death lands from its composed hour. Positive = late."""
    when = {d["voice_id"]: d["timestamp"] for d in run.deaths}
    return {v: (when[v] - TARGETS[v]) / 3600.0 if v in when else None
            for v in CAST}


VARIANTS = [
    ("today: tone 5 s, empty field x11", 5.0, 11.0),
    ("long tones, x11 kept", 11.5, 11.0),
    ("long tones, x5", 11.5, 5.0),
    ("long tones, x3", 11.5, 3.0),
]

print("DEATH DRIFT with the seeded wallets, hours from the composed hour")
print("(positive = dies late; the first row is the plan and should be ~0)")
print()
hdr = f"{'voice':<15}" + "".join(f"{lbl.split(',')[1].strip():>12}"
                                 if "," in lbl else f"{'plan':>12}"
                                 for lbl, _, _ in VARIANTS)
runs = [run_with(b, a, SEEDED) for _, b, a in VARIANTS]
drifts = [drift_table(r) for r in runs]
print(f"{'voice':<15}" + "".join(f"{lbl:>24}" for lbl, _, _ in VARIANTS))
for v in CAST:
    row = f"{S.NAMES[v]:<15}"
    for d in drifts:
        h = d[v]
        row += f"{'alive at end' if h is None else f'{h:+.1f} h':>24}"
    print(row)

print()
print("WALLET each voice would have needed, from birth, per variant")
print("(gift = needed - seeded; negative means it now has too much and")
print(" would die late — money can be given but never taken back)")
print()
tables = [calibrate_with(b, a) for _, b, a in VARIANTS[1:]]
print(f"{'voice':<15}{'seeded':>10}" + "".join(
    f"{lbl.split(',')[1].strip():>16}" for lbl, _, _ in VARIANTS[1:]))
gift_sum = [0.0] * len(tables)
for v in CAST:
    row = f"{S.NAMES[v]:<15}{SEEDED[v]:>10.2f}"
    for i, tb in enumerate(tables):
        diff = tb[v] - SEEDED[v]
        if diff > 0:
            gift_sum[i] += diff
        row += f"{diff:>+16.2f}"
    print(row)
print(f"{'gift total':<15}{'':>10}" + "".join(
    f"{g:>16.2f}" for g in gift_sum))
print()
for (lbl, _, _), tb in zip(VARIANTS[1:], tables):
    print(f"  {lbl}: needed total ${sum(tb.values()):,.2f} "
          f"(declared ${M.DECLARED_USD:,.2f})")
