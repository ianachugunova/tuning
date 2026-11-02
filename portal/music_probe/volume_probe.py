"""Does a file written into /weights survive the container? A probe.

    .venv-modal/bin/modal run portal/music_probe/volume_probe.py

`bodies_probe.py` printed BORN for all eight voices on two consecutive runs,
where the second should have printed found. Either the commit does not persist
or the second container cannot see it, and the difference matters: the whole
design rests on a body being generated once and then never again.

Two functions so there are two containers, and nothing else in the picture —
no GPU, no synthesiser, no choir. One writes and commits, the other reloads
and looks.
"""

from __future__ import annotations

import modal

app = modal.App("tuning-volume-probe")

weights_volume = modal.Volume.from_name("tuning-weights", create_if_missing=True)
CACHE_DIR = "/weights"
PROBE = f"{CACHE_DIR}/bodies/_probe.txt"

image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={CACHE_DIR: weights_volume}, timeout=300)
def write_one() -> str:
    import os
    import time

    os.makedirs(os.path.dirname(PROBE), exist_ok=True)
    stamp = str(time.time())
    with open(PROBE, "w") as fh:
        fh.write(stamp)
    print(f"wrote {PROBE}, exists before commit: {os.path.exists(PROBE)}")
    weights_volume.commit()
    print(f"committed, exists after commit: {os.path.exists(PROBE)}")
    return stamp


@app.function(image=image, volumes={CACHE_DIR: weights_volume}, timeout=300)
def read_one() -> str | None:
    import os

    weights_volume.reload()
    listing = os.listdir(f"{CACHE_DIR}/bodies") if os.path.isdir(
        f"{CACHE_DIR}/bodies") else "no such directory"
    print(f"/weights/bodies -> {listing}")
    if not os.path.exists(PROBE):
        return None
    with open(PROBE) as fh:
        return fh.read()


@app.local_entrypoint()
def main():
    written = write_one.remote()
    found = read_one.remote()
    print("\n" + "=" * 70)
    print(f"  wrote  {written}")
    print(f"  found  {found}")
    print("  " + ("the volume persists" if written == found
                  else "THE VOLUME DOES NOT PERSIST"))
    print("=" * 70)
