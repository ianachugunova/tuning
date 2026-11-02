# Testing five (or more) microphone channels on a Mac **without** M32R

You do **not** need a virtual machine (VMware, Parallels, etc.). You need a **virtual multichannel audio device** so macOS exposes many input channels through one “cable” — exactly what USB consoles like the MIDAS M32R do in hardware.

## Recommended: BlackHole 16ch (free)

1. Download **BlackHole 16ch** from [Existential Audio](https://existential.audio/blackhole/).
2. Install, reboot if the installer asks.
3. Run:

   ```bash
   cd ~/Documents/GitHub/Tuning_Meditation_LLMs
   python3 -m live_v1.concert --list-devices
   ```

   You should see something like **BlackHole 16ch** with **16 inputs / 16 outputs**.

4. In `live_v1/concert.py`, set for a dry run:

   ```python
   INPUT_DEVICE = "BlackHole"   # substring of the device name
   INPUT_CHANNELS_COUNT = 5
   HUMAN_CHANNELS = {
       "human_00": 1,
       "human_01": 2,
       "human_02": 3,
       "human_03": 4,
       "human_04": 5,
   }
   ```

5. **Feed audio into BlackHole**  
   The multiplexer opens BlackHole as the capture device; it does **not** create sound by itself. For a test you can:

   - Play five different tracks (or test tones) from Logic, Ableton, or Audacity, each routed to **BlackHole channels 1–5**, **or**
   - Use one real microphone on your built‑in input for one singer, and leave other channels silent to verify the pipeline starts without errors.

Until something is routed into those channels, VAD will usually hear silence — that is expected.

## Alternative: Rogue Amoeba Loopback (paid)

Loopback is easier for visual routing (drag wires between apps and channel numbers). Same idea: create a multi‑output device, point `INPUT_DEVICE` at it, set `INPUT_CHANNELS_COUNT` and `HUMAN_CHANNELS`.

## What you already have in code

- `MicMultiplexer` in `live_v1/human.py` — one `InputStream`, fan‑out per channel.
- `concert.py` — wires N humans to N channels when `INPUT_DEVICE` is not `None`.

At the **venue**, replace BlackHole with the **M32R** name string from `--list-devices` and the same channel map your engineer uses on the console.
