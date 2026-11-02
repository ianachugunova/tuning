# The room — Mac mini at Gray Area

One process, `portal/room.py`, is the body of the choir in the hall. It plays
each voice's breaths on its own speaker and turns the five microphones into
numbers for the portal. Nothing else runs on the Mac mini; the kiosk screen
just shows `https://tuning-grayarea--tuning-portal-web.modal.run/field`.

## Wiring (what the software assumes)

```
Modal (choir, portal) ──HTTPS──▶ Mac mini ──USB──▶ Behringer UMC1820
                                                  out 1–2 (MAIN OUT, TRS) + out 3–9 (line, TRS) → nine monitors
                                                  in  1–5 (XLR, Midas preamps, +48 V)            ← five microphones
Five SM57/58 + Cubilux PTT → XLR → UMC1820 inputs 1–5   (no mixer in between)
```

Channel numbers are 1-based, as printed on the interface; CoreAudio counts
the UMC1820's ten analog outputs 1–10 and eight analog inputs 1–8 in the same
order, so the defaults in `room.example.json` (voices on outputs 1–9, mics on
inputs 1–5) should be right as cabled. `listen --print` tells you which
channel moves if they are not.

Four switches on the UMC1820 itself, set once and taped:

- **+48V** on for bank **1–4** and for bank **5–8** (mic 5 is in the second
  bank). The dynamic mics do not need it; the PTT buttons' LEDs do. Nothing
  else is plugged into 6–8, so the second bank powering empty jacks is fine.
- **PAD** off on 1–5; **GAIN** so a person singing into the mic lights the
  green SIG LED steadily and never the red CLIP. Turn it up until the hall
  alone barely moves the LED, then back a little.
- **MIX** knob (direct monitoring) fully to **PB** (playback). Anything
  toward INPUTS sends the microphones straight to MAIN OUT, i.e. to
  speakers 1 and 2 — feedback, and the room singing to itself.
- **MAIN OUT** knob: it only affects outputs 1–2, so the two voices on those
  speakers would be quieter or louder than the seven on fixed-level line
  outs. Set it so speaker 1 matches speaker 3 on `tone --all`, then tape it.
  (Or put the two 8″ monitors on 1–2 and let the knob be their trim.)

## Install (once, 10 minutes)

```bash
# 1. Python 3 is on macOS; add the two libraries
pip3 install sounddevice numpy

# 2. The repo (no secrets needed; the portal is public)
cd ~ && git clone https://github.com/ianachugunova/tuning-meditation-llms.git
cd tuning-meditation-llms/portal

# 3. Find the interface's name as CoreAudio sees it
python3 room.py devices
#  > 2 UMC1820, Core Audio (18 in, 20 out)      ← something like this

# 4. Config
cp room.example.json room.json
#   edit: output_device / input_device = a substring of the name ("UMC1820")
#         channels = which output each voice sits on
#         mics = the five input channels the microphones are in
```

Set the UMC1820 to 48 kHz in Audio MIDI Setup (or leave 44.1; the software
reads whatever the device is set to). No driver: macOS sees it as soon as it
is plugged in.

Mac mini settings: System Settings → Energy → never sleep, start up after
power failure; Lock Screen → never; Users → auto-login; Software Update →
automatic updates off for the week.

## Test (walk the ring)

```bash
python3 room.py tone --channel 1              # 2 s of A3 on output 1
python3 room.py tone --all                    # outputs 1..9 in turn, naming each voice
```

Stand in the room and check the voice named in the terminal is the speaker
that sounds. Fix `channels` in `room.json` rather than re-plugging cables.

```bash
python3 room.py listen --print                # measures, posts nothing
```

Sing a held note into each mic. One line per utterance:

```
18:25:40  mic3  1.5s rms 0.212 gate 0.020  A3 ~220Hz [very steady]  → human_gallery3_f4ea
18:25:41  mic4  0.3s rms 0.210 gate 0.020  — too short for a sung note
```

If a mic never shows up, the `mics` list is wrong — try `--mics 3,4,5,6,7`.
If the hall shows up as notes, raise the input GAIN so a voice sits well
above the room: the gate opens at rms 0.02 and follows the room up to 0.10.
Nothing is sent to the portal in `--print` mode.

```bash
python3 room.py play                          # the nine speakers, live choir
```

The first breath arrives within ~30 s (the portal tails the choir's log from a
volume; the page has the same delay). Each printed line is one breath, its
channel and voice.

## Run for the week

```bash
python3 room.py run                           # play + listen, until Ctrl-C
```

Autostart so a power cycle from the LPC-3 is a complete repair:

```bash
cp room.plist.example ~/Library/LaunchAgents/com.tuning.room.plist
# edit the two paths inside to where the repo actually is
launchctl load ~/Library/LaunchAgents/com.tuning.room.plist
tail -f /tmp/tuning-room.log
```

`launchctl unload …` stops it. The process keeps nothing on disk; audio from
the microphones is measured in memory and discarded.

## Options worth knowing

| flag | what |
|---|---|
| `--gain '{"voice_00": 0.7}'` | software trim per voice, on top of the monitors' own knobs |
| `--humans-channel 9` | also sound people singing from phones/zoom (as a sine) on one speaker; the room's own mics are never played back |
| `--room gallery` | prefix of the people this room contributes: `human_gallery3_xxxx` (mic 3) |
| `--portal URL` | a different portal, e.g. a test deployment |

## What the software does and does not decide

- It mirrors the phone page: same YIN (`live_v1/human.py`), same gallery gate
  (`pitch-core.js`, `GALLERY`), same posted fields (`POST /breath`,
  `source: "gallery"`). A person at a room mic and a person on a phone are
  notated to one standard.
- Each microphone is its own person in the field. A turn ends after 90 s of
  silence at that mic, then the next person gets a new id.
- A dead voice sends no breaths, so its speaker goes quiet by itself. Nothing
  here knows about money.
- At most 20 breaths a minute per mic; the hall feeding back into itself costs
  the score twenty lines, not a night.
