"""The room's software, without a sound card.

Everything that decides what is heard and what is posted is arithmetic on
arrays and dicts; the hardware is two callbacks around it. These tests pin the
arithmetic to the page it mirrors (`pitch-core.js`, `field.html`), so that a
person singing into the Motu is notated the same as a person singing into a
phone, and a breath from the choir lands on the speaker that is that voice.
"""
import io
import math
import time
import wave

import unittest

import numpy as np
import pytest

import room


def tone(hz, seconds, sr, amp=0.2):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class TheArithmetic(unittest.TestCase):
    def test_notes_and_cents(self):
        assert room.hz_to_note(440) == "A4"
        assert room.hz_to_note(146.8) == "D3"
        assert room.hz_to_note(0) == ""
        assert room.cents_off(440) == 0
        assert room.cents_off(440 * 2 ** (25 / 1200)) == 25

    def test_stability_words(self):
        assert room.stability_label(0, 220) == "very steady"
        assert room.stability_label(220 * (2 ** (50 / 1200) - 1), 220) == "steady"
        assert room.stability_label(220 * (2 ** (300 / 1200) - 1), 220) == "sliding"

    def test_contour_thins_to_quarter_seconds(self):
        pitches = [200.0] * 40 + [300.0] * 40
        c = room.contour_of(pitches, 2.0)
        assert len(c) == 8
        assert c[0] == 200.0 and c[-1] == 300.0
        assert room.contour_of([], 2.0) == []

    def test_wav_roundtrip_and_resample(self):
        x = tone(220, 0.5, 24000)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
            wf.writeframes((x * 32767).astype("<i2").tobytes())
        y, sr = room.decode_wav(buf.getvalue())
        assert sr == 24000 and len(y) == len(x)
        assert np.max(np.abs(y - x)) < 1e-3
        z = room.resample(y, 24000, 48000)
        assert abs(len(z) - 2 * len(y)) <= 1

    def test_sine_follows_a_contour(self):
        x = room.sine(220, 1.0, 16000, contour=[200, 300])
        assert len(x) == 16000 and np.max(np.abs(x)) <= 0.2
        p, _ = room.track_pitch(x[:4000], 16000)
        q, _ = room.track_pitch(x[-4000:], 16000)
        assert np.median(p) < 240 < np.median(q)

    def test_channel_spec(self):
        assert room.parse_channels("voice_00=1, voice_06=9") == {"voice_00": 1, "voice_06": 9}
        d = room.default_channels()
        assert len(d) == 9 and sorted(d.values()) == list(range(1, 10))


class TheGate(unittest.TestCase):
    """pitch-core GALLERY: four things true at once, or the hall is notated."""

    def test_a_sung_note_is_admitted(self):
        assert room.gallery_admits(2.0, 220, 1.0, 0.9, 0.05) == ""

    def test_the_hall_is_not(self):
        assert "short" in room.gallery_admits(0.8, 220, 1.0, 0.9, 0.05)
        assert "broken" in room.gallery_admits(2.0, 220, 1.0, 0.3, 0.05)
        assert "wandering" in room.gallery_admits(2.0, 220, 80.0, 0.9, 0.05)
        assert "quiet" in room.gallery_admits(2.0, 220, 1.0, 0.9, 0.01)
        assert "no pitch" in room.gallery_admits(2.0, 0, 0, 0, 0.05)

    def test_measure_a_real_note(self):
        sr = 48000
        u = {"audio": tone(196, 1.5, sr, amp=0.1), "onset": 0.0, "ended": 1.5,
             "tail_s": 0.5, "peak_rms": 0.07}
        m = room.measure(u, sr)
        assert m["why_not"] == ""
        assert m["note"] == "G3" and abs(m["hz"] - 196) < 2
        assert m["voiced_fraction"] > 0.9
        assert 5 <= len(m["contour_hz"]) <= 7

    def test_noise_is_not_a_voice(self):
        sr = 48000
        rng = np.random.default_rng(0)
        u = {"audio": (0.05 * rng.standard_normal(sr * 2)).astype(np.float32),
             "onset": 0.0, "ended": 2.0, "tail_s": 0.5, "peak_rms": 0.05}
        assert room.measure(u, sr)["why_not"] != ""


class OneMicrophone(unittest.TestCase):
    def feed_all(self, gate, blocks, sr):
        out, now = [], 0.0
        for b in blocks:
            now += len(b) / sr
            u = gate.feed(b, now)
            if u is not None:
                out.append(u)
        return out

    def test_an_utterance_closes_after_half_a_second_of_silence(self):
        sr = 16000
        gate = room.MicGate(sr)
        blk = int(sr * room.BLOCK_S)
        quiet = [np.zeros(blk, dtype=np.float32)] * 5
        loud = [tone(220, room.BLOCK_S, sr, amp=0.1)] * 15      # 1.5 s
        utts = self.feed_all(gate, quiet + loud + quiet * 2, sr)
        assert len(utts) == 1
        u = utts[0]
        assert abs(u["onset"] - 0.5) < 0.11
        assert abs(len(u["audio"]) / sr - 1.5) < 0.11
        assert abs(u["tail_s"] - room.SILENCE_TAIL_S) < 1e-6
        assert u["peak_rms"] > 0.05
        assert not gate.open

    def test_the_gate_never_sinks_below_the_floor(self):
        gate = room.MicGate(16000)
        for _ in range(200):
            gate.feed(np.zeros(1600, dtype=np.float32), 0)
        assert gate.gate == room.GALLERY["GATE_FLOOR"]

    def test_a_loud_room_raises_the_gate_up_to_the_ceiling(self):
        sr = 16000
        gate = room.MicGate(sr)
        rng = np.random.default_rng(1)
        for i in range(400):                      # 40 s of a hall at rms 0.019
            gate.feed((0.019 * rng.standard_normal(1600)).astype(np.float32), i * 0.1)
        assert room.GALLERY["GATE_FLOOR"] < gate.gate <= room.GALLERY["GATE_CEIL"]

    def test_nothing_outlasts_the_maximum(self):
        sr = 16000
        gate = room.MicGate(sr)
        blk = tone(220, room.BLOCK_S, sr, amp=0.1)
        utts = self.feed_all(gate, [blk] * 320, sr)     # 32 s held
        assert len(utts) >= 1
        assert len(utts[0]["audio"]) / sr <= room.MAX_UTTERANCE_S + 0.11


class WhoIsAtTheMicrophone(unittest.TestCase):
    def test_each_mic_is_its_own_turn_and_turns_end_after_90s(self):
        lst = room.Listener(None, [1, 2], 48000, post=False)
        a, rest = lst.singer_at(1, 100.0)
        assert a.startswith("human_gallery1_") and rest is None
        lst.turn[1] = (a, 103.0)
        b, rest = lst.singer_at(1, 110.0)
        assert b == a and rest == 7.0
        c, _ = lst.singer_at(2, 110.0)
        assert c != a and c.startswith("human_gallery2_")
        lst.turn[1] = (a, 110.0)
        d, rest = lst.singer_at(1, 110.0 + 91)
        assert d != a and rest is None

    def test_payload_is_what_the_page_posts(self):
        lst = room.Listener(None, [1], 48000, room="g", post=False)
        u = {"audio": None, "onset": 10.0, "ended": 12.0, "tail_s": 0.5, "peak_rms": 0.06}
        m = {"hz": 220.0, "note": "A3", "cents": 0, "stability": "steady", "duration_s": 2.0,
             "voiced_fraction": 0.95, "std_hz": 1.2, "contour_hz": [220.0] * 8, "why_not": ""}
        body = lst.payload(1, u, m, now=12.7)
        assert body["source"] == "gallery"
        assert body["voice_id"].startswith("human_g1_")
        assert body["tail_s"] == pytest.approx(0.7)       # 0.5 waited + 0.2 measuring
        assert body["own_rest_s"] is None
        assert set(body) >= {"hz", "note", "cents", "stability", "duration_s",
                             "voiced_fraction", "std_hz", "contour_hz", "tail_s"}

    def test_a_flooding_microphone_is_held_back(self):
        lst = room.Listener(None, [1], 48000, post=False)
        lst.accepted[1] = [100.0 + i for i in range(20)]
        assert lst.flooding(1, 121.0)
        assert not lst.flooding(1, 200.0)


class TheSpeakers(unittest.TestCase):
    def test_each_voice_sums_into_its_own_column(self):
        mx = room.Mixer(9)
        mx.add(np.ones(100, dtype=np.float32) * 0.5, 1)
        mx.add(np.ones(50, dtype=np.float32) * 0.5, 9)
        out = mx.render(80)
        assert out.shape == (80, 9)
        assert np.all(out[:, 0] == 0.5) and np.all(out[:50, 8] == 0.5) and np.all(out[50:, 8] == 0)
        assert np.all(out[:, 1:8] == 0)
        assert mx.sounding() == 1                     # the short one is done
        rest = mx.render(80)
        assert np.all(rest[:20, 0] == 0.5) and np.all(rest[20:, 0] == 0)
        assert mx.sounding() == 0

    def test_two_breaths_on_one_speaker_overlap_and_clip_safely(self):
        mx = room.Mixer(2)
        mx.add(np.ones(10, dtype=np.float32) * 0.8, 2)
        mx.add(np.ones(10, dtype=np.float32) * 0.8, 2)
        assert np.all(mx.render(10)[:, 1] == 1.0)

    def test_channel_off_the_stream_is_refused(self):
        with pytest.raises(ValueError):
            room.Mixer(2).add(np.zeros(1, dtype=np.float32), 3)


class ThePlayerChooses(unittest.TestCase):
    def player(self):
        p = room.Player(room.Mixer(9), 48000, room.default_channels(), verbose=False)
        p.heard_since = 1000.0
        return p

    def test_sung_while_hearing_once_each_not_too_late(self):
        p = self.player()
        now = 1100.0
        evs = [
            {"ts": 999.0, "who": "model", "voice_id": "voice_00", "audio": "old.wav"},
            {"ts": 1050.0, "who": "model", "voice_id": "voice_01", "audio": "a.wav"},
            {"ts": 1060.0, "who": "model", "voice_id": "voice_02"},          # no wav
            {"ts": 1070.0, "who": "human", "voice_id": "h1", "hz": 220},
            {"ts": 700.0, "who": "model", "voice_id": "voice_03", "audio": "late.wav"},
        ]
        got = p.choose(evs, now)
        assert [e["audio"] for e in got] == ["a.wav"]
        assert p.choose(evs, now) == []                 # already heard

    def test_people_elsewhere_sound_only_if_given_a_channel(self):
        p = self.player()
        evs = [{"ts": 1050.0, "who": "human", "voice_id": "h1", "hz": 220, "name": "a voice"},
               {"ts": 1051.0, "who": "human", "voice_id": "h2", "hz": 220,
                "name": "a voice in the room"}]
        assert p.choose(evs, 1100.0) == []
        p.humans_channel = 9
        got = p.choose(evs, 1100.0)
        assert [e["voice_id"] for e in got] == ["h1"]   # the room hears itself directly

    def test_poll_advances_since(self):
        p = self.player()
        p._get = lambda path, timeout=20: (
            b'{"recent":[{"ts": 5.0, "who":"model"},{"ts": 7.5, "who":"model"}]}')
        assert len(p.poll()) == 2 and p.since == 7.5

    def _scheduled(self, p, batch):
        starts = []
        p._sound = lambda e, at: starts.append(at)
        real = room.threading.Thread
        room.threading.Thread = lambda target, args, daemon: type("T", (), {"start": lambda s: target(*args)})()
        try:
            p.schedule(batch)
        finally:
            room.threading.Thread = real
        return starts

    def test_gaps_kept_and_capped(self):
        p = self.player()
        p.hold_s = 0.0                                  # on arrival, the old way
        t0 = time.monotonic()
        starts = self._scheduled(p, [{"ts": 0.0}, {"ts": 2.0}, {"ts": 30.0}])
        assert len(starts) == 3
        assert starts[1] - starts[0] == pytest.approx(2.0, abs=0.01)
        assert starts[2] - starts[1] == pytest.approx(room.GAP_MAX_S, abs=0.01)
        assert starts[0] >= t0

    def test_held_breaths_keep_the_choirs_rhythm(self):
        """Breaths reach the portal 10–19 s after their stamp, each by a
        different amount. Held to `ts + HOLD_S`, the room sounds them exactly
        as far apart as the choir stamped them; one that is already later than
        the hold is sounded at once."""
        p = self.player()
        assert p.hold_s == room.HOLD_S == 30.0
        now = time.time()
        t0 = time.monotonic()
        starts = self._scheduled(p, [{"ts": now - 12}, {"ts": now - 25}, {"ts": now - 45}])
        assert starts[0] - t0 == pytest.approx(18.0, abs=0.05)
        assert starts[1] - t0 == pytest.approx(5.0, abs=0.05)
        assert starts[2] - t0 == pytest.approx(0.0, abs=0.05)
        assert starts[0] - starts[1] == pytest.approx(13.0, abs=0.05)
