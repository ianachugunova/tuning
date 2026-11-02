"""Tests for the recording an ear is handed.

    python3 test_hearing.py

The claims here are audible ones, checked as arithmetic. That a person built
from numbers actually has the pitch those numbers named. That a glide glides
rather than buzzing, which is the difference between integrating a changing
frequency and multiplying by it — a mistake that sounds like a broken radio and
looks like one character. And that two voices at once come out as two, because
counting is the one thing the best ear in the probe could do and it can only do
it if the recording preserves the overlap.

Numpy only, like `hearing.py`.
"""

import unittest

import numpy as np

import hearing as H
from score import Note


def tone(t, dur, hz, who="human", audio="", contour=None, hop=0.0):
    return Note(t=t, kind="tone", who=who, sound_s=dur, pitch_hz=hz,
                audio=audio, contour_hz=contour or [], contour_hop_s=hop)


def dominant_hz(x, sr=H.SR):
    """The strongest frequency in a signal, to a bin's accuracy."""
    if x.size < 64:
        return 0.0
    win = x * np.hanning(x.size)
    spec = np.abs(np.fft.rfft(win))
    return float(np.fft.rfftfreq(x.size, 1.0 / sr)[int(np.argmax(spec))])


class SoundingAPerson(unittest.TestCase):
    """No recording of a voice exists, so the numbers have to be enough."""

    def test_the_tone_is_at_the_pitch_that_was_named(self):
        x = H.human_tone(440.0, 2.0)
        self.assertAlmostEqual(dominant_hz(x), 440.0, delta=2.0)
        self.assertEqual(x.size, int(2.0 * H.SR))

    def test_a_breath_with_no_pitch_makes_no_sound(self):
        """A person can be present without having sung a pitch, and silence is
        the honest rendering of that — not a hum at zero hertz."""
        self.assertEqual(H.human_tone(0.0, 2.0).size, 0)
        self.assertEqual(H.human_tone(-5.0, 2.0).size, 0)

    def test_a_glide_arrives_where_it_was_going(self):
        """The whole reason a contour is sent. The first half of a rising breath
        must be lower than the second, or the trace was thrown away."""
        x = H.human_tone(220.0, 3.0, contour=[220.0, 247.0, 277.0, 311.0])
        half = x.size // 2
        self.assertLess(dominant_hz(x[:half]), dominant_hz(x[half:]) - 20)

    def test_a_glide_is_continuous_rather_than_stepped(self):
        """`sin(2π f t)` with a changing f jumps phase on every sample and
        buzzes; the frequency has to be integrated. A step shows up as samples
        that leap, so the largest jump is the test."""
        x = H.human_tone(220.0, 2.0, contour=[220.0, 440.0])
        # At 440 Hz and 24 kHz the honest maximum step between samples is about
        # 2π·440/24000 ≈ 0.115 in radians, so ~0.12 in amplitude. Anything near
        # a full swing is a phase discontinuity.
        self.assertLess(float(np.abs(np.diff(x)).max()), 0.2)

    def test_a_short_contour_holds_rather_than_stretches(self):
        """A trace covering one second of a four-second breath describes the
        first second. Stretching it would invent three seconds of glide the
        singer never sang."""
        x = H.human_tone(220.0, 4.0, contour=[220.0, 330.0], hop_s=0.5)
        tail = x[int(2.0 * H.SR):]
        self.assertAlmostEqual(dominant_hz(tail), 330.0, delta=5.0)

    def test_both_ends_are_eased(self):
        """A square edge is the loudest thing in the field, and an encoder fed a
        click hears a click."""
        x = H.human_tone(300.0, 1.0)
        self.assertLess(abs(float(x[0])), 0.02)
        self.assertLess(abs(float(x[-1])), 0.02)

    def test_an_absurd_duration_cannot_take_the_window_hostage(self):
        """The number came off somebody's phone and is not to be trusted."""
        self.assertLessEqual(H.human_tone(300.0, 9999.0).size,
                             int(H.MAX_TONE_S * H.SR))
        self.assertGreater(H.human_tone(300.0, 0.0001).size, 0)


class KeepingTime(unittest.TestCase):
    """A paragraph is a list. A field is a room."""

    def test_the_window_is_as_long_as_it_says(self):
        x = H.field_audio([], window_s=10.0, end_ts=1000.0)
        self.assertEqual(x.size, 10 * H.SR)

    def test_a_breath_lands_where_its_onset_puts_it(self):
        """Onsets, not arrival order. The score already settled this and the
        recording has to agree with the score."""
        x = H.field_audio([tone(t=996.0, dur=1.0, hz=440.0)],
                          window_s=10.0, end_ts=1000.0)
        loud = np.flatnonzero(np.abs(x) > 0.01)
        self.assertAlmostEqual(loud[0] / H.SR, 6.0, delta=0.2)
        self.assertAlmostEqual(loud[-1] / H.SR, 7.0, delta=0.2)

    def test_silence_between_breaths_survives(self):
        """The rests are the piece. An ear handed the breaths with the gaps
        squeezed out is answering about a different room."""
        x = H.field_audio(
            [tone(t=991.0, dur=1.0, hz=300.0), tone(t=998.0, dur=1.0, hz=300.0)],
            window_s=10.0, end_ts=1000.0)
        middle = x[int(3.0 * H.SR):int(7.0 * H.SR)]
        self.assertLess(float(np.abs(middle).max()), 0.01)

    def test_two_at_once_come_out_as_two(self):
        """Counting was the one thing the best ear could do, and it can only do
        it if the overlap is still there when it listens."""
        x = H.field_audio(
            [tone(t=995.0, dur=3.0, hz=293.66), tone(t=995.0, dur=3.0, hz=440.0)],
            window_s=10.0, end_ts=1000.0)
        both = x[int(5.2 * H.SR):int(7.8 * H.SR)]
        win = both * np.hanning(both.size)
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(both.size, 1.0 / H.SR)
        for want in (293.66, 440.0):
            near = spec[(freqs > want - 8) & (freqs < want + 8)]
            self.assertGreater(float(near.max()), 0.05 * float(spec.max()),
                               f"{want} Hz is not in the mix")

    def test_a_voice_already_sounding_is_heard_from_the_door(self):
        """Somebody holding a note when you walk in is somebody you can hear.
        Dropping them would tell the ear the room was empty."""
        x = H.field_audio([tone(t=985.0, dur=10.0, hz=300.0)],
                          window_s=10.0, end_ts=1000.0)
        self.assertGreater(float(np.abs(x[:H.SR]).max()), 0.05)

    def test_nothing_leaks_past_the_end_of_the_window(self):
        x = H.field_audio([tone(t=999.5, dur=20.0, hz=300.0)],
                          window_s=10.0, end_ts=1000.0)
        self.assertEqual(x.size, 10 * H.SR)

    def test_a_crowd_is_turned_down_rather_than_clipped(self):
        """Clipping is heard by an encoder as noise, and it would be told there
        was noise in the room."""
        many = [tone(t=995.0, dur=3.0, hz=200.0 + 40 * i) for i in range(12)]
        x = H.field_audio(many, window_s=10.0, end_ts=1000.0)
        self.assertLessEqual(float(np.abs(x).max()), 1.0)

    def test_deaths_and_births_are_not_sounded(self):
        x = H.field_audio([Note(t=995.0, kind="death", who="machine"),
                           Note(t=996.0, kind="birth", who="system")],
                          window_s=10.0, end_ts=1000.0)
        self.assertEqual(float(np.abs(x).max()), 0.0)


class MachinesAndPeopleTogether(unittest.TestCase):
    """One recording out of a volume of wavs and a log of numbers."""

    def test_a_machine_is_played_back_and_not_re_synthesised(self):
        """The wav was rendered once and is what the room actually heard."""
        kept = np.full(H.SR, 0.5, dtype="float32")
        n = tone(t=995.0, dur=1.0, hz=440.0, who="machine", audio="a.wav")
        got = H.tone_of(n, load_wav=lambda name: kept)
        np.testing.assert_array_equal(got, kept)

    def test_a_lost_wav_falls_back_to_the_numbers(self):
        """A month unattended: a missing file is a voice heard differently, not
        a round that dies."""
        def missing(_name):
            raise FileNotFoundError("gone")

        n = tone(t=995.0, dur=1.0, hz=440.0, who="machine", audio="a.wav")
        got = H.tone_of(n, load_wav=missing)
        self.assertAlmostEqual(dominant_hz(got), 440.0, delta=3.0)

    def test_a_person_is_never_looked_for_on_the_volume(self):
        """There is no file to find. If this ever calls the loader, somebody has
        started keeping recordings of people."""
        def forbidden(_name):
            raise AssertionError("a person has no recording, by design")

        got = H.tone_of(tone(t=995.0, dur=1.0, hz=440.0), load_wav=forbidden)
        self.assertAlmostEqual(dominant_hz(got), 440.0, delta=3.0)

    def test_both_kinds_land_in_one_recording(self):
        kept = (0.4 * np.sin(2 * np.pi * 660.0
                             * np.arange(H.SR) / H.SR)).astype("float32")
        x = H.field_audio(
            [tone(t=995.0, dur=1.0, hz=660.0, who="machine", audio="a.wav"),
             tone(t=995.0, dur=1.0, hz=220.0)],
            window_s=10.0, end_ts=1000.0, load_wav=lambda name: kept)
        seg = x[int(5.1 * H.SR):int(5.9 * H.SR)]
        win = seg * np.hanning(seg.size)
        spec = np.abs(np.fft.rfft(win))
        freqs = np.fft.rfftfreq(seg.size, 1.0 / H.SR)
        for want in (220.0, 660.0):
            near = spec[(freqs > want - 10) & (freqs < want + 10)]
            self.assertGreater(float(near.max()), 0.05 * float(spec.max()),
                               f"{want} Hz is missing from the mix")


if __name__ == "__main__":
    unittest.main(verbosity=2)
