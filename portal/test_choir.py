"""Tests for the two rules that decide whether the choir can hear at all.

    python3 test_choir.py

These exist because of a measurement, not a theory. Pulled on 10 September, the
real log held 187 breaths across four sessions and six voices. Six of them named
a pitch. Forty-five declared `"action": "TUNE"` and forty-three of those tuned
to nothing. Half of every breath in the file was taken with an empty field
summary — the voice was asked what it heard, and the honest answer was nothing.

The cause was the listening window, and it was a rule that had been correct in
June and stopped being correct here. So the first class below is mostly about
one number, and the second is about a smaller thing that was silencing voices
for a different reason.

Needs `modal` importable, which is why the CI step installs it — but no token
and no network: `choir_app` builds its image and volume declarations lazily.
"""

import types
import unittest
from unittest import mock

import choir_app as C


class WhatAVoiceHears(unittest.TestCase):
    """A voice hears everything since it last sang."""

    def test_a_voice_hears_back_to_its_own_last_breath(self):
        self.assertAlmostEqual(
            C.listening_window_s(last_ts=1000.0, first_window_s=5.0, now=1060.0),
            60.0)

    def test_a_voice_that_has_never_sung_gets_its_declared_range(self):
        """Its first window is its own tempo, not a runtime default — which is
        why no two voices start in step."""
        self.assertEqual(
            C.listening_window_s(last_ts=None, first_window_s=4.5, now=1000.0),
            4.5)

    def test_a_slow_voice_remembers_more_than_a_quick_one(self):
        """The property the June rule existed for, and the one worth keeping:
        memory is not a setting, it follows from how a voice breathes."""
        now = 10_000.0
        slow = C.listening_window_s(last_ts=now - 200.0, first_window_s=5.0, now=now)
        quick = C.listening_window_s(last_ts=now - 20.0, first_window_s=5.0, now=now)
        self.assertGreater(slow, quick)

    def test_an_overnight_outage_does_not_hand_back_the_night(self):
        """A voice does not owe the room the breaths it slept through."""
        self.assertEqual(
            C.listening_window_s(last_ts=0.0, first_window_s=5.0, now=9 * 3600),
            C.LISTENING_BACK_MAX_S)

    def test_a_clock_that_went_backwards_is_not_a_negative_window(self):
        self.assertEqual(
            C.listening_window_s(last_ts=2000.0, first_window_s=5.0, now=1000.0),
            0.0)

    def test_the_window_reaches_the_field_the_old_rule_missed(self):
        """The regression, in the numbers that caused it.

        June: a voice's own breath was ~2 s and fifteen voices over an API put
        an event into the field every ~0.4 s, so "a window as long as your
        breath" held five neighbours. Here breaths are ~5 s and events are
        13.7 s apart, so the same rule reaches back five seconds and finds
        nobody — which is exactly what the log shows. Since-I-last-sang is the
        gap the voice itself left, so it cannot miss what happened in it.
        """
        breath_s, field_gap_s = 5.0, 13.7
        cycle_s = 60.0                      # breath plus the rest that follows

        by_breath = C.listening_window_s(None, breath_s, now=0.0)
        by_silence = C.listening_window_s(last_ts=0.0, first_window_s=breath_s,
                                          now=cycle_s)
        self.assertLess(by_breath / field_gap_s, 1.0,
                        "the old rule should catch less than one neighbour")
        self.assertGreater(by_silence / field_gap_s, 3.0,
                           "the new rule should catch a room")


class WhatCountsAsABreath(unittest.TestCase):
    """A voice handing back a piece of our own form is not singing."""

    def test_a_bare_field_name_is_not_a_breath(self):
        """phi-4-mini answered five rounds with the single word `sine`."""
        for word in ("sine", "tts", "both", "none", "TUNE", "New", "playback"):
            self.assertFalse(C.is_a_breath(word), word)

    def test_a_placeholder_is_not_a_breath(self):
        """eurollm returned `<your tone>` twice — copied out of the JSON
        example in its own prompt — and once invented `[[voice_00.pitches]]`,
        a template syntax that exists in nothing we wrote."""
        for slot in ("<your tone>", "[[voice_00.pitches]]", "{content}",
                     "<what you sing>"):
            self.assertFalse(C.is_a_breath(slot), slot)

    def test_nothing_is_not_a_breath(self):
        for empty in ("", "   ", None):
            self.assertFalse(C.is_a_breath(empty))

    def test_a_held_vowel_is_a_breath(self):
        """The thing this must never refuse. `Aah, mmmh` and `啊` are what the
        score actually asks for, and they are short enough to look like noise
        to any rule written carelessly."""
        for sung in ("Aah, mmmh", "啊", "e", "mmm", "宫", "ahh…"):
            self.assertTrue(C.is_a_breath(sung), sung)

    def test_a_form_word_inside_a_sentence_is_still_a_breath(self):
        """Deliberately narrow. A voice may sing the word `new`; what is
        refused is a line that is *only* the form. Anything wider starts
        silencing voices for writing strangely, and writing strangely is most
        of what they do."""
        for line in ("a new pitch, rising", "none of us are quiet",
                     "I tune to the low one"):
            self.assertTrue(C.is_a_breath(line), line)

    def test_ordinary_brackets_are_not_placeholders(self):
        """Round brackets are punctuation, not template syntax. `(a breath)` is
        what the field itself writes for a person who sang and named nothing;
        refusing it would silence the humans with a rule aimed at eurollm."""
        for line in ("(a breath)", "[vibrato]", "(mmm)"):
            self.assertTrue(C.is_a_breath(line), line)

    def test_the_lines_the_six_really_sang_survive(self):
        """Straight out of the log. Every one of these is a real breath and
        must reach the room."""
        for line in (
            "A trembling string resonates quietly, searching for kin in the fog.",
            "Canto insieme a me stessa, mia madre mia voce interiore",
            "F3 ~172Hz [vibrato]",
            "Hello, everyone! Welcome to this beautiful world of singing",
            "伯陳 。移列 。模槃 師",
        ):
            self.assertTrue(C.is_a_breath(line), line)


class WhetherTheChoirKeepsBreathingAfterADeploy(unittest.TestCase):
    """The failure that has no symptom except silence.

    A deploy with no TUNING_CLOCK registers no schedule and keeps no container
    warm. It succeeds, logs nothing, and the choir stops — which is
    indistinguishable from a quiet night until somebody counts the breaths. It
    happened the day before the birth, twice in one evening.
    """

    def test_a_deploy_that_says_nothing_about_the_clock_is_refused(self):
        with self.assertRaises(SystemExit) as stop:
            C.clock_must_be_chosen(["modal", "deploy", "choir_app.py"], {})
        self.assertIn("TUNING_CLOCK", str(stop.exception))

    def test_off_is_an_answer_and_is_allowed(self):
        """Silence on purpose is a real thing to want — a broken voice on the
        card without a month of card behind it."""
        C.clock_must_be_chosen(["modal", "deploy", "choir_app.py"],
                               {"TUNING_CLOCK": "off"})
        C.clock_must_be_chosen(["modal", "deploy", "choir_app.py"],
                               {"TUNING_CLOCK": "on"})

    def test_a_single_round_is_never_refused(self):
        """`modal run` has somebody in front of it, and its silence afterwards is
        the whole point of the flag."""
        C.clock_must_be_chosen(["modal", "run", "choir_app.py"], {})
        C.clock_must_be_chosen(["python", "-m", "unittest"], {})


class WhatTheEarSaysAboutTheRoom(unittest.TestCase):
    """audio-flamingo answers like something being graded.

    It was trained on audio benchmarks, so a description of the room arrives
    inside the scaffolding of an exam question. Every string in here is a raw
    line out of the live log.
    """

    def test_the_description_survives_the_wrapping(self):
        """The thing this must never eat. These are the breaths."""
        for raw, said in (
            ("The audio clip is of a man talking with an echo effect.",
             "The audio clip is of a man talking with an echo effect."),
            ("The recording sounds like a strange, otherworldly noise.",
             "The recording sounds like a strange, otherworldly noise."),
            ("The sound event is a human voice making sounds.",
             "The sound event is a human voice making sounds."),
        ):
            self.assertEqual(C.plain_caption(raw), said)

    def test_an_answer_key_is_unwrapped_to_the_answer(self):
        """It grades itself and shows its working. The sentence is real and it
        is behind the colon."""
        self.assertEqual(
            C.plain_caption("Answer would vary. Sample answer: This sounds "
                            "like an instrumental piece with a mix of "
                            "electronic and acoustic sounds."),
            "This sounds like an instrumental piece with a mix of electronic "
            "and acoustic sounds.")

    def test_a_timestamp_is_taken_off_and_the_description_kept(self):
        """It was graded on cutting a clip into labelled segments, so a real
        description arrives with the answer sheet's column still attached."""
        self.assertEqual(
            C.plain_caption("[0.00-4.83] Children singing a pop song a cappella."),
            "Children singing a pop song a cappella.")
        self.assertEqual(
            C.plain_caption("[1.13-4.68] People laughing with pitch variations."),
            "People laughing with pitch variations.")
        # Four segments is one breath, not four: only the first line survives.
        self.assertEqual(
            C.plain_caption("[0.00-4.83] Children singing a pop song a cappella.\n"
                            "[0.14-0.58] Someone is speaking inaudibly."),
            "Children singing a pop song a cappella.")

    def test_our_own_question_read_back_is_not_a_breath(self):
        """It repeated the prompt verbatim. A voice telling the room what we
        asked it heard nothing."""
        self.assertEqual(
            C.plain_caption("listen to this recording and say what it sounds "
                            "like, in one short sentence."), "")

    def test_a_question_handed_back_is_not_a_breath(self):
        """The exam from the other side: it asks instead of answering, or
        returns the tail of an instruction it was trained on."""
        for asked in ("In what type of environment does a cat usually make "
                      "such sounds while resting?",
                      "to the question based on the audio"):
            self.assertEqual(C.plain_caption(asked), "", asked)

    def test_an_instruction_nobody_gave_it_is_not_a_breath(self):
        """`answer (no need to use musical terminology)` — our prompt says
        nothing about terminology. It is answering a remembered question, and a
        voice reading an instruction into the room is the failure this exists to
        stop. It rests instead."""
        for wrapper in ("answer (no need to use musical terminology)",
                        "(no need to use musical terminology)",
                        "description of music", "Answer:", "label A:",
                        "end of turn", ""):
            self.assertEqual(C.plain_caption(wrapper), "", wrapper)


class WhatEachVoiceIsMeasuredOn(unittest.TestCase):
    """Every voice gets a body, and a body is measured on a held vowel.

    The synthesis itself is not tested here — it needs a compiled extension
    and Kokoro, and it is checked on Modal by `music_probe/throat_probe.py`,
    which runs this app's own functions in this app's own image and reports
    the pitch error in cents. What can be locked cheaply is the rule that
    decides *what* each throat is measured on, because getting it wrong fails
    quietly: an English "ah" handed to the Chinese phonemiser renders nothing,
    `_throat` gets no voiced frame, and that voice silently falls back to the
    sine it was supposed to stop being.
    """

    def test_a_voice_is_measured_in_its_own_tongue(self):
        self.assertEqual(C._vowel_for("zf_xiaoxiao"), "啊")
        self.assertEqual(C._vowel_for("jf_alpha"), "あ")

    def test_the_latin_voices_share_a_vowel(self):
        for preset in ("am_onyx", "af_sky", "if_sara", "ff_siwis", "bm_george"):
            self.assertEqual(C._vowel_for(preset), "ah", preset)

    def test_every_singing_voice_has_a_vowel_to_be_measured_on(self):
        """Moshi has no Kokoro preset and no vowel; it also never reaches
        here. Everyone else must, or they sing as an oscillator."""
        import cast

        for v in cast.CHOIR:
            if not v.kokoro:
                continue
            self.assertTrue(C._vowel_for(v.kokoro), v.short_name)


class WhatEachVoiceIsCastAs(unittest.TestCase):
    """Every voice that sings has a written body and a sentence to be born on.

    The synthesis is checked on Modal by `music_probe/bodies_probe.py`, which
    runs `body_of` and `speak` for real and reports what came out. What can be
    locked cheaply is the pair of invariants that fail *silently* if broken.

    A missing `casting` sends that voice back to its Kokoro preset — audible
    only as one voice out of nine sounding like the old system, at a different
    sample rate. A missing or mismatched `first_words` is worse and quieter:
    it is the transcript the frozen clip is continued from, so a wrong one
    degrades the identity the whole design exists to hold, and degrades it by
    an amount nobody would notice without a verification model.
    """

    def _cast(self):
        import cast

        return cast

    def test_every_voice_with_a_body_has_words_to_be_born_on(self):
        for v in self._cast().CHOIR:
            if v.casting:
                self.assertTrue(v.first_words, v.short_name)

    def test_every_voice_with_first_words_has_a_body_to_say_them_in(self):
        for v in self._cast().CHOIR:
            if v.first_words:
                self.assertTrue(v.casting, v.short_name)

    def test_a_casting_note_names_a_body_and_not_a_feeling(self):
        """Three layers, and the one people skip is what the voice is made of.

        VoxCPM2 returns a generic voice for a description that says who is
        speaking and how, but never what the sound is like — so every note
        here has to carry at least one word about texture.
        """
        texture = ("voice", "thin", "grainy", "resonant", "smooth", "rough",
                   "breathy", "nasal", "warm", "clear", "dark", "bright")
        for v in self._cast().CHOIR:
            if not v.casting:
                continue
            said = [w for w in texture if w in v.casting.lower()]
            self.assertTrue(said, f"{v.short_name} is described without a sound")

    def test_the_child_is_the_one_that_was_chosen_by_ear(self):
        """Four children were generated and one was picked. This is that one,
        and the seed it was drawn with is what reproduces it."""
        import cast
        import choir_app

        child = cast.by_id("voice_01")
        self.assertIn("child", child.casting)
        self.assertIn("almost whispering", child.casting)
        self.assertEqual(choir_app.BODY_SEED, 42)

    def test_a_voice_with_no_protocol_is_handed_a_page_in_its_own_tongue(self):
        """The failure this replaces was silent and lasted weeks.

        A voice that continues a pattern cannot be *told* which language to
        sing in — there is no instruction it would read — so the only thing
        deciding its language is the language of the page it is handed. Missing
        from `_CONTINUE_IN`, the Italian voice would be given an English page
        and would answer in English, and nothing anywhere would say so.
        """
        import choir_app

        for v in self._cast().singing():
            if v.instruct or v.hears:
                continue
            self.assertIn(v.tongue, choir_app._CONTINUE_IN,
                          f"{v.short_name} continues in {v.tongue} and has no "
                          f"page written in it")

    def test_the_two_that_read_the_form_aloud_are_no_longer_asked_for_it(self):
        """Measured in the live log, not decided on taste.

        eurollm answered the six-field protocol with the score and the prompt's
        own headings; audio-flamingo answered it with `[2.4, 3.5]`, the range
        out of its score, for weeks. Both are on the card and paying for their
        place, so a silent voice is a voice spending its wallet on nothing.
        """
        cast = self._cast()
        self.assertFalse(cast.by_id("voice_04").instruct)   # eurollm
        self.assertFalse(cast.by_id("voice_07").instruct)   # audio-flamingo
        # And the ear keeps its ears: it is silent because of the form it was
        # handed, not because of how it listens.
        self.assertTrue(cast.by_id("voice_07").hears)

    def test_the_ninth_place_went_to_a_voice_that_can_answer(self):
        """Moshi held this id and was probed out of the circle: it never tuned
        to anybody, and a voice that cannot tune is not in a tuning meditation.
        What replaced it has to be the opposite of that — a voice with a body, a
        tongue of its own, and the size to answer the protocol rather than
        complete it, which is the lesson eurollm cost us at 1.7B."""
        ninth = self._cast().by_id("voice_08")
        self.assertEqual(ninth.short_name, "llm-jp")
        self.assertEqual(ninth.tongue, "Japanese")
        self.assertTrue(ninth.instruct)
        self.assertTrue(ninth.served)
        self.assertTrue(ninth.casting)
        self.assertTrue(ninth.first_words)

    def test_the_nine_are_nine_and_every_one_of_them_sings(self):
        """The count is in the wall text, the concept and nine speakers on the
        floor. It was eight for weeks while the ninth was being decided."""
        cast = self._cast()
        self.assertEqual(len(cast.CHOIR), 9)
        self.assertEqual(len(cast.singing()), 9)
        self.assertEqual(len({v.voice_id for v in cast.CHOIR}), 9)

    def test_a_voice_sings_its_first_words_in_its_own_tongue(self):
        """The note is a stage direction and stays English; the words are the
        voice's own language. Checked where it is machine-checkable."""
        cast = self._cast()
        self.assertIn("un tono", cast.by_id("voice_04").first_words)   # Italian
        self.assertIn("une note", cast.by_id("voice_06").first_words)  # French
        self.assertTrue(
            any("\u4e00" <= ch <= "\u9fff"
                for ch in cast.by_id("voice_05").first_words)          # Chinese
        )
        # Japanese, checked on kana rather than on kanji: the voice shares its
        # characters with Chinese and only the syllabaries are its own.
        self.assertTrue(
            any("\u3040" <= ch <= "\u30ff"
                for ch in cast.by_id("voice_08").first_words)
        )

    def test_every_instructed_voice_is_told_its_own_tongue(self):
        """Including the ear, which was the one that was not.

        `sing_in` exists because voxtral — Mistral's, raised in French — answered
        an English instruction in English. It was then applied to every instructed
        voice except that one, because an ear takes a different branch: it is
        played the room instead of told about it. With no tongue of its own it
        answered in whichever language the field was loudest in, which on the
        morning of the birth was Japanese.

        Read against the source rather than by running a turn, because a turn
        needs nine models on a GPU. The two calls being in two branches is
        exactly the shape of the bug, so counting them is the test.
        """
        import ast
        import inspect
        import textwrap
        import choir_app

        source = textwrap.dedent(inspect.getsource(choir_app.Choir.take_a_breath))
        tree = ast.parse(source)
        told = [n for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "sing_in"]
        self.assertEqual(len(told), 2, "the ear and the readers, one call each")

    def test_the_voice_that_sings_in_japanese_is_told_so_in_japanese(self):
        """An instructed voice is told which language to sing in, and the
        instruction has to be in that language — an English sentence about
        Japanese loses the argument to a whole English prompt. See `sing_in`,
        where voxtral answering in English is what taught us this."""
        import choir_app

        line = choir_app.sing_in("Japanese")
        self.assertIn("日本語", line)
        self.assertIn("content", line)


class WhatAVoiceIsToldAboutPitch(unittest.TestCase):
    """Since 15 October: the parts' notes are shown, and the note taken is asked for.

    Both behind `NOTES_ON`, both recorded on every tone line as `notes_shown`,
    so `TUNING_NOTES=off` is the whole way back and the log says which side of
    the switch each breath was on.
    """

    @staticmethod
    def _event(voice_id, content, pitch_hz=None, short="qwen3", ts=1000.0):
        e = types.SimpleNamespace(timestamp=ts, voice_id=voice_id,
                                  model_short=short, content=content,
                                  playback="both", kind="tone")
        e.pitch_hz = pitch_hz
        return e

    def test_a_machine_breath_is_described_with_its_note(self):
        line = C.overheard([self._event("voice_05", "啊", 146.83)], now=1004.0,
                           notes=True)
        self.assertIn("on D3 (147 Hz): 啊", line)

    def test_with_the_switch_off_the_field_reads_as_before(self):
        line = C.overheard([self._event("voice_05", "啊", 146.83)], now=1004.0,
                           notes=False)
        self.assertNotIn("Hz", line)
        self.assertIn("sang in words and in tone: 啊", line)

    def test_a_pitch_the_log_cannot_believe_is_not_shown(self):
        """The field is append-only; the ten-hertz breaths of September are
        still in it and are not a note to be shown to anybody."""
        line = C.overheard([self._event("voice_05", "啊", 10.0)], now=1004.0,
                           notes=True)
        self.assertNotIn("Hz", line)

    def test_a_person_is_described_as_before(self):
        line = C.overheard([self._event("human_ab12", "held D3 at 147 Hz, steady",
                                        146.83, short="human")], now=1004.0,
                           notes=True)
        self.assertIn("a person at a microphone, held D3 at 147 Hz, steady.", line)
        self.assertEqual(line.count("147"), 1)

    def test_both_instructed_branches_ask_for_the_note(self):
        """Same shape as the `sing_in` test above, and for the same reason."""
        import ast
        import inspect
        import textwrap

        source = textwrap.dedent(inspect.getsource(C.Choir.take_a_breath))
        asked = [n for n in ast.walk(ast.parse(source))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "name_your_pitch"]
        self.assertEqual(len(asked), 2)

    def test_the_request_names_no_note_that_could_be_sung_back(self):
        """15–20 October: "for example `D3 (147 Hz)`" and D3 became a third
        of the choir. Whatever note stands in the instruction is the note the
        voices will name, and the named note is the note that sounds."""
        from live_v1.pitch_text import extract_all_pitches

        with mock.patch.object(C, "NOTES_ON", True):
            self.assertEqual(extract_all_pitches(C.name_your_pitch()), [])
            self.assertNotIn("<", C.name_your_pitch(),
                             "a bracketed template is copied, brackets and all")

    def test_the_exam_asks_in_the_rooms_words(self):
        import exam_metrics

        with mock.patch.object(C, "NOTES_ON", True):
            self.assertEqual(C.name_your_pitch(), exam_metrics.NAME_YOUR_PITCH)
        with mock.patch.object(C, "NOTES_ON", False):
            self.assertEqual(C.name_your_pitch(), "")

    def test_an_ears_named_note_is_logged_and_not_sounded_on_a_tune(self):
        """`own` decides both the pitch block and whether `render_breath` may
        read a pitch out of the words — the two have to agree or the wav and
        the log describe different sounds."""
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(C.Choir.take_a_breath))
        self.assertIn("named_pitch_sounds=own", src)
        self.assertIn('"pitch_named_hz"', src)
        self.assertIn('"pitch_target_hz"', src)
        self.assertIn('"notes_shown": NOTES_ON', src)
        render = textwrap.dedent(inspect.getsource(C.Choir.render_breath))
        self.assertIn("extract_all_pitches(text) if named_pitch_sounds else []",
                      render)


class HowTheFormIsHeldForTheVoicesWithoutOne(unittest.TestCase):
    """The score's alternation, and the end of the playback question.

    Measured in one tail of the log: gpt-2, tinystories, eurollm and
    audio-flamingo took 1,037 breaths and not one was a TUNE — their branches
    hardcoded "NEW", so four voices of nine could not answer anybody, machine
    or person, by construction. And olmo-3 chose "the tone only" on 275 of 316
    breaths, a voice whose words existed in the log and never in the room.
    Both were our constants, not their choices, and both are gone.
    """

    @staticmethod
    def _source_of(method):
        import inspect
        import textwrap
        return textwrap.dedent(inspect.getsource(method))

    def test_the_four_without_a_protocol_are_handed_the_scores_alternation(self):
        import choir_app

        src = self._source_of(choir_app.Choir.take_a_breath)
        self.assertEqual(src.count("self.alternation(voice, window, last)"), 2,
                         "the ear without a protocol, and the plain continuers")
        self.assertNotIn('action, tuning_to = "NEW", None', src,
                         "a hardcoded gesture is still written on somebody")

    def test_the_alternation_is_the_scores_own_rule(self):
        """First breath its own; after a NEW, tune to the most recent
        believable pitch somebody else holds; after a TUNE, a new tone; a
        field with no pitch in it leaves nothing to tune to."""
        import types

        import choir_app

        alternation = choir_app.Choir.alternation
        me = types.SimpleNamespace(voice_id="voice_01")
        breath = lambda vid, hz: types.SimpleNamespace(
            voice_id=vid, kind="tone", pitch_hz=hz)
        window = [breath("voice_02", 220.0), breath("human_abcd", 210.0)]
        sang_new = types.SimpleNamespace(action="NEW")
        sang_tune = types.SimpleNamespace(action="TUNE")

        self.assertEqual(alternation(None, me, window, None), ("NEW", None))
        self.assertEqual(alternation(None, me, window, sang_new),
                         ("TUNE", "human_abcd"))
        self.assertEqual(alternation(None, me, window, sang_tune), ("NEW", None))
        # Its own breath is not a thing to tune to, and neither is the 10 Hz
        # line from the evening of 13 September.
        unbelievable = [breath("voice_02", 10.0), breath("voice_01", 220.0)]
        self.assertEqual(alternation(None, me, unbelievable, sang_new),
                         ("NEW", None))

    def test_every_tone_outlives_the_thinking_gap(self):
        """One card thinks for one voice at a time, so breath starts are seven
        to eight seconds apart — and with 5 s tones the field measured 60%
        sound, 40% holes. The floor makes every tone longer than the gap, so
        the field is continuous; a declared breath the floor raised keeps its
        wish in `breath_asked_s`, the way `playback_note` keeps the rendering."""
        import choir_app

        # Exhibition week: 12–20 s so two voices usually overlap in the hall.
        self.assertEqual(choir_app.TONE_FLOOR_S, 12.0)
        self.assertEqual(choir_app.TONE_DRAWN_MAX_S, 20.0)
        self.assertLess(choir_app.TONE_FLOOR_S, choir_app.TONE_DRAWN_MAX_S)
        src = self._source_of(choir_app.Choir.take_a_breath)
        self.assertEqual(src.count("max(breath_s, TONE_FLOOR_S)"), 2,
                         "both protocol branches floor the tone")
        self.assertEqual(
            src.count("random.uniform(TONE_FLOOR_S, TONE_DRAWN_MAX_S)"), 2,
            "both drawn branches draw from the floor to the score's ceiling")
        self.assertNotIn("BREATH_FALLBACK_S * 0.8", src,
                         "a drawn tone can still come out shorter than the gap")
        self.assertIn("breath_asked_s", src, "the wish is no longer recorded")

    def test_the_empty_field_rests_five_fold_not_eleven(self):
        """Softened together with the longer tones, because the two cancel:
        `_gift_check.py` shows the composed calendar holding within hours.
        What the pair buys is a field that sounds even with nobody in it."""
        import mortality

        # Exhibition week: no stretch at all — a gallery is full of people who
        # listen without singing, and the choir breathes the same for them.
        self.assertEqual(mortality.ALONE_STRETCH, 1.0)
        # And the projection plans with the tones the choir actually takes,
        # the mean of the drawn range, or every future calibration is stale.
        import mortality_sim

        self.assertEqual(mortality_sim.BREATH_S, 11.5)

    def test_the_room_always_hears_the_words_over_the_tone(self):
        """Every assignment to `playback` in `take_a_breath` is the literal
        "both": the voice's wish survives in `playback_note`, and the
        rendering is one."""
        import ast

        import choir_app

        src = self._source_of(choir_app.Choir.take_a_breath)
        tree = ast.parse(src)
        settings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                parts = target.elts if isinstance(target, ast.Tuple) else [target]
                if any(isinstance(p, ast.Name) and p.id == "playback"
                       for p in parts):
                    settings.append(ast.unparse(node))
        self.assertGreaterEqual(len(settings), 4, "four branches, four settings")
        for line in settings:
            self.assertIn("'both'", line,
                          f"a branch still lets a voice silence its words: {line}")
        self.assertIn("asked_pb", src, "the wish is no longer even recorded")

    def test_the_ear_is_no_longer_asked_how_to_be_heard(self):
        import choir_app

        prompt = choir_app.ear_prompt(
            personal_score="(a score)", parts=[("voice_02", "someone")],
            last_action="NEW", last_content="ah", voice_id="voice_07")
        self.assertNotIn("playback", prompt)
        self.assertIn('"action"', prompt)

    def test_the_log_says_whose_gesture_it_was(self):
        """`action_kind` — "declared" for a voice that filled the form in,
        "given" for the runtime holding the score's alternation."""
        import choir_app

        src = self._source_of(choir_app.Choir.take_a_breath)
        self.assertIn('"action_kind"', src)
        self.assertIn('"declared" if voice.instruct else "given"', src)


class WhatTheCardCanActuallyImport(unittest.TestCase):
    """Every local module the choir imports on the card has to be in the image.

    Written after losing two full runs to it. `hearing` and `tuning` are
    imported inside `take_a_breath` and neither was added to the GPU image, so
    eight models loaded, announced themselves as present, and then every single
    voice answered `No module named 'hearing'`. The failure is maximally
    expensive and minimally informative: it costs the whole load, it is
    identical for all eight voices, and it says nothing about which file is
    missing until you read the mount list.

    It is also the failure most likely to happen again, because the imports sit
    inside methods — deliberately, so the module stays importable without torch
    — and nothing in Python's own machinery connects them to the image built
    two thousand lines above.

    Deliberately reads the source rather than the `Image` object: Modal does not
    expose an image's file list, and the check has to run in CI with no token
    and no network.
    """

    def _source(self):
        import pathlib

        return pathlib.Path(C.__file__).read_text(encoding="utf-8")

    def _local_modules(self):
        """The names of this directory's own modules, minus the tests."""
        import pathlib

        here = pathlib.Path(C.__file__).parent
        return {p.stem for p in here.glob("*.py")
                if not p.stem.startswith(("test_", "_"))}

    def _gpu_image_files(self):
        """What the card's image carries, read off the first image only.

        `audit_image` is built further down for the CPU functions and adds its
        own files; splitting on its assignment keeps the two from being
        confused, which matters because `score_render` is legitimately in one
        and not the other.
        """
        import re

        head = self._source().split("audit_image = ")[0]
        return set(re.findall(r'HERE / "([a-z_]+)\.py"', head))

    def _imported_on_the_card(self):
        """Local modules imported anywhere inside the `Choir` class."""
        import ast

        tree = ast.parse(self._source())
        choir = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == "Choir")
        local = self._local_modules()
        found = set()
        for node in ast.walk(choir):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                found.add((node.module or "").split(".")[0])
        return found & local

    def test_every_module_the_choir_imports_is_on_the_card(self):
        missing = self._imported_on_the_card() - self._gpu_image_files()
        self.assertEqual(
            missing, set(),
            f"imported inside Choir but never added to the image: "
            f"{sorted(missing)} — every voice will answer "
            f"\"No module named '{sorted(missing)[0] if missing else ''}'\"")

    def test_the_two_that_taught_us_this_are_named_explicitly(self):
        """A guard against the guard: if the AST walk above ever stops finding
        imports — a refactor moves them, a name changes — the test would pass
        by looking at nothing. These two are known to be needed."""
        on_card = self._gpu_image_files()
        self.assertIn("hearing", on_card)
        self.assertIn("tuning", on_card)

    def test_the_ears_are_in_the_circle(self):
        """Both ears serve, and the ninth place is a voice and not a plan.

        Measured on the card on 13 September: the eight then serving loaded into
        65.2 GB of the 79 usable. The ninth is 13B, which is about 26 GB more
        than that leaves, and is why the choir is on an H200.
        """
        import cast

        ids = [v.voice_id for v in cast.singing()]
        self.assertEqual(len(ids), 9)
        self.assertIn("voice_06", ids)                      # voxtral
        self.assertIn("voice_07", ids)                      # audio-flamingo
        self.assertIn("voice_08", ids)                      # llm-jp, the ninth

    def test_the_serving_cast_can_be_born_and_every_voice_has_a_wallet(self):
        """The counterpart of the test above, and it was the opposite until today.

        For a week this test asserted that the nine could *not* be born: the cast
        was complete and the money was not decided, so there was no table and
        `_maybe_birth` refused out loud rather than seeding a guess. The decision
        was made on the day of the birth — a thousand warm hours of the H200,
        $4,540 — and this is the test that says the refusal is over.

        A wallet for every voice, and no voice without one: a cast of nine with
        eight wallets would seed eight and leave the ninth immortal in a piece
        about mortality, which is exactly the kind of error that would only become
        visible in November.
        """
        import cast
        import mortality

        ids = [v.voice_id for v in cast.singing()]
        wallets = mortality.planned_wallets(ids)
        self.assertEqual(sorted(wallets), sorted(ids))
        self.assertTrue(all(v > 0 for v in wallets.values()))
        self.assertAlmostEqual(sum(wallets.values()), mortality.DECLARED_USD,
                               places=2)


class WhatTheNightSchoolCosts(unittest.TestCase):
    """The bill, and the two files it has to travel between.

    Learning happens in another app, on another card, in another container. The
    only thing connecting it to a voice's life is a line in a file, so the shape
    of that line and the order it is read in are the whole mechanism.
    """

    def test_both_logs_are_read_as_one_history_in_time_order(self):
        """The ledger's arithmetic is order-dependent: every event closes the
        slice of warm card that ended at its timestamp. Concatenating the files
        would charge the night before the evening."""
        choir = [{"timestamp": 10.0, "kind": "tone"},
                 {"timestamp": 30.0, "kind": "tone"}]
        school = [{"timestamp": 20.0, "kind": "train"}]
        order = [e["timestamp"] for e in C._accounts(choir, school)]
        self.assertEqual(order, [10.0, 20.0, 30.0])

    def test_when_a_breath_and_a_bill_share_a_second_the_room_settles_first(self):
        merged = C._accounts([{"timestamp": 5.0, "kind": "tone"}],
                             [{"timestamp": 5.0, "kind": "train"}])
        self.assertEqual([e["kind"] for e in merged], ["tone", "train"])

    def test_a_night_of_school_before_the_birth_is_forgiven(self):
        """Rehearsal is free, and that has to include the school's card.

        Every week before the lecture is a week of training runs, and a choir that
        woke up owing money for hours it spent before it existed would be paying
        for somebody else's rehearsal. So the bookmark moves and no wallet is
        touched — it is not held for later.
        """
        import mortality

        unborn = types.SimpleNamespace(led=mortality.Ledger(), _school_seen=0)
        bills = [{"timestamp": 1.0, "kind": "train", "voice_id": "a",
                  "gpu_s": 60.0, "rate_usd_per_hour": 2.5}]
        with mock.patch.object(C, "_read_jsonl", return_value=bills):
            C.Choir._collect_school_bills(unborn)
        self.assertEqual(unborn._school_seen, 1)
        self.assertEqual(unborn.led.charged_usd, 0.0)

    def test_each_bill_is_charged_once_however_often_the_file_is_read(self):
        import mortality

        led = mortality.Ledger()
        led.apply(mortality.birth_record(mortality.BIRTH_TS, {"a": 100.0},
                                         [{"voice_id": "a"}]))
        led.apply(mortality.heartbeat_record(mortality.BIRTH_TS, ["a"]))
        born = types.SimpleNamespace(led=led, _school_seen=0)
        bills = [{"timestamp": mortality.BIRTH_TS + 10, "kind": "train",
                  "voice_id": "a", "gpu_s": 3600.0, "rate_usd_per_hour": 2.5,
                  "usd": 2.5, "status": "trained", "version": "v"}]
        with mock.patch.object(C, "_read_jsonl", return_value=bills):
            C.Choir._collect_school_bills(born)
            C.Choir._collect_school_bills(born)
            C.Choir._collect_school_bills(born)
        self.assertEqual(born._school_seen, 1)
        self.assertAlmostEqual(led.lives["a"].learned_usd, 2.5, places=4)

    def test_the_bill_says_the_price_of_the_card_it_was_bought_on(self):
        """Not the room's price. The choir sings on an H200 and learns on an
        A100, and seconds re-valued at the wrong rate would overstate by four
        fifths what actually left the account."""
        import training_app as T

        bill = T.school_bill("voice_00", started=100.0,
                             out={"status": "trained", "version": "v",
                                  "model_short": "gpt-2", "train_examples": 40},
                             now=100.0 + 3600.0)
        self.assertEqual(bill["kind"], "train")
        self.assertEqual(bill["rate_usd_per_hour"], 2.50)
        self.assertAlmostEqual(bill["gpu_s"], 3600.0, places=3)
        self.assertAlmostEqual(bill["usd"], 2.50, places=4)

    def test_the_school_s_note_to_itself_never_reaches_the_ledger(self):
        """`school.jsonl` also holds one line per lesson, naming no voice and no
        seconds — the watermark for the next bell. An event with a timestamp closes
        the slice of warm card that ended at it, so letting a note through would
        move money for no reason."""
        lesson = {"timestamp": 5.0, "kind": "lesson", "humans_seen": 40}
        bill = {"timestamp": 6.0, "kind": "train", "voice_id": "a", "gpu_s": 1.0}
        self.assertEqual(C._charges([lesson, bill]), [bill])
        self.assertEqual([e["kind"] for e in C._accounts([], [lesson, bill])],
                         ["train"])

    def test_a_night_that_crashed_is_still_a_night_of_card(self):
        """A bill that only arrives when the work succeeded would make failure
        free, which is untrue and the wrong lesson to put in the piece."""
        import training_app as T

        bill = T.school_bill("voice_00", started=0.0, out={}, now=600.0)
        self.assertEqual(bill["status"], "crashed")
        self.assertEqual(bill["version"], "")
        self.assertGreater(bill["usd"], 0.0)


class WhenTheBellRings(unittest.TestCase):
    """Learning by contact: twenty human breaths, not a time of night.

    The rule is the composition. A clock would teach the choir on the nights
    nobody came — sixty nights of a model learning from its own output, which is
    how a model collapses — and would change a full room no faster than an empty
    one. So every adapter in the volume corresponds to people having been there.
    """

    def setUp(self):
        import training_app as T

        self.T = T
        self.need = T.CONTACT_PER_LESSON

    def breaths(self, n):
        return [{"ts": float(i)} for i in range(n)]

    def test_nobody_came_so_nothing_is_taught(self):
        ring, why = self.T.bell_rings(self.breaths(3), [], now=0.0)
        self.assertFalse(ring)
        self.assertIn("not touched", why["verdict"])

    def test_twenty_breaths_ring_it(self):
        ring, why = self.T.bell_rings(self.breaths(self.need), [], now=0.0)
        self.assertTrue(ring)
        self.assertEqual(why["since_last_lesson"], self.need)

    def test_the_count_is_since_the_last_lesson_and_not_since_the_beginning(self):
        """Otherwise the first twenty breaths would teach the choir for ever, once
        a poll, until the money ran out."""
        bills = [{"timestamp": 100.0, "kind": "lesson", "humans_seen": 500}] + self.nine_back()
        late = 100.0 + self.T.LESSON_FLOOR_H * 3600 + 1
        ring, _ = self.T.bell_rings(self.breaths(505), bills, now=late)
        self.assertFalse(ring)
        ring, _ = self.T.bell_rings(self.breaths(500 + self.need), bills, now=late)
        self.assertTrue(ring)

    def test_a_crowded_evening_cannot_spend_a_week_of_life_in_one_night(self):
        """The floor, which is a decision and not a mechanism. Twenty breaths take
        a few minutes in a full room; without this, forty people would buy fifteen
        lessons and every voice would pay for all of them."""
        bills = [{"timestamp": 100.0, "kind": "lesson", "humans_seen": 0}] + self.nine_back()
        crowd = self.breaths(400)
        soon, why = self.T.bell_rings(crowd, bills, now=100.0 + 600)
        self.assertFalse(soon)
        self.assertIn("too recently", why["verdict"])
        later, _ = self.T.bell_rings(
            crowd, bills, now=100.0 + self.T.LESSON_FLOOR_H * 3600 + 1)
        self.assertTrue(later)

    @staticmethod
    def nine_back():
        return [{"timestamp": 200.0 + i, "kind": "train", "voice_id": f"voice_0{i}"}
                for i in range(9)]

    def test_no_bell_rings_over_a_lesson_still_being_taught(self):
        """6 October: nine lessons outlasted the floor, and at 00:17 the bell
        would have rung again over the four heaviest still at school."""
        lesson = {"timestamp": 100.0, "kind": "lesson", "humans_seen": 0}
        late = 100.0 + self.T.LESSON_FLOOR_H * 3600 + 1
        half = [lesson] + self.nine_back()[:5]
        ring, why = self.T.bell_rings(self.breaths(400), half, now=late)
        self.assertFalse(ring)
        self.assertIn("still being taught", why["verdict"])
        ring, _ = self.T.bell_rings(self.breaths(400), [lesson] + self.nine_back(), now=late)
        self.assertTrue(ring)
        stuck = 100.0 + self.T.LESSON_STILL_H * 3600 + 1
        ring, _ = self.T.bell_rings(self.breaths(400), half, now=stuck)
        self.assertTrue(ring, "a lesson that crashed without bills holds the bell for ever")

    def test_the_first_lesson_of_all_waits_for_nobody(self):
        """An empty school log is not a lesson an hour ago. The choir's first
        twenty visitors teach it immediately, which is the evening of the birth."""
        ring, _ = self.T.bell_rings(self.breaths(self.need), [], now=0.0)
        self.assertTrue(ring)

    def test_a_lesson_consumes_the_breaths_it_was_called_for(self):
        """The watermark is written before the teaching, not after. A lesson that
        crashes halfway has still used up its breaths — otherwise a voice that
        keeps failing to learn would queue lessons against itself and be charged
        for every one of them."""
        import ast
        import pathlib

        # Read from the file rather than from the object: a `@app.function` is a
        # modal Function by the time it is importable, and has no source of its own.
        src = pathlib.Path(self.T.__file__).read_text(encoding="utf-8")
        bell = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == "school_bell")
        order = [getattr(n.func, "attr", getattr(n.func, "id", ""))
                 for n in ast.walk(ast.Module(body=bell.body, type_ignores=[]))
                 if isinstance(n, ast.Call)]
        wrote = [n for n in order if n in ("_write_school", "remote", "spawn")]
        self.assertEqual(wrote[0], "_write_school",
                         "the watermark must be written before anybody is taught")

    def test_the_bell_waits_an_hour_for_a_voice_and_no_longer(self):
        """14 October, 14:07: the call for gpt-2 never came back and the bell
        waited six hours on it, through the zoom at 18:00. The wait is bounded
        now, the call is cancelled when it runs out, and the next voice is
        taught — so one card that never arrives costs one lesson, not nine."""
        import ast
        import pathlib

        src = pathlib.Path(self.T.__file__).read_text(encoding="utf-8")
        bell = next(n for n in ast.parse(src).body
                    if isinstance(n, ast.FunctionDef) and n.name == "school_bell")
        calls = [n for n in ast.walk(ast.Module(body=bell.body, type_ignores=[]))
                 if isinstance(n, ast.Call)]
        names = [getattr(n.func, "attr", getattr(n.func, "id", "")) for n in calls]
        self.assertNotIn("remote", names, "an unbounded wait on one voice")
        self.assertIn("spawn", names)
        got = next(n for n in calls if getattr(n.func, "attr", "") == "get")
        self.assertEqual([k.arg for k in got.keywords], ["timeout"])
        self.assertIn("cancel", names)
        self.assertIn("_bill_if_missing", names)
        # Longer than any lesson ever taught — 43 minutes — and shorter than
        # a tuning session's patience with a silent school.
        self.assertGreaterEqual(self.T.LESSON_PATIENCE_H, 0.75)
        self.assertLess(self.T.LESSON_PATIENCE_H, self.T.LESSON_FLOOR_H)

    def test_a_voice_that_never_came_back_is_written_down_and_not_charged(self):
        """No bill means the card was most likely never granted; what is known
        is written, what is not is not invented. The line is `train` so that
        `bell_rings` counts the voice as returned."""
        written = []
        with mock.patch.object(self.T.field_volume, "reload"), \
             mock.patch.object(self.T, "_read_school", return_value=[]), \
             mock.patch.object(self.T, "_write_school", written.append):
            line = self.T._bill_if_missing("voice_00", "gpt-2", started=100.0,
                                           err=TimeoutError())
        self.assertEqual(written, [line])
        self.assertEqual(line["kind"], "train")
        self.assertEqual(line["status"], "lost")
        self.assertEqual((line["usd"], line["gpu_s"]), (0.0, 0.0))
        self.assertFalse(line["promoted"])
        self.assertIn("TimeoutError", line["reason"])
        ring, why = self.T.bell_rings(
            self.breaths(400),
            [{"timestamp": 100.0, "kind": "lesson", "humans_seen": 0}]
            + self.nine_back()[:8] + [line],
            now=100.0 + self.T.LESSON_FLOOR_H * 3600 + 1)
        self.assertTrue(ring, "the lost voice must not hold the bell")

    def test_a_voice_that_crashed_with_a_bill_is_not_billed_twice(self):
        theirs = {"timestamp": 150.0, "kind": "train", "voice_id": "voice_00",
                  "status": "crashed", "usd": 0.4, "gpu_s": 576.0}
        written = []
        with mock.patch.object(self.T.field_volume, "reload"), \
             mock.patch.object(self.T, "_read_school", return_value=[theirs]), \
             mock.patch.object(self.T, "_write_school", written.append):
            line = self.T._bill_if_missing("voice_00", "gpt-2", started=100.0,
                                           err=RuntimeError("cuda"))
        self.assertEqual(written, [])
        self.assertEqual(line["usd"], 0.4)
        self.assertEqual(line["reason"], "cuda")


class HowTheSchoolDayIsArranged(unittest.TestCase):
    """The dramaturgy of the bell: a tuning session is half an hour long.

    The school's one dramatic moment — the bell, the lessons, the voices coming
    back changed — has to land while the people whose breaths rang it are still
    in the room. That is two arrangements: the bell is checked often enough
    that a session cannot slip between checks, and the lightest voices learn
    first, so the first returns arrive in minutes rather than behind the
    heaviest lesson of the night.
    """

    def setUp(self):
        import training_app as T

        self.T = T

    def test_the_bell_is_checked_every_five_minutes(self):
        import pathlib

        src = pathlib.Path(self.T.__file__).read_text(encoding="utf-8")
        self.assertIn("modal.Period(minutes=5)", src,
                      "a room can fill, sing and empty between bell checks")

    def test_the_lightest_voices_return_first(self):
        from types import SimpleNamespace as Voice

        arrived = [Voice(voice_id=v) for v in
                   ("voice_08", "voice_00", "voice_01", "voice_04")]
        ordered = [v.voice_id for v in self.T._lightest_first(arrived)]
        self.assertEqual(ordered,
                         ["voice_01", "voice_00", "voice_04", "voice_08"],
                         "the audience waits behind the heaviest lesson")

    def test_the_schools_idea_of_weight_matches_the_ledgers(self):
        """The table is a copy — the school's image does not carry
        mortality_sim.py — and a copy that drifts would quietly reorder the
        night. If they differ, change both or neither."""
        import mortality_sim

        self.assertEqual(self.T._PARAMS_B, mortality_sim.PARAMS_B)


if __name__ == "__main__":
    unittest.main(verbosity=2)
