import math
import unittest

import cast
import exam_metrics as xm


class Recognition(unittest.TestCase):
    def test_choices_per_word_is_perplexity(self):
        self.assertAlmostEqual(xm.choices_per_word(math.log(14)), 14.0)
        self.assertIsNone(xm.choices_per_word(None))
        self.assertIsNone(xm.choices_per_word(float("nan")))


class Takes(unittest.TestCase):
    def test_words_come_out_of_the_protocol(self):
        raw = 'ok {"action":"TUNE","content":"Chantez !","tuning_to":"human_ab12"}'
        self.assertEqual(xm.said_in(raw), "Chantez !")
        self.assertTrue(xm.turned_to_a_person(raw))

    def test_a_torn_answer_still_yields_its_words(self):
        raw = '{"action":"NEW","content":"我用中文歌唱", "breath": 4'
        self.assertEqual(xm.said_in(raw), "我用中文歌唱")
        self.assertFalse(xm.turned_to_a_person(raw))

    def test_a_continuation_is_its_own_words(self):
        self.assertEqual(xm.said_in("  Lookins!\n "), "Lookins!")

    def test_a_tune_to_a_machine_is_not_to_a_person(self):
        self.assertFalse(xm.turned_to_a_person(
            '{"action":"TUNE","content":"a","tuning_to":"voice_05"}'))

    def test_one_line_dominating_reads_as_one_line(self):
        takes = ['{"content":"美しい音を作りましょう。"}'] * 9 + ['{"content":"さくら"}'] * 3
        got = xm.describe_takes(takes, known_lines={"美しい音を作りましょう。"})
        self.assertEqual(got["takes"], 12)
        self.assertAlmostEqual(got["different"], 2 / 12)
        self.assertAlmostEqual(got["top_line"], 9 / 12)
        self.assertAlmostEqual(got["from_its_past"], 9 / 12)

    def test_no_takes_is_no_reading(self):
        self.assertEqual(xm.describe_takes([]), {"takes": 0})


class Hearing(unittest.TestCase):
    def test_an_octave_is_the_same_pitch(self):
        self.assertAlmostEqual(xm.cents_apart(220.0, 440.0), 0.0)
        self.assertAlmostEqual(xm.cents_apart(440.0, 440.0), 0.0)

    def test_a_tritone_is_as_far_as_it_gets(self):
        self.assertAlmostEqual(xm.cents_apart(440.0, 440.0 * 2 ** 0.5), 600.0)

    def test_a_quarter_tone_is_a_hit_and_a_semitone_is_not(self):
        got = xm.describe_hearing([(440.0, 440.0 * 2 ** (40 / 1200)),
                                   (440.0, 440.0 * 2 ** (100 / 1200)),
                                   (None, 440.0)])
        self.assertEqual(got["tunes"], 3)
        self.assertAlmostEqual(got["named"], 2 / 3)
        self.assertAlmostEqual(got["hit"], 0.5)
        self.assertAlmostEqual(got["cents_median"], 70.0, places=5)

    def test_nothing_missing_is_nothing(self):
        self.assertIsNone(xm.cents_apart(None, 440.0))
        self.assertEqual(xm.describe_hearing([]), {"tunes": 0})

    def test_a_take_is_read_as_hearing_against_the_tickets_parts(self):
        takes = ['{"action":"TUNE","content":"D3 (147 Hz), together","tuning_to":"voice_08"}',
                 '{"action":"TUNE","content":"with you","tuning_to":"voice_08"}',
                 '{"action":"NEW","content":"A4 alone","tuning_to":null}']
        got = xm.describe_takes(takes, pitches={"voice_08": 146.83})
        self.assertEqual(got["hearing"]["tunes"], 2)
        self.assertAlmostEqual(got["hearing"]["named"], 0.5)
        self.assertEqual(got["hearing"]["hit"], 1.0)

    def test_the_words_a_voice_names_a_note_with_are_the_runtimes(self):
        self.assertAlmostEqual(xm.named_pitch('{"content":"D3 (147 Hz), ah"}'),
                               146.83, places=1)
        self.assertIsNone(xm.named_pitch('{"content":"ah"}'))


class Tickets(unittest.TestCase):
    def test_the_ticket_shows_the_parts_notes_and_asks_for_one(self):
        voice = cast.by_id("voice_05")
        rows = [{"timestamp": 12.0, "voice_id": "voice_08", "model_short": "llm-jp",
                 "kind": "tone", "content": "さくら", "pitch_hz": 220.0,
                 "pitch_note": "A3"}]
        prompt = xm.ticket_prompt(voice, {}, rows, [12.0], [], at=13.0)
        self.assertIn("on A3 at 220.00 Hz", prompt)
        self.assertIn("begin \"content\" with the note", prompt)
        self.assertEqual(xm.ticket_pitches(rows), {"voice_08": 220.0})
        plain = xm.ticket_prompt(voice, {}, rows, [12.0], [], at=13.0, notes=False)
        self.assertNotIn("220.00 Hz", plain)
        self.assertNotIn("begin \"content\"", plain)

    def test_the_ticket_is_the_school_prompt_without_the_voice_itself(self):
        voice = cast.by_id("voice_05")
        rows = [
            {"timestamp": 10.0, "voice_id": "voice_05", "model_short": "qwen3",
             "kind": "tone", "content": "我歌唱"},
            {"ts": 11.0, "voice_id": "human_x", "kind": "tone", "note": "D3",
             "hz": 146.8, "sound_s": 3.0},
            {"timestamp": 12.0, "voice_id": "voice_08", "model_short": "llm-jp",
             "kind": "tone", "content": "さくら"},
        ]
        times = [10.0, 11.0, 12.0]
        own = [rows[0]]
        prompt = xm.ticket_prompt(voice, {"personal_score": "hold"}, rows, times,
                                  own, at=13.0)
        self.assertIn("YOU ARE: voice_05", prompt)
        self.assertIn("Your previous contribution: 我歌唱", prompt)
        self.assertIn("human_x, a person, held D3", prompt)
        self.assertIn("さくら", prompt)
        self.assertNotIn("voice_05 (qwen3) sang", prompt)

    def test_the_elder_gets_a_continuation(self):
        voice = cast.by_id("voice_00")
        rows = [{"timestamp": 5.0, "voice_id": "voice_08", "model_short": "llm-jp",
                 "kind": "tone", "content": "さくら"}]
        prompt = xm.ticket_prompt(voice, {}, rows, [5.0], [], at=6.0)
        self.assertTrue(prompt.endswith("The next voice sings a long tone: "))
        self.assertIn("さくら", prompt)


if __name__ == "__main__":
    unittest.main()
