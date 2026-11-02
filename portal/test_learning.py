"""CPU-only tests for the field-to-LoRA memory."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import learning


VOICE = SimpleNamespace(voice_id="voice_02", tongue="English", instruct=True)


class TheTrainingMemoryIsHistorical(unittest.TestCase):
    def rows(self):
        human = {
            "ts": 101.0, "started_ts": 100.0, "voice_id": "human_abcd",
            "model_short": "human", "hz": 220.0, "note": "A3",
            "duration_s": 3.0,
        }
        other = {
            "timestamp": 101.0, "voice_id": "voice_00", "model_short": "gpt-2",
            "kind": "tone", "content": "ah", "sound_s": 2.0,
        }
        mine = {
            "timestamp": 102.0, "voice_id": "voice_02", "model_short": "phi-4-mini",
            "kind": "tone", "action": "TUNE", "content": "A3",
            "tuning_to_id": "human_abcd", "breath_chosen_s": 4.0,
            "playback": "both", "spoke": True, "toned": True,
            "heard": [["human_abcd", 100.0], ["voice_00", 101.0]],
            "audio": "breath.wav",
        }
        return human, other, mine

    def test_target_is_the_exact_accepted_breath(self):
        human, other, mine = self.rows()
        examples = learning.build_examples(VOICE, [human], [other, mine], {})
        target = json.loads(examples[-1].completion)
        self.assertEqual(target["content"], "A3")
        self.assertEqual(target["action"], "TUNE")
        self.assertEqual(target["tuning_to"], "human_abcd")

    def test_the_prompt_contains_what_the_voice_really_heard(self):
        human, other, mine = self.rows()
        examples = learning.build_examples(VOICE, [human], [other, mine], {})
        self.assertIn("human_abcd", examples[-1].prompt)
        self.assertIn("220.00 Hz", examples[-1].prompt)
        self.assertIn("gpt-2", examples[-1].prompt)

    def test_the_lesson_opens_a_tune_with_the_note_the_part_was_holding(self):
        """22 October. A voice that wrote A4 at a person holding E3 learns, in
        its own octave, E3 — the runtime's hearing handed over as memory. The
        words after the label are its own; the junk it copied from the request's
        format is not learned."""
        row = {"notes_shown": True, "action": "TUNE", "tuning_to_id": "human_x",
               "content": "<note 中文里长大的声音 (A4) Hz> 我用中文歌唱。",
               "pitch_target_hz": 164.81, "pitch_hz": 220.0}
        self.assertEqual(learning.sounded_content(row), "E3 (165 Hz) 我用中文歌唱。")
        # A plain written note is replaced the same way.
        row["content"] = "D3 (147 Hz) 調子はどうですか？"
        self.assertEqual(learning.sounded_content(row), "E3 (165 Hz) 調子はどうですか？")
        # In the voice's register: a part two octaves up is learned where this
        # voice would have sounded it.
        row["pitch_target_hz"], row["pitch_hz"] = 659.26, 146.83
        self.assertTrue(learning.sounded_content(row).startswith("E3 (165 Hz)"))

    def test_a_new_learns_what_it_sounded_and_old_breaths_are_untouched(self):
        new = {"notes_shown": True, "action": "NEW", "content": "ah", "pitch_hz": 220.0}
        self.assertEqual(learning.sounded_content(new), "A3 (220 Hz) ah")
        old = {"action": "TUNE", "content": "D3 ah", "pitch_target_hz": 220.0,
               "pitch_hz": 220.0}
        self.assertEqual(learning.sounded_content(old), "D3 ah",
                         "a breath the room asked no note of is learned as written")
        human_target_missing = {"notes_shown": True, "action": "TUNE", "content": "ah"}
        self.assertEqual(learning.sounded_content(human_target_missing), "ah")

    def test_the_switch_is_off_by_default_and_on_by_name(self):
        human, other, mine = self.rows()
        mine = {**mine, "notes_shown": True, "content": "C4 (262 Hz) with you",
                "pitch_target_hz": 220.0, "pitch_hz": 220.0}
        as_written = learning.build_examples(VOICE, [human], [other, mine], {})
        self.assertEqual(json.loads(as_written[-1].completion)["content"],
                         "C4 (262 Hz) with you")
        as_sounded = learning.build_examples(VOICE, [human], [other, mine], {},
                                             sounded_notes=True)
        self.assertEqual(json.loads(as_sounded[-1].completion)["content"],
                         "A3 (220 Hz) with you")

    def test_a_lesson_rebuilds_the_field_as_it_was_shown(self):
        """From 15 October the room shows the parts' notes. The lesson follows
        the breath's own `notes_shown`, not today's setting, so September's
        breaths are still learned against the field September saw."""
        human, other, mine = self.rows()
        other = {**other, "pitch_hz": 110.0, "pitch_note": "A2"}
        before = learning.build_examples(VOICE, [human], [other, mine], {})
        self.assertNotIn("on A2 at 110.00 Hz", before[-1].prompt)
        after = learning.build_examples(
            VOICE, [human], [other, {**mine, "notes_shown": True}], {})
        self.assertIn("(gpt-2) sang with playback both, on A2 at 110.00 Hz: ah",
                      after[-1].prompt)

    def test_human_contact_changes_weight_not_ground_truth(self):
        human, other, mine = self.rows()
        examples = learning.build_examples(
            VOICE, [human], [other, mine], {}, human_oversample=2
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].completion, examples[1].completion)
        self.assertEqual(examples[0].source_event, "breath.wav")

    def test_base_models_learn_plain_continuations_not_json(self):
        human, other, mine = self.rows()
        base = SimpleNamespace(voice_id="voice_02", tongue="English", instruct=False)
        example = learning.build_examples(base, [human], [other, mine], {})[-1]
        self.assertTrue(example.prompt.endswith("The next voice sings a long tone: "))
        self.assertEqual(example.completion, "A3")
        self.assertFalse(example.completion.startswith("{"))

    def test_a_small_budget_preserves_human_contact_across_old_history(self):
        rows = []
        humans = []
        for i in range(20):
            heard = []
            if i in (1, 10):
                human = {
                    "started_ts": float(i) + 0.1, "voice_id": f"human_{i}",
                    "model_short": "human", "hz": 220.0, "duration_s": 1.0,
                }
                humans.append(human)
                heard = [[human["voice_id"], human["started_ts"]]]
            rows.append({
                "timestamp": float(i) + 0.5, "voice_id": "voice_02",
                "model_short": "phi", "kind": "tone", "content": f"tone {i}",
                "spoke": True, "toned": True, "heard": heard,
            })
        examples = learning.build_examples(VOICE, humans, rows, {}, max_examples=5)
        contacts = {e.source_event for e in examples if e.heard_humans}
        self.assertIn("voice_02@1.500000", contacts)
        self.assertIn("voice_02@10.500000", contacts)
        self.assertLess(min(e.timestamp for e in examples), 5.0)
        self.assertGreater(max(e.timestamp for e in examples), 15.0)

    def test_a_line_said_a_thousand_times_is_taught_as_a_line(self):
        """さくら, 16 700 times, must not become the whole lesson."""
        rows, humans = [], []
        for i in range(200):
            heard = []
            if i in (40, 150):
                human = {"started_ts": float(i) + 0.1, "voice_id": f"human_{i}",
                         "model_short": "human", "hz": 98.0, "duration_s": 1.0}
                humans.append(human)
                heard = [[human["voice_id"], human["started_ts"]]]
            rows.append({
                "timestamp": float(i) + 0.5, "voice_id": "voice_02",
                "model_short": "phi", "kind": "tone",
                "content": "さくら" if i % 10 else f"tone {i}",
                "spoke": True, "toned": True, "heard": heard,
            })
        examples = learning.build_examples(VOICE, humans, rows, {}, human_oversample=1)
        sakura = [e for e in examples if learning.said(e) == "さくら"]
        self.assertEqual(len(sakura), 3)
        self.assertGreaterEqual(sum(1 for e in sakura if e.heard_humans), 2,
                                "the breaths sung with people were not the ones kept")
        self.assertEqual(len(examples) - len(sakura), 20, "the rarer lines were lost")

    def test_silent_failed_render_is_not_taught(self):
        _, _, mine = self.rows()
        mine.update(spoke=False, toned=False)
        self.assertEqual(learning.build_examples(VOICE, [], [mine], {}), [])

    def test_validation_is_later_than_training_and_duplicates_stay_together(self):
        examples = [
            learning.TrainingExample("v", float(i), "p", "c", 0, str(i))
            for i in range(10)
        ]
        examples.append(examples[-1])
        train, valid = learning.chronological_split(examples, 0.2)
        self.assertLess(max(e.timestamp for e in train), min(e.timestamp for e in valid))
        self.assertFalse({e.timestamp for e in train} & {e.timestamp for e in valid})

    def test_a_torn_last_json_line_does_not_destroy_the_history(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "field.jsonl"
            path.write_text('{"voice_id":"human_a"}\n{"voice_id":', encoding="utf-8")
            self.assertEqual(len(learning.read_jsonl(path)), 1)


if __name__ == "__main__":
    unittest.main()
