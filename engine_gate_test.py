#!/usr/bin/env python3
"""Unit tests for engine_gate.py: the ONNX engine against Core ML on one
pack. Standard library only."""

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_spec = importlib.util.spec_from_file_location("engine_gate", os.path.join(_HERE, "engine_gate.py"))
engine_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine_gate)


def photos(n_right, n_wrong, group="fish"):
    out = {}
    for i in range(n_right + n_wrong):
        out[f"{group}{i}"] = {"named_right": i < n_right, "held_out": True, "sample": "natural", "group": group}
    return out


def evaluation(switched_off=(), junk=0.01, moved=None):
    out = {
        "fixed bars held out": {"switched off": list(switched_off)},
        "junk named (tune, at the chosen bars)": junk,
    }
    if moved is not None:
        out["held-out junk rows moved"] = list(moved)
    return out


def junk(named):
    return {f"cfd:{i}:0": flag for i, flag in enumerate(named)}


class GateTests(unittest.TestCase):
    def check(self, coreml=None, onnx=None, onnx_eval=None, coreml_junk=None, onnx_junk=None):
        base = photos(300, 100)
        return engine_gate.check(
            coreml if coreml is not None else base,
            onnx if onnx is not None else base,
            onnx_eval if onnx_eval is not None else evaluation(),
            coreml_junk if coreml_junk is not None else junk([False] * 50),
            onnx_junk if onnx_junk is not None else junk([False] * 50),
        )

    def test_equal_results_pass(self):
        result = self.check()
        self.assertTrue(result["pass"], result)

    def test_the_packs_bars_must_stand_held_out(self):
        self.assertFalse(self.check(onnx_eval=evaluation(switched_off=["confident"]))["pass"])

    def test_a_run_not_at_the_packs_bars_fails(self):
        result = self.check(onnx_eval={"fixed bars held out": None, "junk named (tune, at the chosen bars)": 0.0})
        self.assertFalse(result["pass"])
        self.assertFalse(result["pack bars hold"]["pass"])

    def test_ten_photos_named_right_only_by_core_ml_fails(self):
        onnx = photos(290, 110)
        result = self.check(onnx=onnx)
        self.assertFalse(result["not fewer named right"]["pass"])
        self.assertFalse(result["pass"])

    def test_one_photo_either_way_is_noise(self):
        onnx = dict(photos(300, 100))
        onnx["fish0"] = {**onnx["fish0"], "named_right": False}
        onnx["fish350"] = {**onnx["fish350"], "named_right": True}
        self.assertTrue(self.check(onnx=onnx)["pass"])

    def test_five_photos_to_none_is_still_noise(self):
        # Exact McNemar, two-sided: 5 to 0 is p 0.0625, over alpha 0.05.
        self.assertTrue(self.check(onnx=photos(295, 105))["not fewer named right"]["pass"])

    def test_six_photos_to_none_fails(self):
        # 6 to 0 is p 0.031.
        self.assertFalse(self.check(onnx=photos(294, 106))["not fewer named right"]["pass"])

    def test_junk_exactly_at_the_limit_passes(self):
        self.assertTrue(self.check(onnx_eval=evaluation(junk=0.02))["junk"]["pass"])

    def test_a_group_significantly_worse_fails(self):
        coreml = {**photos(300, 100), **photos(40, 10, group="coral")}
        onnx = {**photos(300, 100), **photos(28, 22, group="coral")}
        result = self.check(coreml=coreml, onnx=onnx)
        self.assertEqual(result["no group worse"]["worse"], ["coral"])

    def test_junk_over_the_limit_fails(self):
        self.assertFalse(self.check(onnx_eval=evaluation(junk=0.021))["junk"]["pass"])

    def test_junk_named_significantly_more_often_fails(self):
        result = self.check(onnx_junk=junk([True] * 10 + [False] * 40))
        self.assertFalse(result["junk"]["pass"])

    def test_photos_held_in_only_one_run_are_left_out(self):
        coreml = photos(300, 100)
        onnx = dict(coreml)
        del onnx["fish0"]
        self.assertEqual(self.check(coreml=coreml, onnx=onnx)["not fewer named right"]["photos"], 399)

    def test_junk_with_no_crops_in_common_fails(self):
        result = self.check(coreml_junk={"cfd:a:0": False}, onnx_junk={"cfd:b:0": False})
        self.assertFalse(result["junk"]["pass"])
        self.assertEqual(result["junk"]["held-out crops"], 0)

    def test_a_few_crops_only_one_engine_found_are_left_out(self):
        onnx = junk([False] * 50)
        for key in ("cfd:0:0", "cfd:1:0"):
            del onnx[key]
        result = self.check(onnx_junk=onnx)["junk"]
        self.assertTrue(result["pass"], result)
        self.assertEqual((result["held-out crops"], result["only in one run"]), (48, 2))

    def test_many_crops_only_one_engine_found_fail(self):
        onnx = junk([False] * 50)
        for i in range(5):
            del onnx[f"cfd:{i}:0"]
        self.assertFalse(self.check(onnx_junk=onnx)["junk"]["pass"])

    @staticmethod
    def lost(n):
        """ONNX's junk with its first `n` crops lost, and their keys."""
        onnx = junk([False] * 50)
        keys = [f"cfd:{i}:0" for i in range(n)]
        for key in keys:
            del onnx[key]
        return onnx, keys

    def test_crops_onnx_drew_no_box_over_are_not_counted(self):
        # It named nothing there, so leaving them out hides nothing.
        onnx, _ = self.lost(5)
        result = self.check(onnx_eval=evaluation(moved=[]), onnx_junk=onnx)["junk"]
        self.assertTrue(result["pass"], result)
        self.assertEqual((result["only in one run"], result["counted against the pairing"]), (5, 0))

    def test_crops_onnx_moved_its_box_off_are_counted(self):
        # Its moved box could have named junk the pairing does not see.
        onnx, keys = self.lost(5)
        result = self.check(onnx_eval=evaluation(moved=keys), onnx_junk=onnx)["junk"]
        self.assertFalse(result["pass"])
        self.assertEqual(result["counted against the pairing"], 5)

    def test_a_run_that_does_not_record_moved_boxes_counts_every_lost_crop(self):
        onnx, _ = self.lost(5)
        result = self.check(onnx_eval=evaluation(), onnx_junk=onnx)["junk"]
        self.assertFalse(result["pass"])
        self.assertEqual(result["counted against the pairing"], 5)

    def test_crops_only_onnx_found_are_counted(self):
        coreml, _ = self.lost(5)
        result = self.check(onnx_eval=evaluation(moved=[]), coreml_junk=coreml)["junk"]
        self.assertFalse(result["pass"])
        self.assertEqual(result["counted against the pairing"], 5)


class PackBarsTests(unittest.TestCase):
    NAMING = {
        "provisional": False,
        "minOpenScore": 0.1,
        "suggestScore": 0.7,
        "suggestScoreNoLocation": 0.8,
        "confidentScore": 0.93,
        "confidentScoreNoLocation": 1.01,
    }
    BARS = {"min_open": 0.1, "suggest": 0.7, "suggest_no_location": 0.8, "confident": 0.93, "confident_no_location": 1.01}

    def confirm(self, naming=None, coreml=None, onnx=None):
        engine_gate.confirm_pack_bars(
            {"naming": naming or self.NAMING},
            {"bars after held-out gate": coreml or self.BARS},
            {"bars after held-out gate": onnx or self.BARS},
        )

    def test_runs_named_at_the_pack_s_bars_pass(self):
        self.confirm()

    def test_an_onnx_run_at_another_pack_s_bars_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "ONNX run"):
            self.confirm(onnx={**self.BARS, "suggest": 0.6})

    def test_a_core_ml_run_at_bars_it_found_again_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "Core ML run.*--naming-from"):
            self.confirm(coreml={**self.BARS, "confident": 0.95})

    def test_a_pack_with_provisional_bars_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "provisional"):
            self.confirm(naming={"provisional": True})


class SetKeyChoiceTest(unittest.TestCase):
    def test_the_key_comes_from_the_pack_unless_asked(self):
        pack = {
            "detectors": [{"id": "cfd-x"}],
            "boxes": {},
            "classifier": {"cropFit": "padSquare"},
        }
        self.assertEqual(engine_gate.chosen_set_key(pack, None), "cfd-x padSquare")
        with self.assertRaisesRegex(SystemExit, "not the combination this pack ships"):
            engine_gate.chosen_set_key(pack, "both centreCrop")


if __name__ == "__main__":
    unittest.main()
