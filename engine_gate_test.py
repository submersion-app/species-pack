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


def evaluation(switched_off=(), junk=0.01):
    return {
        "fixed bars held out": {"switched off": list(switched_off)},
        "junk named (tune, at the chosen bars)": junk,
    }


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


if __name__ == "__main__":
    unittest.main()
