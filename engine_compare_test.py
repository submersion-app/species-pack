#!/usr/bin/env python3
"""Unit tests for engine_compare.py: one engine's run against another's on
the same photos, crop by crop. Needs numpy."""

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_spec = importlib.util.spec_from_file_location("engine_compare", os.path.join(_HERE, "engine_compare.py"))
engine_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine_compare)

import numpy as np  # noqa: E402

TEXT = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float64)
NAMING = {
    "minCropSide": 64,
    "minOpenScore": 0.1,
    "suggestScore": 0.5,
    "suggestScoreNoLocation": 0.5,
    "topK": 3,
}


def line(path, embeddings, x=0.1):
    return {
        "path": path,
        "detections": [
            {"x": x + 0.3 * i, "y": 0.1, "w": 0.2, "h": 0.2, "pixelWidth": 100, "pixelHeight": 100, "embedding": e}
            for i, e in enumerate(embeddings)
        ],
    }


def compare(reference, other):
    return engine_compare.compare(reference, other, TEXT, 20.0, NAMING)


class CompareTests(unittest.TestCase):
    def test_the_same_run_passes(self):
        run = {f"p{i}": line(f"p{i}", [[1.0, 0.1]]) for i in range(100)}
        result = compare(run, run)
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["crops"], 100)

    def test_a_crop_under_the_cosine_floor_fails(self):
        reference = {f"p{i}": line(f"p{i}", [[1.0, 0.0]]) for i in range(100)}
        other = dict(reference)
        # cosine of (1, 0) and (1, 0.0633) is about 0.998
        other["p0"] = line("p0", [[1.0, 0.0633]])
        result = compare(reference, other)
        self.assertFalse(result["pass"])
        self.assertEqual(result["below cosine floor"], 1)

    def test_two_in_a_hundred_named_differently_fails(self):
        reference = {f"p{i}": line(f"p{i}", [[1.0, 0.0]]) for i in range(100)}
        other = dict(reference)
        for i in (0, 1):
            other[f"p{i}"] = line(f"p{i}", [[0.0, 1.0]])
        result = compare(reference, other)
        self.assertAlmostEqual(result["same outcome share"], 0.98)
        self.assertFalse(result["pass"])

    def test_a_box_found_by_one_run_only_counts_as_different(self):
        reference = {f"p{i}": line(f"p{i}", [[1.0, 0.0]]) for i in range(100)}
        other = dict(reference)
        other["p0"] = line("p0", [[1.0, 0.0], [0.0, 1.0]])
        result = compare(reference, other)
        self.assertEqual(result["unmatched"], 1)


if __name__ == "__main__":
    unittest.main()
