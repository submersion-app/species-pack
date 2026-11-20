#!/usr/bin/env python3
"""Unit tests for engine_compare.py: one engine's run against another's on
the same photos, crop by crop. Needs numpy."""

import importlib.util
import json
import os
import sys
import tempfile
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

    def test_a_partial_run_fails(self):
        reference = {f"p{i}": line(f"p{i}", [[1.0, 0.1]]) for i in range(100)}
        result = compare(reference, {"p0": reference["p0"]})
        self.assertFalse(result["pass"])
        self.assertEqual(result["photos in one run only"], 99)

    def test_an_error_line_fails(self):
        reference = {f"p{i}": line(f"p{i}", [[1.0, 0.1]]) for i in range(100)}
        other = {**reference, "p0": {"path": "p0", "error": "unreadable"}}
        result = compare(reference, other)
        self.assertFalse(result["pass"])
        self.assertEqual(result["error lines"], 1)

    def test_a_zero_embedding_is_under_the_cosine_floor(self):
        reference = {f"p{i}": line(f"p{i}", [[1.0, 0.0]]) for i in range(100)}
        other = {**reference, "p0": line("p0", [[0.0, 0.0]])}
        result = compare(reference, other)
        self.assertFalse(result["pass"])
        self.assertEqual(result["below cosine floor"], 1)

    def test_each_run_is_named_at_its_own_crop_size(self):
        reference = {f"p{i}": line(f"p{i}", [[1.0, 0.1]]) for i in range(100)}
        other = {k: {**v, "detections": [dict(d) for d in v["detections"]]} for k, v in reference.items()}
        # On the other machine two crops come out under the 64 px floor, so
        # the app there would not suggest them.
        for i in range(2):
            other[f"p{i}"]["detections"][0].update(pixelWidth=60, pixelHeight=60)
        result = compare(reference, other)
        self.assertAlmostEqual(result["same outcome share"], 0.98)
        self.assertFalse(result["pass"])


class ReadTests(unittest.TestCase):
    def write(self, folder, rows):
        path = os.path.join(folder, "run.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def test_photos_are_matched_by_name_across_machines(self):
        with tempfile.TemporaryDirectory() as d:
            mac = engine_compare.read_run(self.write(d, [line("/Users/a/work/photos/inaturalist_1.jpg", [[1.0, 0.0]])]))
        with tempfile.TemporaryDirectory() as d:
            pc = engine_compare.read_run(
                self.write(d, [line("C:\\work\\photos\\inaturalist_1.jpg", [[1.0, 0.0]])])
            )
        self.assertEqual(set(mac), {"inaturalist_1.jpg"})
        self.assertEqual(set(pc), {"inaturalist_1.jpg"})

    def test_two_photos_of_one_name_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.write(d, [line("/a/x.jpg", [[1.0, 0.0]]), line("/b/x.jpg", [[1.0, 0.0]])])
            with self.assertRaisesRegex(SystemExit, "x.jpg"):
                engine_compare.read_run(path)

    def test_error_lines_are_kept(self):
        with tempfile.TemporaryDirectory() as d:
            run = engine_compare.read_run(self.write(d, [{"path": "/a/x.jpg", "error": "unreadable"}]))
        self.assertEqual(run["x.jpg"]["error"], "unreadable")


if __name__ == "__main__":
    unittest.main()
