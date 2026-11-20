#!/usr/bin/env python3
"""Unit tests for score.py's measured entries (issue #4721): the figures
NAMING_MEASURED copies into manifest.py, from levels measured by
levels.py."""

import argparse
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bars  # noqa: E402
import levels  # noqa: E402
import score  # noqa: E402

NEVER = bars.NEVER
SURER = bars.Bars(min_open=0.5, suggest=0.80, suggest_no_location=0.80, confident=NEVER, confident_no_location=NEVER)
MANIFEST = {
    "naming": {"minCropSide": 64, "minAllowedCandidates": 200, "topK": 5},
    "classifier": {"id": "classifier"},
    "packVersion": "2026.11.18.0356",
    "files": [],
}


def crop(with_top, without_top, correct):
    return {
        "with": bars.Scored(open_score=0.6, top_score=with_top, correct=correct, shorter_side=100, limited=True),
        "without": bars.Scored(open_score=0.6, top_score=without_top, correct=correct, shorter_side=100, limited=False),
    }


def entry(fish):
    found, surer_precision, _ = levels.measure(SURER, fish, [], fish, [], min_side=64)
    evaluation = {
        "levels measured under": {"bars": SURER.__dict__, "published": True},
        "surer precision": surer_precision,
        "levels": found,
    }
    return score._measured_entry(evaluation, MANIFEST, ["detector"], "centreCrop")["naming"]


class MeasuredEntryTests(unittest.TestCase):
    def test_a_measured_entry_carries_the_shown_figures(self):
        # Balanced is right 170 of 200 times with a location and 170 of 205
        # without; Surer 100 of 100 times either way.
        fish = (
            [crop(0.9, 0.9, True)] * 100
            + [crop(0.55, 0.55, True)] * 70
            + [crop(0.55, 0.55, False)] * 30
            + [crop(0.3, 0.56, False)] * 5
        )
        naming = entry(fish)
        self.assertEqual(naming["precision"], 1.0)
        self.assertEqual([(level["id"], level["precision"]) for level in naming["levels"]], [("balanced", 0.829)])
        self.assertEqual((naming["suggestScore"], naming["suggestScoreNoLocation"]), (0.8, 0.8))

    def test_a_surer_with_no_share_to_show_carries_no_precision(self):
        naming = entry([crop(0.9, 0.9, False)] * 40)
        self.assertNotIn("precision", naming)
        self.assertEqual(naming["levels"], [])


class EngineTests(unittest.TestCase):
    def test_a_core_ml_run_measures_the_swift_and_an_onnx_run_the_dart(self):
        coreml = [os.path.basename(p) for p in score.source_files("coreml")]
        onnx = [os.path.basename(p) for p in score.source_files("onnx")]
        self.assertIn("VisionPipeline.swift", coreml)
        self.assertNotIn("VisionPipeline.swift", onnx)
        for dart in ("onnx_pipeline.dart", "pixels.dart", "detector_reading.dart", "flutter_image_decoder.dart"):
            self.assertIn(dart, onnx)
        self.assertIn("species_vision_onnx_test.dart", onnx)

    def test_onnx_results_sit_beside_core_mls_not_over_them(self):
        self.assertEqual(score.results_dir("/w", "coreml"), "/w")
        self.assertEqual(score.results_dir("/w", "onnx"), os.path.join("/w", "onnx"))
        self.assertNotEqual(
            score.run_file("/w", "both", "centreCrop", "v1", "coreml"),
            score.run_file("/w", "both", "centreCrop", "v1", "onnx"),
        )

    def test_the_onnx_request_names_everything_the_runner_reads(self):
        request = score.onnx_request(
            manifest="/m.json",
            detectors=["/d1/model.onnx", "/d2/model.onnx"],
            classifier="/c/model.onnx",
            fit="centreCrop",
            detector_boxes=True,
            photos=["/p/a.jpg"],
            out="/out.jsonl",
            library="/lib/libonnxruntime.dylib",
        )
        self.assertEqual(
            sorted(request),
            ["classifier", "cropFit", "detectorBoxes", "detectors", "manifest", "ortLibrary", "out", "photos"],
        )
        self.assertEqual(request["detectors"], ["/d1/model.onnx", "/d2/model.onnx"])

    def test_the_mac_library_is_pinned(self):
        import ort_library

        self.assertRegex(ort_library.NUPKG_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(ort_library.DYLIB_SHA256, r"^[0-9a-f]{64}$")
        self.assertTrue(ort_library.NUPKG_URL.startswith("https://api.nuget.org/"))

class JunkMatchTests(unittest.TestCase):
    ROW = {"source": "commons", "id": "7", "detector": "cfd", "box": "0", "box_x": "0.1", "box_y": "0.1", "box_w": "0.2", "box_h": "0.2"}

    def test_a_core_ml_detector_run_must_find_every_junk_row(self):
        with self.assertRaisesRegex(SystemExit, "1 junk rows match no box"):
            score._junk_records("cfd", {}, [self.ROW], None, "/w", [0])

    def test_an_onnx_detector_run_leaves_unmatched_junk_rows_out(self):
        records, missing = score._junk_records("cfd", {}, [self.ROW], None, "/w", [0], strict=False)
        self.assertEqual((records, missing), ([], ["cfd:7:0"]))


class PathTests(unittest.TestCase):
    def test_the_pack_and_work_folders_are_made_absolute(self):
        args = argparse.Namespace(pack="pack", work=os.path.join("w", "x"), naming_from=None)
        score.absolute_paths(args)
        self.assertEqual(args.pack, os.path.abspath("pack"))
        self.assertEqual(args.work, os.path.abspath(os.path.join("w", "x")))
        self.assertIsNone(args.naming_from)

class OnnxRunTests(unittest.TestCase):
    def test_the_onnx_fingerprint_covers_what_its_output_depends_on(self):
        onnx = {os.path.basename(p) for p in score.source_files("onnx")}
        for dart in ("vision_analysis.dart", "species_pack_manifest.dart", "ort_native.dart", "ort_c_api.g.dart"):
            self.assertIn(dart, onnx)

    def test_an_onnx_run_records_no_peak_memory(self):
        timing = "  123456  maximum resident set size"
        self.assertEqual(score.peak_memory("coreml", timing), 123456)
        self.assertIsNone(score.peak_memory("onnx", timing))

    def test_a_failing_runner_stops_with_its_output(self):
        import sys as _sys
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(SystemExit, "boom"):
                score.run_logged(
                    [_sys.executable, "-c", "print('boom'); raise SystemExit(3)"],
                    cwd=d,
                    env=dict(os.environ),
                    log=os.path.join(d, "runner.txt"),
                )


class EvaluateHelperTests(unittest.TestCase):
    def test_each_held_out_junk_crop_is_paired_with_its_own_result(self):
        crops = [{"row": {"detector": "cfd", "id": "7", "box": "0"}}, {"row": {"detector": "mega", "id": "7", "box": "1"}}]
        self.assertEqual(score.junk_named_by_row(crops, [False, True]), {"cfd:7:0": False, "mega:7:1": True})

    def test_the_held_out_check_is_at_the_fixed_bars_only(self):
        self.assertIsNone(score.fixed_bars_held_out(None, [], min_side=64))
        fixed = bars.Bars(min_open=0.1, suggest=0.7, suggest_no_location=0.8, confident=0.93, confident_no_location=1.01)
        # With nothing held out there is no evidence, so no bar stands.
        held = score.fixed_bars_held_out(fixed, [], min_side=64)
        self.assertEqual(sorted(held["switched off"]), ["confidentScore", "suggestScore", "suggestScoreNoLocation"])


if __name__ == "__main__":
    unittest.main()
