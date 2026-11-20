#!/usr/bin/env python3
"""Unit tests for onnx_models.py: 16-bit weight storage, the zip layout the
app unpacks, and the agreement check. Those needing onnx and onnxruntime
skip without them."""

import importlib.util
import os
import tempfile
import unittest
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


onnx_models = _load("onnx_models")
manifest = _load("manifest")

try:
    import numpy as np
    import onnx
    import onnxruntime  # noqa: F401
    from onnx import TensorProto, helper, numpy_helper

    HAVE_ONNX = True
except ImportError:
    HAVE_ONNX = False


def _matmul_model():
    """y = x @ W, x [1, 3], W 3 by 3 of values exact in 16 bits."""
    w = np.array([[1.0, 0.5, -2.0], [0.25, 3.0, 1.5], [-1.0, 0.0, 4.0]], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "matmul",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(w, "W")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), w


@unittest.skipUnless(HAVE_ONNX, "needs onnx and onnxruntime")
class Float16WeightsTests(unittest.TestCase):
    def test_weights_are_stored_in_16_bits_and_cast_back(self):
        model, _ = _matmul_model()
        half = onnx_models.weights_to_float16(model)
        types = {i.name: i.data_type for i in half.graph.initializer}
        self.assertEqual(types, {"W_f16": TensorProto.FLOAT16})
        first = half.graph.node[0]
        self.assertEqual((first.op_type, list(first.input), list(first.output)), ("Cast", ["W_f16"], ["W"]))
        onnx.checker.check_model(half)

    def test_the_ir_version_is_one_the_pinned_runtime_reads(self):
        model, _ = _matmul_model()
        model.ir_version = 14
        self.assertEqual(onnx_models.weights_to_float16(model).ir_version, onnx_models.MAX_IR_VERSION)
        self.assertEqual(onnx_models.MAX_IR_VERSION, 13)

    def test_inputs_and_outputs_stay_32_bit(self):
        half = onnx_models.weights_to_float16(_matmul_model()[0])
        self.assertEqual(half.graph.input[0].type.tensor_type.elem_type, TensorProto.FLOAT)
        self.assertEqual(half.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)

    def test_the_model_still_runs_to_the_same_answer(self):
        model, w = _matmul_model()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            onnx.save(onnx_models.weights_to_float16(model), path)
            x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            (y,) = onnx_models.run(path, x)
            np.testing.assert_allclose(y, x @ w, rtol=0, atol=1e-6)

    def test_cosine_of_a_copy_is_one(self):
        a = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        self.assertAlmostEqual(onnx_models.cosine(a, a.copy()), 1.0, places=6)
        self.assertLess(onnx_models.cosine(a, -a), 0)


class ZipTests(unittest.TestCase):
    def test_the_model_is_at_the_top_of_its_zip(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "anything.onnx")
            with open(model, "wb") as fh:
                fh.write(b"model")
            dest = onnx_models.zip_onnx(model, os.path.join(d, "classifier.onnx.zip"))
            with zipfile.ZipFile(dest) as zf:
                self.assertEqual(zf.namelist(), ["model.onnx"])

    def test_external_weights_travel_beside_it(self):
        with tempfile.TemporaryDirectory() as d:
            model = os.path.join(d, "model.onnx")
            for name in ("model.onnx", "model.onnx.data"):
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(b"x")
            dest = onnx_models.zip_onnx(model, os.path.join(d, "detector.onnx.zip"))
            with zipfile.ZipFile(dest) as zf:
                self.assertEqual(sorted(zf.namelist()), ["model.onnx", "model.onnx.data"])

    def test_the_pack_reader_finds_an_unpacked_onnx_model(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("model.onnx", "model.onnx.data"):
                with open(os.path.join(d, name), "wb") as fh:
                    fh.write(b"x")
            self.assertEqual(manifest.model_in(d), os.path.join(d, "model.onnx"))


@unittest.skipUnless(HAVE_ONNX, "needs onnx and onnxruntime")
class SaveCheckedTests(unittest.TestCase):
    """A converted model reaches its target only after its check passes on
    the saved file, so a build never zips one that failed."""

    def test_a_model_that_fails_its_check_is_not_left_behind(self):
        model, _ = _matmul_model()

        def drifted(path):
            raise SystemExit("converted model differs from the original")

        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "model.onnx")
            with open(target, "wb") as fh:
                fh.write(b"an older conversion")
            with self.assertRaisesRegex(SystemExit, "differs"):
                onnx_models.save_checked(model, target, drifted)
            self.assertEqual(os.listdir(d), [])

    def test_a_model_that_passes_is_saved_at_its_target_after_its_check(self):
        model, w = _matmul_model()
        model = onnx_models.weights_to_float16(model)
        checked = []

        def check(path):
            x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
            (got,) = onnx_models.run(path, x)
            np.testing.assert_allclose(got, x @ w)
            checked.append(os.path.exists(target))

        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "model.onnx")
            self.assertEqual(onnx_models.save_checked(model, target, check), target)
            self.assertEqual(checked, [False], "checked before it reached the target")
            self.assertEqual(os.listdir(d), ["model.onnx"])
            onnx.checker.check_model(target)

@unittest.skipUnless(HAVE_ONNX, "needs onnx and onnxruntime")
class OutOfRangeWeightsTests(unittest.TestCase):
    def test_a_weight_beyond_16_bits_stays_32_bit(self):
        model, w = _matmul_model()
        model.graph.initializer[0].CopyFrom(numpy_helper.from_array(w * 1e5, "W"))
        converted = onnx_models.weights_to_float16(model)
        kept = {i.name: i.data_type for i in converted.graph.initializer}
        self.assertEqual(kept, {"W": TensorProto.FLOAT})
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.onnx")
            onnx.save(converted, path)
            (got,) = onnx_models.run(path, x)
        np.testing.assert_allclose(got, x @ (w * 1e5), rtol=1e-6)


class ExportedModelTests(unittest.TestCase):
    def test_a_folder_gives_the_model_just_exported(self):
        with tempfile.TemporaryDirectory() as d:
            for name, when in (("a_old.onnx", 1_000), ("inference_model.onnx", 2_000)):
                path = os.path.join(d, name)
                with open(path, "wb") as fh:
                    fh.write(b"x")
                os.utime(path, (when, when))
            self.assertEqual(onnx_models.exported_model(d), os.path.join(d, "inference_model.onnx"))

    def test_a_file_is_itself_and_an_empty_folder_stops(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(onnx_models.exported_model(os.path.join(d, "m.onnx")), os.path.join(d, "m.onnx"))
            with self.assertRaisesRegex(SystemExit, "no ONNX model"):
                onnx_models.exported_model(d)


if __name__ == "__main__":
    unittest.main()
