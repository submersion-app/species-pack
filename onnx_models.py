#!/usr/bin/env python3
"""ONNX helpers for the pack's ONNX engine files (the app's Android, Windows
and Linux engine, piece 3 of marine life identification): 16-bit weight
storage, the zip layout the app unpacks, and the checks a conversion must
pass. Needs onnx, onnxruntime and numpy where a function says so."""

import os
import zipfile

# The model's name inside its zip; the app unpacks `<stem>.onnx.zip` into
# `<stem>/` and loads the `.onnx` file there.
MODEL_NAME = "model.onnx"

# The newest model IR version ONNX Runtime 1.28 (the app's pinned version)
# reads; a newer onnx package writes 14, which it refuses.
MAX_IR_VERSION = 13


def weights_to_float16(model):
    """`model` (an onnx ModelProto, changed in place and returned) with each
    32-bit float weight stored in 16 bits and cast back to 32 bits before
    use. The file is about half the size, like the Core ML models' 16-bit
    weights, while ONNX Runtime still computes in 32 bits on every
    processor (it folds the casts when it opens the model). Inputs and
    outputs stay 32-bit, and the IR version is capped at
    [MAX_IR_VERSION]."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    graph = model.graph
    graph_inputs = {i.name for i in graph.input}
    kept, casts = [], []
    for init in graph.initializer:
        if init.data_type != TensorProto.FLOAT or init.name in graph_inputs:
            kept.append(init)
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), f"{init.name}_f16")
        kept.append(half)
        casts.append(
            helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT, name=f"{init.name}_to_float")
        )
    del graph.initializer[:]
    graph.initializer.extend(kept)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    model.ir_version = min(model.ir_version, MAX_IR_VERSION)
    return model


def zip_onnx(model_path, dest_zip):
    """Zips the model at `model_path` as `model.onnx` at the top of
    `dest_zip`, with its external weights file (`<model>.data`) beside it
    when it has one."""
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, MODEL_NAME)
        data = model_path + ".data"
        if os.path.exists(data):
            zf.write(data, MODEL_NAME + ".data")
    return dest_zip


def run(path, x):
    """The model at `path` run on the processor with its first input set to
    `x`: every output, as numpy arrays."""
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: x})


def cosine(a, b):
    """The cosine similarity of two arrays, flattened."""
    import numpy as np

    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
