#!/usr/bin/env python3
"""Compares two runs of the app's ONNX engine on the same photos, crop by
crop: the Mac's ONNX Runtime (ARM64) against a PC's (x64), whose code the
gate does not measure (piece 3, section 6). Each crop's embedding must agree
(cosine 0.999 or more), and at least 99 in 100 crops must get the same top
name and the same suggestion outcome, named as the app names without a
place.

    python -I engine_compare.py --pack PACK REFERENCE.jsonl OTHER.jsonl

The runs are the app's scoring runner's JSON lines. Needs numpy.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import boxes as box_math  # noqa: E402
import naming  # noqa: E402

COSINE_FLOOR = 0.999
SAME_SHARE_FLOOR = 0.99
MATCH_IOU = 0.9


def _outcome(embedding, shorter_side, text_rows, scale, config):
    """The crop's top candidate and whether the app would suggest it, with
    no place."""
    import numpy as np

    vector = np.asarray(embedding, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        return None
    named = naming.name((text_rows @ (vector / norm) * scale).tolist(), None, config["topK"])
    if named is None:
        return None
    suggested = naming.suggestable(
        named,
        shorter_side,
        min_crop_side=config["minCropSide"],
        min_open_score=config["minOpenScore"],
        suggest_score=config["suggestScore"],
        suggest_score_no_location=config["suggestScoreNoLocation"],
    )
    return named.ranked[0][0], suggested


def compare(reference, other, text_rows, scale, config):
    """`reference` and `other` map a photo to its runner line."""
    import numpy as np

    crops = unmatched = below = same = 0
    lowest = 1.0
    for path in sorted(set(reference) & set(other)):
        ref = [d for d in reference[path].get("detections", [])]
        got = list(other[path].get("detections", []))
        used = set()
        for d in ref:
            candidates = [(i, g) for i, g in enumerate(got) if i not in used]
            found = box_math.best_match(
                (d["x"], d["y"], d["w"], d["h"]), [(g["x"], g["y"], g["w"], g["h"]) for _, g in candidates]
            )
            if found is None:
                unmatched += 1
                continue
            index, match = candidates[found]
            used.add(index)
            crops += 1
            a, b = np.asarray(d["embedding"], dtype=np.float64), np.asarray(match["embedding"], dtype=np.float64)
            cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            lowest = min(lowest, cosine)
            if cosine < COSINE_FLOOR:
                below += 1
            side = min(d["pixelWidth"], d["pixelHeight"])
            if _outcome(d["embedding"], side, text_rows, scale, config) == _outcome(
                match["embedding"], side, text_rows, scale, config
            ):
                same += 1
        unmatched += len(got) - len(used)
    total = crops + unmatched
    share = same / total if total else 1.0
    return {
        "pass": below == 0 and share >= SAME_SHARE_FLOOR and total > 0,
        "crops": crops,
        "unmatched": unmatched,
        "lowest cosine": lowest,
        "below cosine floor": below,
        "same outcome share": share,
    }


def _read(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if "error" not in row:
                out[row["path"]] = row
    return out


def main(argv=None):
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", required=True)
    parser.add_argument("reference")
    parser.add_argument("other")
    args = parser.parse_args(argv)
    with open(os.path.join(args.pack, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(args.pack, "candidates.json"), encoding="utf-8") as fh:
        count = len(json.load(fh))
    text = (
        np.fromfile(os.path.join(args.pack, "text_embeddings.f16"), dtype="<f2")
        .astype(np.float64)
        .reshape(count, manifest["classifier"]["embeddingSize"])
    )
    result = compare(
        _read(args.reference), _read(args.other), text, manifest["classifier"]["logitScale"], manifest["naming"]
    )
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
