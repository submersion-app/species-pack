#!/usr/bin/env python3
"""Gates a pack's ONNX engine against its Core ML engine (piece 3 of marine
life identification): the app on Android, Windows and Linux must name the
test set as well as on Apple devices, at the pack's own bars. Every pack
passes it before it is published.

    python -I score.py --pack PACK --engine coreml --naming-from PACK --work WORK
    python -I score.py --pack PACK --engine onnx --naming-from PACK --work WORK
    python -I engine_gate.py --work WORK --pack PACK

Four checks, on the held-out photos:

1. the pack's bars stand held out on the ONNX engine's results (none is
   switched off by the held-out gate);
2. the ONNX engine names right no fewer photos than Core ML (paired
   McNemar test, p < 0.05 counts as fewer);
3. no group is significantly worse, by the same test;
4. junk is named within the pack's junk limit, and not significantly more
   often than by Core ML.

A failure is fixed in the engine (its resizing, its precision), never in
the bars. Writes engine_gate.json to the ONNX results folder; exits 1 on a
failure. Standard library only.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bars as bar_search  # noqa: E402
import gate  # noqa: E402

ALPHA = gate.ALPHA
JUNK_LIMIT = gate.JUNK_LIMIT
# The share of held-out junk crops only one engine's run found (its box
# moved past the 0.9 overlap), left out of the junk pairing, above which the
# junk check fails: the ones left out could hide junk named.
UNPAIRED_JUNK_SHARE = 0.05


def _paired(first, second, ids):
    """Photos named right only by `first`, only by `second`, and McNemar's p."""
    only_first = sum(1 for i in ids if first[i]["named_right"] and not second[i]["named_right"])
    only_second = sum(1 for i in ids if second[i]["named_right"] and not first[i]["named_right"])
    return only_first, only_second, bar_search.mcnemar_p(only_first, only_second)


def pack_bars_hold(onnx_eval):
    held = onnx_eval.get("fixed bars held out")
    if held is None:
        return {"pass": False, "why": "the ONNX run was not scored at the pack's bars (score.py --naming-from PACK)"}
    return {"pass": not held["switched off"], "switched off": held["switched off"]}


def not_fewer_named_right(coreml, onnx):
    ids = sorted(i for i, p in onnx.items() if p["held_out"] and p["sample"] == "natural" and i in coreml)
    only_coreml, only_onnx, p = _paired(coreml, onnx, ids)
    return {
        "pass": not (p < ALPHA and only_coreml > only_onnx),
        "photos": len(ids),
        "core ml named right": sum(1 for i in ids if coreml[i]["named_right"]),
        "onnx named right": sum(1 for i in ids if onnx[i]["named_right"]),
        "only core ml": only_coreml,
        "only onnx": only_onnx,
        "p": p,
    }


def no_group_worse(coreml, onnx):
    groups = {}
    for i, p in onnx.items():
        if p["held_out"] and i in coreml:
            groups.setdefault(p["group"], []).append(i)
    by_group, worse = {}, []
    for group, ids in sorted(groups.items()):
        only_coreml, only_onnx, p = _paired(coreml, onnx, sorted(ids))
        by_group[group] = {"photos": len(ids), "only core ml": only_coreml, "only onnx": only_onnx, "p": p}
        if p < ALPHA and only_coreml > only_onnx:
            worse.append(group)
    return {"pass": not worse, "worse": worse, "by group": by_group}


def junk(onnx_eval, coreml_junk, onnx_junk):
    """The ONNX engine names junk no more than the limit, and not more often
    than Core ML on the same held-out junk crops (paired). A junk crop is
    paired when both runs found its labelled box; one only one run found is
    left out of the pairing, and the check fails when more than
    UNPAIRED_JUNK_SHARE of the crops are, or none are paired, since the
    left-out ones could hide junk named."""
    rate = onnx_eval["junk named (tune, at the chosen bars)"]
    keys = sorted(set(coreml_junk) & set(onnx_junk))
    unpaired = len(set(coreml_junk) ^ set(onnx_junk))
    every = len(set(coreml_junk) | set(onnx_junk))
    only_onnx = sum(1 for k in keys if onnx_junk[k] and not coreml_junk[k])
    only_coreml = sum(1 for k in keys if coreml_junk[k] and not onnx_junk[k])
    p = bar_search.mcnemar_p(only_onnx, only_coreml)
    more = p < ALPHA and only_onnx > only_coreml
    paired_enough = bool(keys) and unpaired <= UNPAIRED_JUNK_SHARE * every
    return {
        "pass": rate <= JUNK_LIMIT and not more and paired_enough,
        "junk named": rate,
        "limit": JUNK_LIMIT,
        "held-out crops": len(keys),
        "only in one run": unpaired,
        "only onnx": only_onnx,
        "only core ml": only_coreml,
        "p": p,
    }


def confirm_pack_bars(manifest, coreml_eval, onnx_eval):
    """Stops unless both runs named at the pack's own bars: the ONNX run
    fixed at them (score.py --naming-from PACK), the Core ML run the one
    that measured them or fixed at them too. The pairings are fair only at
    one set of bars."""
    published = bar_search.published_bars(manifest["naming"])
    if published is None:
        raise SystemExit("the pack's bars are provisional: measure them on Core ML first, then score ONNX at them")
    want = published.__dict__
    for label, evaluation in (("Core ML", coreml_eval), ("ONNX", onnx_eval)):
        named_at = evaluation["bars after held-out gate"]
        if any(abs(named_at[k] - want[k]) > 1e-9 for k in want):
            raise SystemExit(
                f"the {label} run was named at bars {named_at}, not the pack's {want}; "
                "score it with --naming-from PACK"
            )


def check(coreml_per_photo, onnx_per_photo, onnx_eval, coreml_junk, onnx_junk):
    """The four checks; `pass` only when all pass."""
    result = {
        "pack bars hold": pack_bars_hold(onnx_eval),
        "not fewer named right": not_fewer_named_right(coreml_per_photo, onnx_per_photo),
        "no group worse": no_group_worse(coreml_per_photo, onnx_per_photo),
        "junk": junk(onnx_eval, coreml_junk, onnx_junk),
    }
    result["pass"] = all(r["pass"] for r in result.values())
    return result


def _read(folder, name):
    with open(os.path.join(folder, name), encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work", required=True, help="score.py's work folder: Core ML results, ONNX ones in onnx/")
    parser.add_argument("--pack", required=True, help="the pack both runs scored")
    parser.add_argument("--set-key", default=gate.SET_KEY, help="the detector set and fitting the pack names with")
    args = parser.parse_args(argv)
    folders = {"coreml": args.work, "onnx": os.path.join(args.work, "onnx")}
    manifest = _read(args.pack, "manifest.json")
    results, per_photo, per_junk = {}, {}, {}
    for engine, folder in folders.items():
        results[engine] = _read(folder, "results.json")
        gate.confirm_engine(results[engine], engine, folder)
        gate.confirm_scored(results[engine], manifest, engine)
        per_photo[engine] = _read(folder, "per_photo.json")[args.set_key]
        per_junk[engine] = _read(folder, "per_junk.json")[args.set_key]
    gate.confirm_same_photos(per_photo["coreml"], per_photo["onnx"], runs="Core ML and ONNX")
    confirm_pack_bars(
        manifest, results["coreml"]["evaluations"][args.set_key], results["onnx"]["evaluations"][args.set_key]
    )
    result = check(
        per_photo["coreml"],
        per_photo["onnx"],
        results["onnx"]["evaluations"][args.set_key],
        per_junk["coreml"],
        per_junk["onnx"],
    )
    with open(os.path.join(folders["onnx"], "engine_gate.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
