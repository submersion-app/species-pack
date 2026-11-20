#!/usr/bin/env python3
"""Scores a model pack against the accuracy test set, for each detector set
(the fish detector, Megalodon, both) and crop fitting, and chooses the
naming bars (the measured accuracy design,
docs/design/specs/2026-11-17-marine-life-identification-measured-accuracy-design.md,
and the two detectors design,
docs/design/specs/2026-11-17-marine-life-two-detectors-design.md).

    python -I score.py --pack PACK

Runs the app's own vision pipeline over every photo (the Swift vision_cli
for Core ML; with --engine onnx, the app's Dart ONNX pipeline through its
scoring runner, results in <work>/onnx), the pair through the pipeline's own merge, which must equal
boxes.py's on every photo; names each crop with naming.py, chooses bars
with bars.py on the tune half of the natural sample, gates them on the
held-out half, and writes report.md and results.json to the work folder.
The pack must carry the fish detector; Megalodon comes from the pack or,
when it carries none, from its converted model (megalodon.py). Needs numpy
and Pillow. Developer-run; not part of CI.
"""

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app_repo  # noqa: E402

import bars as bar_search  # noqa: E402
import boxes as box_math  # noqa: E402
import levels as level_search  # noqa: E402
import manifest as pack_manifest  # noqa: E402
import megalodon  # noqa: E402
import metrics  # noqa: E402
import naming  # noqa: E402
import region_grid  # noqa: E402
import report  # noqa: E402
import testset  # noqa: E402

DEFAULT_WORK = os.path.join(app_repo.BUILD, "species_accuracy_work")
FITS = ("padSquare", "centreCrop")
MEGALODON = "megalodon-2024"
BOTH = "both"


# Running the vision tool.


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


ENGINES = ("coreml", "onnx")

# The app's ONNX scoring runner, a Flutter test file that hosts the pipeline.
ONNX_RUNNER = ("tool", "species_vision_onnx", "species_vision_onnx_test.dart")

# The app's ONNX pipeline (piece 3), which an ONNX run measures in place of
# the Swift pipeline, and the runner that hosts it.
ONNX_SOURCES = (
    ("lib", "features", "species_id", "data", "engines", "onnx", "pipeline", "detector_reading.dart"),
    ("lib", "features", "species_id", "data", "engines", "onnx", "pipeline", "pixels.dart"),
    ("lib", "features", "species_id", "data", "engines", "onnx", "pipeline", "onnx_pipeline.dart"),
    ("lib", "features", "species_id", "data", "engines", "onnx", "flutter_image_decoder.dart"),
    ("lib", "features", "species_id", "data", "engines", "onnx", "onnx_worker.dart"),
    ("lib", "features", "species_id", "data", "engines", "onnx", "onnx_worker_core.dart"),
    ("lib", "features", "species_id", "data", "engines", "onnx", "ort_model_runner.dart"),
    ("lib", "features", "species_id", "data", "engines", "onnx", "scoring_line.dart"),
    # VisionModels.fromManifest and CropFit, and the manifest it reads.
    ("lib", "features", "species_id", "domain", "entities", "vision_analysis.dart"),
    ("lib", "features", "species_id", "domain", "entities", "species_pack_manifest.dart"),
    ("packages", "submersion_vision", "lib", "src", "ort", "ort_runtime.dart"),
    ("packages", "submersion_vision", "lib", "src", "ort", "ort_native.dart"),
    ("packages", "submersion_vision", "lib", "src", "ort", "ort_c_api.g.dart"),
    ONNX_RUNNER,
)


def source_files(engine):
    """The source files a run's output depends on, besides the models: the
    Swift vision tool and pipeline for Core ML, the app's Dart pipeline and
    its runner for ONNX."""
    here = os.path.dirname(os.path.abspath(__file__))
    if engine == "onnx":
        return [app_repo.app_path(*parts) for parts in ONNX_SOURCES]
    return [
        app_repo.app_path(*app_repo.VISION_PIPELINE),
        os.path.join(here, "vision_cli", "main.swift"),
        os.path.join(here, "vision_cli", "FlutterShim.swift"),
    ]


def results_dir(work, engine):
    """Where a run's results.json, per_photo.json and report go: the work
    folder for Core ML, as always, and `<work>/onnx` for ONNX, so the two
    sit side by side for engine_gate.py."""
    return work if engine == "coreml" else os.path.join(work, "onnx")


def run_file(work, name, fit, version, engine):
    """The JSON lines file one run writes."""
    suffix = "" if engine == "coreml" else "_onnx"
    return os.path.join(work, "runs", f"{name}_{fit}_{version}{suffix}.jsonl")


def onnx_request(*, manifest, detectors, classifier, fit, detector_boxes, photos, out, library):
    """What the app's ONNX scoring runner reads (its own file says how)."""
    return {
        "manifest": manifest,
        "detectors": list(detectors),
        "classifier": classifier,
        "cropFit": fit,
        "detectorBoxes": detector_boxes,
        "photos": list(photos),
        "out": out,
        "ortLibrary": library,
    }


class Runner:
    """Runs the vision tool once per detector set (the fish detector,
    Megalodon, both) and fitting, cached by pack version, and keeps each
    run's results by photo path. Models come from the pack when it carries
    them; Megalodon otherwise from its converted model in the work folder."""

    def __init__(self, pack, work, engine="coreml"):
        if engine not in ENGINES:
            raise SystemExit(f"unknown engine {engine}")
        self.pack, self.work, self.engine = pack, work, engine
        with open(os.path.join(pack, "manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.version = self.manifest["packVersion"]
        detectors, self.boxes = pack_manifest.pack_detectors(self.manifest)
        stems = [d["file"] for d in detectors] + ["classifier"]
        zips = {pack_manifest.model_zip(stem, engine) for stem in stems}
        # The models' own hashes, so a pack rebuilt under the same version
        # with other models is extracted again rather than read stale.
        self.model_hashes = {f["name"]: f["sha256"] for f in self.manifest["files"] if f["name"] in zips}
        self.models = os.path.join(work, "models", self.version, *(() if engine == "coreml" else ("onnx",)))
        source = os.path.join(self.models, "source.json")
        if os.path.exists(self.models) and _read_json(source) != self.model_hashes:
            shutil.rmtree(self.models)
        os.makedirs(self.models, exist_ok=True)
        for stem in stems:
            pack_manifest.unpack_model(pack, stem, os.path.join(self.models, stem), engine=engine)
        with open(source, "w", encoding="utf-8") as fh:
            json.dump(self.model_hashes, fh)
        self.classifier = pack_manifest.model_in(os.path.join(self.models, "classifier"))
        if engine == "coreml":
            self.tool = os.path.join(work, "species_vision")
            subprocess.run([os.path.join(os.path.dirname(__file__), "vision_cli", "build.sh"), self.tool], check=True)
            self.library = None
        else:
            import ort_library

            self.tool = None
            self.library = ort_library.macos_library(work)

        self.cfd = pack_manifest.DETECTORS["cfd"]["id"]
        self.mega = pack_manifest.DETECTORS["megalodon"]["id"]
        in_pack = {d["id"]: (d, os.path.join(self.models, d["file"])) for d in detectors}
        if self.cfd not in in_pack:
            raise SystemExit(f"{pack} does not carry the fish detector, which every comparison measures")
        if self.mega not in in_pack:
            # A copy of the converted package: the tool deletes what it compiles.
            converted = (
                megalodon.tool_model(work, MEGALODON) if engine == "coreml" else megalodon.onnx_model(work, MEGALODON)
            )
            in_pack[self.mega] = (dict(megalodon.SETTINGS), os.path.dirname(converted))
        # Megalodon's checkpoint pins its model when it is not in the pack.
        self.mega_from_pack = self.mega in {d["id"] for d in detectors}
        self.runs = {
            self.cfd: [self.cfd],
            self.mega: [self.mega],
            BOTH: [self.cfd, self.mega],
        }
        self.manifests = {}
        for name, ids in self.runs.items():
            run_manifest = pack_manifest.with_detectors(self.manifest, [in_pack[i][0] for i in ids], self.boxes)
            path = os.path.join(work, "manifests", self.version, f"{name}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(run_manifest, fh)
            self.manifests[name] = path
        self.model_paths = {i: pack_manifest.model_in(folder) for i, (_, folder) in in_pack.items()}
        missing = [i for i, path in self.model_paths.items() if path is None]
        if missing:
            raise SystemExit(f"no model found for {', '.join(missing)}")
        self.memory = {}

    def _fingerprint(self, name, fit, photos):
        """What a run's output depends on: the photos, the detectors' and
        classifier's settings, the fitting, the models, and the tool's
        source."""
        digest = hashlib.sha256()
        with open(self.manifests[name], encoding="utf-8") as fh:
            settings = json.load(fh)
        digest.update(
            json.dumps(
                [
                    sorted(photos),
                    settings["detectors"],
                    settings["boxes"],
                    settings["classifier"],
                    fit,
                    self.model_hashes,
                    None if self.mega_from_pack else megalodon.CHECKPOINTS[MEGALODON]["sha256"],
                    # The converted model itself, so converting again reruns.
                    None if self.mega_from_pack else self._megalodon_stamp(),
                    self.engine,
                    # The ONNX Runtime library the runner loads on this Mac.
                    self._library_pin(),
                ]
            ).encode()
        )
        here = os.path.dirname(os.path.abspath(__file__))
        for source in (os.path.join(here, "megalodon.py"), *source_files(self.engine)):
            with open(source, "rb") as fh:
                digest.update(fh.read())
        return digest.hexdigest()

    def _library_pin(self):
        if self.engine == "coreml":
            return None
        import ort_library

        return ort_library.DYLIB_SHA256

    def _megalodon_stamp(self):
        if self.engine == "coreml":
            return megalodon.converted_stamp(self.work, MEGALODON)
        path = megalodon.onnx_model(self.work, MEGALODON)
        return [os.path.getsize(path), os.path.getmtime(path)]

    def analyze(self, name, fit, photos, out, *, detector_boxes=False, timing=None):
        """Runs the vision tool over `photos` into the JSONL file `out`,
        uncached: written to `out`.partial, then renamed. `timing` keeps
        the tool's time and memory readout."""
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if self.engine == "onnx":
            self._analyze_onnx(name, fit, photos, out, detector_boxes=detector_boxes, timing=timing)
            return
        detectors = [arg for i in self.runs[name] for arg in ("--detector", self.model_paths[i])]
        extra = ["--detector-boxes"] if detector_boxes else []
        with open(out + ".partial", "w", encoding="utf-8") as fh, open(timing or os.devnull, "w", encoding="utf-8") as err:
            subprocess.run(
                [
                    "/usr/bin/time",
                    "-l",
                    self.tool,
                    "--manifest",
                    self.manifests[name],
                    *detectors,
                    "--classifier",
                    self.classifier,
                    "--crop-fit",
                    fit,
                    *extra,
                ],
                input="\n".join(photos) + "\n",
                stdout=fh,
                stderr=err,
                text=True,
                check=True,
            )
        os.replace(out + ".partial", out)
        # A compiled model may have replaced a package the paths named.
        self.model_paths = {i: pack_manifest.model_in(os.path.dirname(p)) for i, p in self.model_paths.items()}
        self.classifier = pack_manifest.model_in(os.path.dirname(self.classifier))

    def _analyze_onnx(self, name, fit, photos, out, *, detector_boxes, timing):
        """The app's own ONNX pipeline over `photos`, through its scoring
        runner under Flutter's test host, so photos are decoded as the app
        decodes them."""
        request_path = out + ".request.json"
        with open(request_path, "w", encoding="utf-8") as fh:
            json.dump(
                onnx_request(
                    manifest=self.manifests[name],
                    detectors=[self.model_paths[i] for i in self.runs[name]],
                    classifier=self.classifier,
                    fit=fit,
                    detector_boxes=detector_boxes,
                    photos=photos,
                    out=out + ".partial",
                    library=self.library,
                ),
                fh,
            )
        run_logged(
            ["flutter", "test", os.path.join(*ONNX_RUNNER)],
            cwd=app_repo.app_root(),
            env={**os.environ, "SPECIES_VISION_REQUEST": request_path},
            log=timing or out + ".runner.txt",
        )
        os.replace(out + ".partial", out)

    def run(self, name, fit, photos):
        out = run_file(self.work, name, fit, self.version, self.engine)
        mem = out[: -len(".jsonl")] + ".time.txt"
        key_path = out[: -len(".jsonl")] + ".key"
        key = self._fingerprint(name, fit, photos)
        stored = open(key_path, encoding="utf-8").read().strip() if os.path.exists(key_path) else None
        if not os.path.exists(out) or stored != key:
            # The pair's merge is checked against boxes.py on its own boxes.
            self.analyze(name, fit, photos, out, detector_boxes=name == BOTH, timing=mem)
            with open(key_path, "w", encoding="utf-8") as fh:
                fh.write(key + "\n")
        with open(mem, encoding="utf-8") as fh:
            self.memory[(name, fit)] = peak_memory(self.engine, fh.read())
        results = {}
        with open(out, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                results[r["path"]] = r
        # A photo with no result, or one the tool could not read, is not a
        # photo where the detector found nothing.
        failed = [p for p in photos if p not in results or "error" in results[p]]
        if failed:
            raise SystemExit(f"{name} {fit}: {len(failed)} photos have no usable result, for example {failed[:2]}")
        return results


# Naming.


class Namer:
    """The pack's candidates, text embeddings and region table, naming
    embeddings as the app does. Logits and region masks take fast paths
    that are checked against naming.py on the first crops."""

    def __init__(self, pack, manifest):
        import numpy as np

        self.np = np
        with open(os.path.join(pack, "candidates.json"), encoding="utf-8") as fh:
            self.candidates = json.load(fh)
        size = manifest["classifier"]["embeddingSize"]
        self.text = (
            np.fromfile(os.path.join(pack, "text_embeddings.f16"), dtype="<f2")
            .astype(np.float64)
            .reshape(len(self.candidates), size)
        )
        with open(os.path.join(pack, "regions.bin"), "rb") as fh:
            self.bits = fh.read()
        self.scale = manifest["classifier"]["logitScale"]
        self.config = manifest["naming"]
        self.cell_degrees = manifest["region"]["cellDegrees"]
        self.rings = manifest["region"]["neighbourCells"]
        self.index = {c["gbifKey"]: i for i, c in enumerate(self.candidates)}
        self.checked = 0

    @functools.lru_cache(maxsize=None)
    def _base_mask(self, cell):
        cells = region_grid.neighbourhood(cell, self.rings, self.cell_degrees)
        return tuple(region_grid.allowed_mask(self.bits, len(self.candidates), cells, self.cell_degrees))

    def _allowed(self, cell, overrides):
        if cell is None:
            return None
        mask = list(self._base_mask(cell))
        for i, allowed in overrides.items():
            mask[i] = allowed
        count = sum(mask)
        return mask if count and count >= self.config["minAllowedCandidates"] else None

    def name(self, embedding, cell, overrides):
        """(with location, without location) namings of one embedding."""
        e = self.np.asarray(embedding, dtype=self.np.float64)
        norm = self.np.linalg.norm(e)
        if not norm or not self.np.isfinite(norm):
            return None, None
        logits = (self.text @ (e / norm) * self.scale).tolist()
        allowed = self._allowed(cell, overrides)
        if self.checked < 20:
            slow = naming.logits(naming.unit(list(embedding)), self.text.tolist(), self.scale)
            if max(abs(a - b) for a, b in zip(slow, logits)) >= 1e-6:
                raise SystemExit("fast logits disagree with naming.py")
            slow_mask = naming.allowed_for(
                cell,
                self.bits,
                len(self.candidates),
                cell_degrees=self.cell_degrees,
                rings=self.rings,
                min_allowed=self.config["minAllowedCandidates"],
                overrides=overrides,
            )
            if slow_mask != allowed:
                raise SystemExit("fast region mask disagrees with naming.py")
            self.checked += 1
        top_k = self.config["topK"]
        return naming.name(logits, allowed, top_k), naming.name(logits, None, top_k)


# One crop's record.


def _crop_record(row, detection, namer, junk_cell=None):
    label = namer.index.get(int(row["gbif_key"])) if row["gbif_key"] else None
    if row["source"] == "commons":
        cell, overrides = junk_cell, {}
    else:
        cell = region_grid.cell_of(float(row["lat"]), float(row["lon"]), namer.cell_degrees)
        # A test photo must not vouch for its own species' place: with no
        # record before the cutoff, the label species is not allowed there.
        # It can only switch the pack's bit off, never on, as the app reads
        # the pack's table alone.
        overrides = {label: False} if label is not None and row["recorded_before_cutoff"] != "yes" else {}
    with_loc, no_loc = namer.name(detection["embedding"], cell, overrides)
    if with_loc is None:
        return None
    side = min(detection["pixelWidth"], detection["pixelHeight"])

    def scored(n):
        top = n.ranked[0] if n.ranked else (None, 0.0)
        return bar_search.Scored(
            open_score=n.open_score,
            top_score=top[1],
            correct=label is not None and top[0] == label,
            shorter_side=side,
            limited=n.location_limited,
        )

    def rank(n):
        order = [i for i, _ in n.ranked]
        return order.index(label) if label in order else None

    return {
        "row": row,
        "with": scored(with_loc),
        "without": scored(no_loc),
        "rank_with": rank(with_loc),
        "rank_without": rank(no_loc),
        "in_pack": label is not None,
        "side": side,
    }


def _key(row):
    return (row["source"], row["id"])


def _largest(detections):
    return max(detections, key=lambda d: d["pixelWidth"] * d["pixelHeight"]) if detections else None


# Scoring one detector and fitting.


def _photo(work, row):
    return os.path.join(work, "photos", f"{row['source']}_{row['id']}.jpg")


def _junk_records(name, results_by_photo, junk_rows, namer, work, fish_cells, strict=True):
    """Each junk row named from the detection it labels, found by its box
    (an intersection over union of 0.9 or more), and the keys of the rows
    no box matched. A single detector's Core ML run (`strict`) must find
    every one of its rows among its own boxes, or the junk check would pass
    on nothing. The rows were labelled on Core ML's boxes, so another
    engine's run leaves the ones it moved out, and engine_gate.py bounds
    how many. In a merged run a row may match a box from either detector,
    since the other detector's box can have replaced its own, and each
    merged box counts once however many rows match it."""
    records, missing, used = [], [], set()
    for row in sorted(junk_rows, key=lambda r: (r["id"], r["detector"], int(r["box"]))):
        if name != BOTH and row["detector"] != name:
            continue
        key = f"{row['detector']}:{row['id']}:{row['box']}"
        path = _photo(work, row)
        detections = [
            (i, d)
            for i, d in enumerate((results_by_photo.get(path) or {}).get("detections", []))
            if ((path, i) not in used if name == BOTH else d.get("source") == row["detector"])
        ]
        target = tuple(float(row[k]) for k in ("box_x", "box_y", "box_w", "box_h"))
        found = box_math.best_match(target, [(d["x"], d["y"], d["w"], d["h"]) for _, d in detections])
        if found is None:
            missing.append(key)
            continue
        index, detection = detections[found]
        used.add((path, index))
        cell = fish_cells[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(fish_cells)]
        record = _crop_record(row, detection, namer, junk_cell=cell)
        if record:
            records.append(record)
    if missing and name != BOTH and strict:
        raise SystemExit(f"{name}: {len(missing)} junk rows match no box in this run, for example {missing[:3]}")
    return records, missing


def peak_memory(engine, timing):
    """The run's peak memory from `/usr/bin/time -l`'s output, in bytes; None
    for an ONNX run, which is not timed that way: there the measured process
    would be the flutter tool, not the test host running the models."""
    if engine != "coreml":
        return None
    m = re.search(r"(\d+)\s+maximum resident set size", timing)
    return int(m.group(1)) if m else None


def run_logged(command, *, cwd, env, log):
    """Runs `command` with its output in `log`; when it fails, stops with the
    output's last lines, so a failing runner says why."""
    with open(log, "w", encoding="utf-8") as fh:
        try:
            done = subprocess.run(command, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            raise SystemExit(f"{command[0]} was not found on PATH") from None
    if done.returncode != 0:
        with open(log, encoding="utf-8", errors="replace") as fh:
            tail = "".join(fh.readlines()[-30:])
        raise SystemExit(f"{' '.join(command)} failed ({done.returncode}); its output, in {log}:\n{tail}")


def junk_named_by_row(held_junk, junk_named):
    """Each held-out junk crop's row key and whether it was named, for
    pairing two engines' junk (engine_gate.py)."""
    return {
        f"{c['row']['detector']}:{c['row']['id']}:{c['row']['box']}": bool(named)
        for c, named in zip(held_junk, junk_named, strict=True)
    }


def fixed_bars_held_out(fixed_bars, held_nat, *, min_side):
    """Whether fixed bars still stand held out on this run (the engine
    gate's first check), or None when the bars were chosen, not fixed."""
    if fixed_bars is None:
        return None
    _, switched_off = bar_search.held_out_gate(
        fixed_bars, [c["with"] for c in held_nat], [c["without"] for c in held_nat], min_side=min_side
    )
    return {"switched off": switched_off}


def absolute_paths(args):
    """Makes the pack and work folders absolute, in place: the ONNX runner
    runs in the app's folder, so a relative path would name another one."""
    args.pack = os.path.abspath(args.pack)
    args.work = os.path.abspath(args.work)


def evaluate(
    name,
    results_by_photo,
    animals,
    junk_rows,
    namer,
    work,
    min_side,
    tune,
    fixed_bars=None,
    published=None,
    strict_junk=True,
):
    """Every number for one detector (or merged pair) and fitting.
    `results_by_photo` maps a photo path to its detections, each tagged
    with the detector (`source`) that found it. With `fixed_bars`, those
    bars are used as given instead of being chosen and gated (the
    rare-species check names at the shipped layer's bars)."""
    natural = [r for r in animals if r["sample"] == "natural"]
    if not natural:
        raise SystemExit("the test set has no underwater photos in its natural sample")
    fish_cells = [region_grid.cell_of(float(r["lat"]), float(r["lon"]), namer.cell_degrees) for r in natural]

    crops = {}
    any_box = {}
    for row in animals:
        detections = (results_by_photo.get(_photo(work, row)) or {}).get("detections", [])
        any_box[row["id"]] = bool(detections)
        biggest = _largest(detections)
        if biggest is not None:
            record = _crop_record(row, biggest, namer)
            if record:
                crops[row["id"]] = record

    junk, junk_unmatched = _junk_records(name, results_by_photo, junk_rows, namer, work, fish_cells, strict=strict_junk)

    def half(records, want_tune):
        return [c for c in records if (_key(c["row"]) in tune) == want_tune]

    nat = [crops[r["id"]] for r in natural if r["id"] in crops]
    tune_nat, held_nat = half(nat, True), half(nat, False)
    tune_junk, held_junk = half(junk, True), half(junk, False)
    if not tune_junk:
        raise SystemExit(f"{name}: no junk crops in the tune half, so the junk check would test nothing")
    chosen = bar_search.choose_bars(
        [c["with"] for c in tune_nat],
        [c["without"] for c in tune_nat],
        [c["with"] for c in tune_junk],
        [c["without"] for c in tune_junk],
        min_side=min_side,
    )
    gated, switched_off = bar_search.held_out_gate(
        chosen, [c["with"] for c in held_nat], [c["without"] for c in held_nat], min_side=min_side
    )
    fixed_held_out = fixed_bars_held_out(fixed_bars, held_nat, min_side=min_side)
    if fixed_bars is not None:
        chosen, gated, switched_off = fixed_bars, fixed_bars, []

    # The naming levels (issue #4721), surest first, each chosen on the tune
    # half and kept only if it stands held out. They are measured under the
    # bars the pack ships with when it has measured ones (`published`), so
    # a run whose own bars come out slightly different never states an
    # accuracy the app does not name at; otherwise under this run's. Bars
    # given with `fixed_bars` come first: they are the ones asked for.
    surer = gated if fixed_bars is not None else published or gated
    levels, surer_precision, surer_held = level_search.measure(
        surer, tune_nat, tune_junk, held_nat, held_junk, min_side=min_side
    )

    def naming_rates(records, confident=False):
        out = {}
        for key, condition in (("with location", "with"), ("without location", "without")):
            scored = [c[condition] for c in records]
            named = bar_search.named(scored, gated, min_side=min_side, confident=confident)
            coverage, precision = metrics.coverage_precision(named, [s.correct for s in scored])
            out[key] = {"coverage": coverage.__dict__, "precision": precision.__dict__}
        return out

    def summary(records):
        return {
            "crops": len(records),
            "top1 with location": metrics.top_k([c["rank_with"] for c in records], 1),
            "top3 with location": metrics.top_k([c["rank_with"] for c in records], 3),
            "top1 without location": metrics.top_k([c["rank_without"] for c in records], 1),
            "top3 without location": metrics.top_k([c["rank_without"] for c in records], 3),
            "suggested": naming_rates(records),
            "confident": naming_rates(records, confident=True),
        }

    held_rows = [r for r in animals if _key(r) not in tune]
    held_all = [crops[r["id"]] for r in held_rows if r["id"] in crops]
    held_natural_rows = [r for r in held_rows if r["sample"] == "natural"]

    def recall(rows):
        found = sum(1 for r in rows if any_box.get(r["id"]))
        named_right = sum(1 for r in rows if r["id"] in crops and crops[r["id"]]["rank_with"] == 0)
        return {
            "any box": metrics.rate(found, len(rows)).__dict__,
            "largest box named right (top-1, no bars)": metrics.rate(named_right, len(rows)).__dict__,
        }

    def correct_at_bars(rows):
        """Per photo: whether its labelled animal was named right at the bars."""
        out = {}
        for r in rows:
            c = crops.get(r["id"])
            out[r["id"]] = bool(
                c and bar_search.named([c["with"]], gated, min_side=min_side)[0] and c["with"].correct
            )
        return out

    def per_photo(rows):
        """Per test photo, both halves: named right, named confidently right
        or wrong, and what the gate groups it by. Named at the bars the app
        names at: the pack's published bars when it has measured ones, so a
        rerun whose own bars differ slightly compares what ships."""
        out = {}
        for r in rows:
            c = crops.get(r["id"])
            named = bool(c and bar_search.named([c["with"]], surer, min_side=min_side)[0])
            sure = bool(c and bar_search.named([c["with"]], surer, min_side=min_side, confident=True)[0])
            right = bool(c and c["with"].correct)
            out[r["id"]] = {
                "named_right": named and right,
                "confident_right": sure and right,
                "confident_wrong": sure and not right,
                "group": r["group"],
                "sample": r["sample"],
                "held_out": _key(r) not in tune,
                "gbif_key": r["gbif_key"],
            }
        return out

    junk_named = [
        a or b
        for a, b in zip(
            bar_search.named([c["with"] for c in held_junk], gated, min_side=min_side),
            bar_search.named([c["without"] for c in held_junk], gated, min_side=min_side),
        )
    ]
    curve = []
    limited = [c["with"] for c in held_nat if c["with"].limited]
    for bar in bar_search.BAR_GRID[::5]:
        picked = [s for s in limited if s.shorter_side >= min_side and s.open_score >= gated.min_open and s.top_score >= bar]
        curve.append(
            {
                "suggestScore": bar,
                "coverage": len(picked) / len(limited) if limited else 0.0,
                "precision": sum(s.correct for s in picked) / len(picked) if picked else None,
            }
        )
    return {
        "detector": name,
        "bars chosen on tune": chosen.__dict__,
        "bars after held-out gate": gated.__dict__,
        "switched off": switched_off,
        "levels measured under": {"bars": surer.__dict__, "published": published is not None},
        # Held out, at the bars the levels sit under: the lower of the
        # precisions with and without a location (levels.shown_precision),
        # None when they name nothing held out: no precision to show.
        "surer precision": surer_precision,
        # Both sides at those bars, which the shown figure is taken from;
        # "held-out natural" is at this run's own bars, which can differ.
        "surer held-out": surer_held,
        "levels": levels,
        "held-out natural": summary(held_nat),
        "held-out natural, detection": {
            "photos": len(held_natural_rows),
            **recall(held_natural_rows),
            "largest box at least 64 px": metrics.rate(
                sum(1 for r in held_natural_rows if r["id"] in crops and crops[r["id"]]["side"] >= 64),
                len(held_natural_rows),
            ).__dict__,
        },
        "fallback to no location": metrics.rate(
            sum(1 for c in held_nat if not c["with"].limited), len(held_nat)
        ).__dict__,
        "junk named (tune, at the chosen bars)": bar_search.junk_rate(
            [c["with"] for c in tune_junk], [c["without"] for c in tune_junk], chosen, min_side=min_side
        ),
        "junk named (held-out)": metrics.rate(sum(junk_named), len(junk_named)).__dict__,
        "fixed bars held out": fixed_held_out,
        "junk crops": {"tune": len(tune_junk), "held-out": len(held_junk), "rows no box matched": len(junk_unmatched)},
        "by group (held-out, natural and top-up)": metrics.breakdown(
            held_rows, lambda r: r["group"], lambda rows: {**recall(rows), **summary([crops[r["id"]] for r in rows if r["id"] in crops])}
        ),
        "by region": metrics.breakdown(held_nat, lambda c: c["row"]["region"], summary),
        "by pack membership": metrics.breakdown(held_nat, lambda c: "in pack" if c["in_pack"] else "not in pack", summary),
        "by habitat": metrics.breakdown(held_nat, lambda c: c["row"]["habitat"] or "unknown", summary),
        "by box size": metrics.breakdown(held_nat, lambda c: metrics.size_bucket(c["side"]), summary),
        "curve": curve,
        "_correct_at_bars": correct_at_bars(held_natural_rows),
        "_suggested_correct": {
            c["row"]["id"]: bool(bar_search.named([c["with"]], gated, min_side=min_side)[0] and c["with"].correct)
            for c in held_nat
        },
        "_all": held_all,
        "_per_photo": per_photo(animals),
        # Held-out junk crops, by row: whether each was named, for pairing
        # two engines' junk (engine_gate.py).
        "_junk_per_photo": junk_named_by_row(held_junk, junk_named),
    }


def _timings(results):
    detect = [r["detectMs"] for r in results.values() if "detectMs" in r]
    embed = [d["embedMs"] for r in results.values() for d in r.get("detections", []) if "embedMs" in d]

    def p(values, q):
        return statistics.quantiles(values, n=10)[q - 1] if len(values) >= 10 else None

    return {
        "detect ms median": statistics.median(detect) if detect else None,
        "detect ms p90": p(detect, 9),
        "embed ms median": statistics.median(embed) if embed else None,
        "embed ms p90": p(embed, 9),
    }


def _labelled(results, detector):
    """Each detection tagged with the detector that found it."""
    return {
        path: dict(r, detections=[dict(d, source=detector) for d in r.get("detections", [])])
        for path, r in results.items()
    }


def _merged(a, b, merge_iou, max_boxes):
    """Two single runs merged in Python, as the first measurement did."""
    out = {}
    for path in set(a) | set(b):
        runs = [a.get(path, {}).get("detections", []), b.get(path, {}).get("detections", [])]
        out[path] = {"path": path, "detections": box_math.merge(runs, iou_threshold=merge_iou, max_boxes=max_boxes)}
    return out


def _detector_run(pair, index):
    """One detector's boxes from the pair run, shaped as a single run."""
    return {path: {"detections": r["detectorBoxes"][index]} for path, r in pair.items()}


def _same_boxes(a, b, tolerance=1e-4):
    """How many photos have the same boxes, in order, in runs a and b."""
    def rects(result):
        return [(d["x"], d["y"], d["w"], d["h"]) for d in (result or {}).get("detections", [])]

    return sum(
        1
        for path in a
        if len(rects(a[path])) == len(rects(b.get(path)))
        and all(max(abs(x - y) for x, y in zip(p, q)) <= tolerance for p, q in zip(rects(a[path]), rects(b.get(path))))
    )


def _measured_entry(evaluation, manifest, detector_ids, crop_fit):
    """A NAMING_MEASURED entry for manifest.py from a run's gated bars,
    with the precision measured at them and the naming levels under them
    (issue #4721)."""
    naming = manifest["naming"]
    # The bars the levels and precision were measured under: the published
    # pack's when it has measured ones, so the entry is always consistent.
    bars = evaluation["levels measured under"]["bars"]
    levels = [
        {
            "id": level["id"],
            "target": level["target"],
            "suggestScore": round(level["bars"]["suggest"], 2),
            "suggestScoreNoLocation": round(level["bars"]["suggest_no_location"], 2),
            # The lower of its held-out precisions with and without a
            # location, already rounded as a manifest carries it.
            "precision": level["precision"],
        }
        for level in evaluation["levels"]
    ]
    # A share in (0, 1] or none, as the app and level_problems take it: a
    # Surer that named only wrong crops has none to show.
    surer_precision = level_search.manifest_precision(evaluation["surer precision"])
    return {
        "naming": {
            "provisional": False,
            "minCropSide": naming["minCropSide"],
            "minOpenScore": round(bars["min_open"], 2),
            "suggestScore": round(bars["suggest"], 2),
            "confidentScore": round(bars["confident"], 2),
            "suggestScoreNoLocation": round(bars["suggest_no_location"], 2),
            "confidentScoreNoLocation": round(bars["confident_no_location"], 2),
            "minAllowedCandidates": naming["minAllowedCandidates"],
            "topK": naming["topK"],
            **({"precision": surer_precision} if surer_precision is not None else {}),
            "levels": levels,
        },
        "detectorIds": list(detector_ids),
        "classifierId": manifest["classifier"]["id"],
        "cropFit": crop_fit,
        "packVersion": manifest["packVersion"],
        "contents": pack_manifest.contents_of(manifest["files"]),
    }


def _paired(a, b):
    only_a = sum(1 for k in a if a[k] and not b.get(k))
    only_b = sum(1 for k in b if b[k] and not a.get(k))
    return {"only first": only_a, "only second": only_b, "p": bar_search.mcnemar_p(only_a, only_b)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pack", required=True)
    parser.add_argument("--set", default=os.path.join(os.path.dirname(__file__), "testdata", "accuracy_set.csv"))
    parser.add_argument("--work", default=DEFAULT_WORK)
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="coreml",
        help="the app engine to measure: Core ML (the Swift vision tool) or ONNX (the app's Dart pipeline)",
    )
    parser.add_argument(
        "--naming-from",
        metavar="PACK",
        help="name at this pack's bars instead of choosing them (the naming layer's rare-species check)",
    )
    args = parser.parse_args(argv)
    absolute_paths(args)
    fixed_bars = None
    if args.naming_from:
        with open(os.path.join(args.naming_from, "manifest.json"), encoding="utf-8") as fh:
            fixed_bars = bar_search.published_bars(json.load(fh)["naming"])
        if fixed_bars is None:
            raise SystemExit(f"{args.naming_from} has provisional bars; --naming-from needs measured ones")

    rows = testset.read_rows(args.set)
    animals = [r for r in rows if r["source"] == "inaturalist" and r["underwater"] == "yes"]
    junk_rows = [r for r in rows if r["source"] == "commons"]
    missing = sorted({_photo(args.work, r) for r in animals + junk_rows if not os.path.exists(_photo(args.work, r))})
    if missing:
        raise SystemExit(
            f"{len(missing)} test photos are not in {args.work}/photos, for example {missing[:2]}; "
            "fetch them with fetch_testset.py photos-from-csv"
        )
    # Keyed by source and id: an iNaturalist id and a Commons page id can
    # be the same number.
    tune = (
        {("inaturalist", i) for i in testset.tune_ids([r for r in animals if r["sample"] == "natural"])}
        | {("inaturalist", i) for i in testset.tune_ids([r for r in animals if r["sample"] == "top-up"])}
        | {("commons", i) for i in testset.tune_ids(junk_rows)}
    )

    runner = Runner(args.pack, args.work, args.engine)
    namer = Namer(args.pack, runner.manifest)
    min_side = runner.manifest["naming"]["minCropSide"]
    merge_iou, max_boxes = runner.boxes["mergeIoU"], runner.boxes["maxBoxes"]
    commons = sorted({_photo(args.work, r) for r in junk_rows})
    photos = [_photo(args.work, r) for r in animals] + commons
    cfd, mega = runner.cfd, runner.mega
    ids = {cfd: [cfd], mega: [mega], BOTH: [cfd, mega]}

    results = {
        "engine": args.engine,
        "pack": runner.version,
        "candidates": len(namer.candidates),
        "classifierId": runner.manifest["classifier"]["id"],
        "contents": pack_manifest.contents_of(runner.manifest["files"]),
        # Every file the run scored, models included, so gate.py can tell
        # these scores from a stale run or another pack's (issue #4724).
        "files": {f["name"]: f["sha256"] for f in runner.manifest["files"]},
        "evaluations": {},
        "speed": {},
        "pair agreement": {},
    }
    kept = {}
    for fit in FITS:
        runs = {}
        for name in (cfd, mega, BOTH):
            raw = runner.run(name, fit, photos)
            results["speed"][f"{name} {fit}"] = {
                **_timings(raw),
                "peak memory bytes": runner.memory.get((name, fit)),
            }
            if name == BOTH:
                # The app's merge must be the one measured, photo by photo.
                bad = box_math.disagreements(raw, iou_threshold=merge_iou, max_boxes=max_boxes)
                if bad:
                    raise SystemExit(
                        f"the app merged {len(bad)} of {len(raw)} photos' boxes unlike boxes.py, "
                        f"for example {bad[:3]}; fix the pipeline before trusting this run"
                    )
                runs[name] = raw
            else:
                runs[name] = _labelled(raw, name)
        # Every photo passed the check above. For information only: GPU runs
        # need not repeat scores bit for bit, so the pair run's boxes may
        # differ slightly from the single runs without anything being wrong.
        results["pair agreement"][fit] = {
            "photos checked against boxes.py": len(runs[BOTH]),
            "per-detector boxes equal to the single run": {
                name: _same_boxes(_detector_run(runs[BOTH], i), runs[name]) for i, name in enumerate((cfd, mega))
            },
            "merged boxes equal to a merge of the single runs": _same_boxes(
                runs[BOTH], _merged(runs[cfd], runs[mega], merge_iou, max_boxes)
            ),
        }
        for name, by_photo in runs.items():
            key = f"{name} {fit}"
            print(f"scoring {key}", flush=True)
            # The bars this exact pack ships with, when it has measured ones.
            published = bar_search.published_bars(
                pack_manifest.naming_for(
                    ids[name],
                    runner.manifest["classifier"]["id"],
                    fit,
                    pack_manifest.contents_of(runner.manifest["files"]),
                )
            )
            kept[key] = evaluate(
                name,
                by_photo,
                animals,
                junk_rows,
                namer,
                args.work,
                min_side,
                tune,
                fixed_bars=fixed_bars,
                published=published,
                strict_junk=args.engine == "coreml",
            )
            if (
                fixed_bars is None
                and published is not None
                and kept[key]["bars after held-out gate"] != published.__dict__
            ):
                print(
                    f"warning: {key} bars came out {kept[key]['bars after held-out gate']}, not the published "
                    f"{published.__dict__}; its levels are measured under the published bars",
                    flush=True,
                )

    for name in (cfd, mega, BOTH):
        results.setdefault("fitting comparison", {})[name] = _paired(
            kept[f"{name} centreCrop"]["_suggested_correct"], kept[f"{name} padSquare"]["_suggested_correct"]
        )
    detectors = {}
    for name in (cfd, mega, BOTH):
        better = (
            "centreCrop" if results["fitting comparison"][name]["p"] < 0.05
            and results["fitting comparison"][name]["only first"] > results["fitting comparison"][name]["only second"]
            else "padSquare"
        )
        detectors[name] = kept[f"{name} {better}"]["_correct_at_bars"]
        results.setdefault("fitting chosen", {})[name] = better
    # Bars pinned at a fitting the packs are not built with would never match
    # a pack: CROP_FIT in manifest.py has to follow the measurement.
    set_keys = {cfd: "cfd", mega: "megalodon", BOTH: "both"}
    results["fitting against the packs"] = pack_manifest.fitting_mismatches(
        {set_keys[name]: fit for name, fit in results["fitting chosen"].items()}
    )
    for name, m in results["fitting against the packs"].items():
        print(f"warning: {name} measured best with {m['chosen']}, but packs use {m['packs use']}; update CROP_FIT", flush=True)
    results["detector comparison"] = {
        f"{a} vs {b}": _paired(detectors[a], detectors[b]) for a, b in ((mega, cfd), (BOTH, cfd), (BOTH, mega))
    }
    for key, value in kept.items():
        results["evaluations"][key] = {k: v for k, v in value.items() if not k.startswith("_")}
    results["named right per detector"] = {name: sum(v.values()) for name, v in detectors.items()}
    # What NAMING_MEASURED in manifest.py needs for each set, at its chosen
    # fitting, to name this exact pack.
    results["measured"] = [
        _measured_entry(
            kept[f"{name} {results['fitting chosen'][name]}"],
            runner.manifest,
            ids[name],
            results["fitting chosen"][name],
        )
        for name in (cfd, mega, BOTH)
    ]
    # Simplest first: one explicitly licensed detector, then the other, then
    # the pair.
    preference = [mega, cfd, BOTH]
    junk_ok = {
        name: kept[f"{name} {results['fitting chosen'][name]}"]["junk named (tune, at the chosen bars)"] <= 0.02
        for name in preference
    }
    # Every group must be found often enough (issue #4691), so a set is not
    # chosen on a sample that is mostly fish.
    results["groups below floor"] = {
        name: metrics.groups_below_floor(
            {
                group: g["any box"]["value"]
                for group, g in kept[f"{name} {results['fitting chosen'][name]}"][
                    "by group (held-out, natural and top-up)"
                ].items()
            }
        )
        for name in preference
    }
    eligible = {name: junk_ok[name] and not results["groups below floor"][name] for name in preference}
    paired = {
        frozenset((a, b)): _paired(detectors[a], detectors[b])["p"]
        for a, b in ((mega, cfd), (BOTH, cfd), (BOTH, mega))
    }
    results["recommendation"] = metrics.recommend(results["named right per detector"], paired, preference, eligible)

    out_dir = results_dir(args.work, args.engine)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=str)
    # Per photo, for comparing two packs on the same photos (gate.py) or two
    # engines on the same pack (engine_gate.py).
    with open(os.path.join(out_dir, "per_photo.json"), "w", encoding="utf-8") as fh:
        json.dump({key: value["_per_photo"] for key, value in kept.items()}, fh)
    with open(os.path.join(out_dir, "per_junk.json"), "w", encoding="utf-8") as fh:
        json.dump({key: value["_junk_per_photo"] for key, value in kept.items()}, fh)
    report.write_report(results, os.path.join(out_dir, "report.md"))
    print(f"wrote {os.path.join(out_dir, 'results.json')} and report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
