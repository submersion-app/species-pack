#!/usr/bin/env python3
"""Builds a marine life model pack for the app (issue #4668).

    python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python build_pack.py --out build/species_pack [--detectors cfd|megalodon|both]

Writes `manifest.json` and the files the Core ML and ONNX engines need: one model
per detector (the Community Fish Detector, FathomNet's Megalodon, or both,
issue #4691), the classifier, and the naming files. GBIF responses and
converted models are kept in `--work`, so a second run is quick; Megalodon
is converted beforehand with `megalodon.py convert`. Developer-run only:
not part of the app build or CI.
"""

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app_repo  # noqa: E402

import candidates as cand  # noqa: E402
import manifest as mf  # noqa: E402
import megalodon  # noqa: E402
import region_grid  # noqa: E402


CLASSIFIER_CREDIT = {
    "name": "BioCLIP 2",
    "licence": "MIT",
    "text": "Imageomics Institute. Gu et al., BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning.",
}
GBIF_CREDIT = {
    "name": "GBIF occurrence and taxonomy data",
    "licence": "CC0 and CC BY 4.0",
    "text": "GBIF.org. Species names, and counts of where each species has been recorded.",
}
# Each detector's credits, in the pack only when the detector is.
DETECTOR_CREDITS = {
    "cfd": [
        {
            "name": "Community Fish Detector",
            "licence": "No licence stated; its authors leave the terms to the user",
            "text": (
                "RF-DETR detector trained on the Community Fish Detection Dataset. "
                "The authors' statement on its terms: "
                "https://github.com/filippovarini/community-fish-detector/issues/8"
            ),
        },
        {
            "name": "RF-DETR",
            "licence": "Apache-2.0",
            "text": "Roboflow.",
        },
    ],
    "megalodon": [
        {
            "name": "FathomNet Megalodon",
            "licence": "CC BY 4.0",
            "text": (
                "Monterey Bay Aquarium Research Institute (MBARI), FathomNet. Megalodon 2024 "
                "(YOLO11x), https://huggingface.co/FathomNet/megalodon-2024-yolov11, converted to "
                "16-bit Core ML. Trained with Ultralytics' code."
            ),
        },
    ],
}


def credits_for(detector_set):
    """The credits of a pack carrying `detector_set`."""
    detectors = [c for k in mf.DETECTOR_SETS[detector_set] for c in DETECTOR_CREDITS[k]]
    return [CLASSIFIER_CREDIT, *detectors, GBIF_CREDIT]


def default_version(now=None):
    """The pack version when none is given: the UTC date and time, so two
    builds on one day differ (devices compare versions to offer an update,
    and to name stored crops again)."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y.%m.%d.%H%M")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", default=os.path.join(app_repo.BUILD, "species_pack_work"))
    parser.add_argument("--detectors", choices=sorted(mf.DETECTOR_SETS), default="cfd")
    parser.add_argument(
        "--megalodon-work",
        default=megalodon.DEFAULT_WORK,
        help="the work folder megalodon.py converted Megalodon in",
    )
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--min-images", type=int, default=50, help="photographed human observations")
    parser.add_argument(
        "--min-survey-records",
        type=int,
        default=50,
        help="present human observations at 0 to 40 m, for species divers record but rarely photograph",
    )
    parser.add_argument(
        "--engines",
        type=lambda value: [e for e in value.split(",") if e],
        default=["coreml", "onnx"],
        help="the engines to build models for, comma separated: coreml, onnx (both by default; every published pack carries both)",
    )
    parser.add_argument("--version", default=default_version())
    parser.add_argument("--skip-models", action="store_true", help="reuse converted models in --work")
    return parser.parse_args(argv)


def model_files(args, *, models, megalodon_module):
    """Converts and zips each detector of the set and the classifier for
    each of `args.engines` into `args.out`: `[(path, engine)]`, detectors
    in pool order, then the classifier, per engine. A zip of a detector
    outside the set, left by an earlier build, is removed: it would be
    served with this pack's folder though its manifest does not list it."""
    import onnx_models

    unknown = set(args.engines) - {"coreml", "onnx"}
    if unknown or not args.engines:
        raise SystemExit(f"unknown engines: {', '.join(sorted(unknown)) or 'none given'}")
    detector_set = mf.DETECTOR_SETS[args.detectors]
    for engine in ("coreml", "onnx"):
        for detector in mf.DETECTORS.values():
            stale = os.path.join(args.out, mf.model_zip(detector["file"], engine))
            if (engine not in args.engines or detector["id"] not in mf.detector_ids(args.detectors)) and os.path.exists(stale):
                os.remove(stale)
        # The classifier of an engine this build leaves out is stale too.
        stale = os.path.join(args.out, mf.model_zip("classifier", engine))
        if engine not in args.engines and os.path.exists(stale):
            os.remove(stale)
    entries = []
    checkpoint = None
    if "coreml" in args.engines:
        classifier_pkg = os.path.join(args.work, "classifier.mlpackage")
        if not (args.skip_models and os.path.exists(classifier_pkg)):
            print("converting the classifier for Core ML", flush=True)
            classifier_pkg = models.convert_classifier(args.work)
        detector_pkg = os.path.join(args.work, "detector.mlpackage")
        for key in detector_set:
            if key == "cfd":
                if not (args.skip_models and os.path.exists(detector_pkg)):
                    print("converting the fish detector for Core ML", flush=True)
                    checkpoint = checkpoint or models.fetch_detector(args.work)
                    detector_pkg = models.convert_detector(checkpoint, args.work)
                package = detector_pkg
            else:
                folder = megalodon_module.converted_dir(args.megalodon_work, megalodon_module.PACK_CHECKPOINT)
                package = os.path.join(folder, "detector.mlpackage")
                if not os.path.exists(package):
                    raise SystemExit(f"{package} is missing: a pack needs the model package, not only a compiled model")
            target = os.path.join(args.out, mf.model_zip(mf.DETECTORS[key]["file"], "coreml"))
            entries.append((models.zip_model(package, target), "coreml"))
        classifier_zip = os.path.join(args.out, mf.model_zip("classifier", "coreml"))
        entries.append((models.zip_model(classifier_pkg, classifier_zip), "coreml"))
    if "onnx" in args.engines:
        classifier_onnx = os.path.join(args.work, "classifier_onnx", onnx_models.MODEL_NAME)
        if not (args.skip_models and os.path.exists(classifier_onnx)):
            print("converting the classifier for ONNX", flush=True)
            classifier_onnx = models.convert_classifier_onnx(args.work)
        for key in detector_set:
            if key == "cfd":
                detector_onnx = os.path.join(args.work, "detector_onnx", onnx_models.MODEL_NAME)
                if not (args.skip_models and os.path.exists(detector_onnx)):
                    print("converting the fish detector for ONNX", flush=True)
                    checkpoint = checkpoint or models.fetch_detector(args.work)
                    detector_onnx = models.convert_detector_onnx(checkpoint, args.work)
            else:
                detector_onnx = megalodon_module.onnx_model(args.megalodon_work, megalodon_module.PACK_CHECKPOINT)
            target = os.path.join(args.out, mf.model_zip(mf.DETECTORS[key]["file"], "onnx"))
            entries.append((onnx_models.zip_onnx(detector_onnx, target), "onnx"))
        classifier_zip = os.path.join(args.out, mf.model_zip("classifier", "onnx"))
        entries.append((onnx_models.zip_onnx(classifier_onnx, classifier_zip), "onnx"))
    return entries


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.work, exist_ok=True)

    import models

    # 1 and 2: models, one per detector, then the classifier, for each engine.
    model_entries = model_files(args, models=models, megalodon_module=megalodon)

    # 3: candidates.
    with open(app_repo.app_path("assets", "data", "species.json"), encoding="utf-8") as fh:
        catalogue = json.load(fh)["species"]
    with open(app_repo.app_path("assets", "data", "species_gbif_keys.json"), encoding="utf-8") as fh:
        keys = json.load(fh)
    rows = cand.catalogue_candidates(catalogue, keys["speciesKeys"])
    have = {r["gbifKey"] for r in rows}
    print(f"catalogue candidates: {len(rows)}", flush=True)
    with ThreadPoolExecutor(8) as pool:
        catalogue_records = list(pool.map(lambda r: cand.fetch_species(r["gbifKey"], args.work), rows))
    orders = cand.expansion_order_keys(
        [(r["category"], rec) for r, rec in zip(rows, catalogue_records)]
    )
    print(f"widening within {len(orders)} orders", flush=True)
    photographed = cand.fetch_photographed_species(sorted(orders), args.work, min_images=args.min_images)
    surveyed = cand.fetch_surveyed_species(sorted(orders), args.work, min_records=args.min_survey_records)
    print(f"photographed species: {len(photographed)}, surveyed species: {len(surveyed)}", flush=True)
    wanted = cand.wanted_keys(photographed, surveyed, have, args.max_candidates - len(rows))
    photographed_keys = {k for k, _ in photographed}
    print(f"fetching {len(wanted)} GBIF species", flush=True)
    with ThreadPoolExecutor(8) as pool:
        records = list(pool.map(lambda k: cand.fetch_species(k, args.work), wanted))
    for record in records:
        row = cand.gbif_candidate(record)
        if row and row["gbifKey"] not in have:
            have.add(row["gbifKey"])
            rows.append(row)
    survey_only = sum(1 for r in rows if "catalogId" not in r and r["gbifKey"] not in photographed_keys)
    print(f"candidates: {len(rows)}, of them from surveys only: {survey_only}", flush=True)
    candidates_path = os.path.join(args.out, "candidates.json")
    with open(candidates_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False)

    # 4: text embeddings.
    print("embedding names", flush=True)
    embeddings = models.text_embeddings(rows)
    embeddings_path = os.path.join(args.out, "text_embeddings.f16")
    embeddings.astype("<f2").tofile(embeddings_path)

    # 5: regions.
    import regions

    cell_degrees = mf.REGION["cellDegrees"]
    print("mapping regions", flush=True)
    with ThreadPoolExecutor(8) as pool:
        cells = list(pool.map(lambda r: regions.cells_for(r["gbifKey"], cell_degrees, args.work), rows))
    regions_path = os.path.join(args.out, "regions.bin")
    with open(regions_path, "wb") as fh:
        fh.write(region_grid.pack_bits([sorted(c) for c in cells], cell_degrees))
    print(f"species with no recorded place: {sum(1 for c in cells if not c)}", flush=True)

    # 6: manifest.
    files = [
        *(mf.file_entry(path, engine) for path, engine in model_entries),
        mf.file_entry(candidates_path, "all"),
        mf.file_entry(embeddings_path, "all"),
        mf.file_entry(regions_path, "all"),
    ]
    manifest = mf.build_manifest(
        pack_version=args.version,
        detector_set=args.detectors,
        classifier_id=models.CLASSIFIER_ID,
        embedding_size=models.EMBEDDING_SIZE,
        files=files,
        credits=credits_for(args.detectors),
    )
    mf.write_manifest(os.path.join(args.out, "manifest.json"), manifest)
    total = sum(f["size"] for f in files)
    print(f"pack {args.version} ({args.detectors}): {len(rows)} candidates, {total / 1e6:.0f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
