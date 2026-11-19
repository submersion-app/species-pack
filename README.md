# Submersion species pack

The model pack behind marine life identification in
[Submersion](https://github.com/submersion-app/submersion), a dive log app:
fish detectors and a species classifier as Core ML models, the candidate
species with their name embeddings, a table of where each species has been
recorded, and the confidence bars the app names a species at. The app
downloads the pack when a diver chooses Download in its settings;
identification then runs on the device.

This repository holds the pipeline that builds, scores and checks the pack.
Its releases are what the app downloads.

## Releases

Each pack is a release tagged with its version. The app reads the latest
release's manifest:

```
https://github.com/submersion-app/species-pack/releases/latest/download/manifest.json
```

and fetches each file the manifest lists from the same place, checking its
size and SHA-256. A pack under test is a pre-release, which the app never
sees. Publishing a pack makes it the latest release; rolling back marks an
older one latest.

Each release's notes list its parts' credits and licences.

## Running the pipeline

The pipeline builds the app's own `VisionPipeline.swift` into its vision
tool, so the scorer measures exactly what ships, and it reads the app's
species list. It needs a checkout of
[the app](https://github.com/submersion-app/submersion): a clone called
`submersion` beside this repository, or any checkout named with
`SUBMERSION_APP`. Work folders go in `build/`.

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
SUBMERSION_APP=../submersion .venv/bin/python -I score.py --pack PACK
```

The tests run with the standard library and numpy; the ones that need
torch skip without it:

```bash
for test in *_test.py; do SUBMERSION_APP=../submersion python3.14 -I "$test"; done
```

The developer guide is in the app repository, at
`docs/developer/species-identification.md`.

## Licence

The code is under the GNU General Public License v3.0, as the app is. See
[LICENSE](LICENSE). The models and data in each release keep their own terms,
listed in that release's notes.

## History

The pipeline moved here from `tool/species_pack/` in the app repository, with
its history (submersion-app/submersion#4769). Pull request numbers in older
commit messages refer to the app repository.
