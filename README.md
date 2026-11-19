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

## Licence

The code is under the GNU General Public License v3.0, as the app is. See
[LICENSE](LICENSE). The models and data in each release keep their own terms,
listed in that release's notes.

## History

The pipeline moved here from `tool/species_pack/` in the app repository, with
its history (submersion-app/submersion#4769). Pull request numbers in older
commit messages refer to the app repository.
