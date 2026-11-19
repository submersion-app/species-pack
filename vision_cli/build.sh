#!/bin/sh
# Builds the accuracy scorer's command-line tool from the app's own
# VisionPipeline.swift, so the scorer measures exactly what ships.
#
#   vision_cli/build.sh build/species_accuracy_work/species_vision
set -eu
here=$(cd "$(dirname "$0")" && pwd)
# The app's checkout: SUBMERSION_APP, or a clone beside this repository.
app=$(cd "${SUBMERSION_APP:-$here/../../submersion}" && pwd)
out=${1:?usage: build.sh OUT}
set -- "$here/main.swift" "$here/FlutterShim.swift" "$app/packages/submersion_vision/darwin/Classes/VisionPipeline.swift"
# Nothing to do when the tool was built from these exact sources. The app's
# checkout can change with SUBMERSION_APP, and another checkout's file can be
# older yet different, so this compares contents, not times.
digest=$(cat "$@" | shasum -a 256 | cut -d ' ' -f 1)
stamp="$out.sources"
if [ -x "$out" ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$digest" ]; then
  exit 0
fi
mkdir -p "$(dirname "$out")"
swiftc -O -o "$out" "$@"
printf '%s\n' "$digest" > "$stamp"
