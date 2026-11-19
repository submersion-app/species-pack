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
sources="$here/main.swift $here/FlutterShim.swift $app/packages/submersion_vision/darwin/Classes/VisionPipeline.swift"
# Nothing to do when the tool is newer than every source it is built from.
if [ -x "$out" ]; then
  stale=0
  for source in $sources; do
    [ "$source" -nt "$out" ] && stale=1
  done
  [ "$stale" -eq 0 ] && exit 0
fi
mkdir -p "$(dirname "$out")"
# shellcheck disable=SC2086
swiftc -O -o "$out" $sources
