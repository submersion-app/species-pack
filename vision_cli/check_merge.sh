#!/bin/sh
# Builds and runs merge_check.swift against the app's own
# VisionPipeline.swift: the pipeline must merge two detectors' boxes exactly
# as boxes.py does. macOS only (Core ML).
#
#   vision_cli/check_merge.sh [OUT_DIR]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
# The app's checkout: SUBMERSION_APP, or a clone beside this repository.
app=$(cd "${SUBMERSION_APP:-$here/../../submersion}" && pwd)
out=${1:-${TMPDIR:-/tmp}/species_merge_check}
mkdir -p "$out"
swiftc -O -parse-as-library -o "$out/merge_check" \
  "$here/merge_check.swift" "$here/FlutterShim.swift" \
  "$app/packages/submersion_vision/darwin/Classes/VisionPipeline.swift"
"$out/merge_check"
