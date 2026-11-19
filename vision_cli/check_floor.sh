#!/bin/sh
# Builds and runs floor_check.swift against the app's own MemoryReading.swift:
# the memory floor for marine life identification on iPhone and iPad must
# let 8 GB devices in and keep 6 GB devices out. Needs only Foundation and
# Darwin, so it runs wherever swiftc does.
#
#   vision_cli/check_floor.sh [OUT_DIR]
set -eu
here=$(cd "$(dirname "$0")" && pwd)
# The app's checkout: SUBMERSION_APP, or a clone beside this repository.
app=$(cd "${SUBMERSION_APP:-$here/../../submersion}" && pwd)
out=${1:-${TMPDIR:-/tmp}/species_floor_check}
mkdir -p "$out"
swiftc -O -parse-as-library -o "$out/floor_check" \
  "$here/floor_check.swift" \
  "$app/packages/submersion_vision/darwin/Classes/MemoryReading.swift"
"$out/floor_check"
