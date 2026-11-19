#!/usr/bin/env python3
"""Fails when a tracked text file holds an em-dash or an en-dash.

    python3.14 no_dashes.py

The repository's writing rule: neither is used as punctuation. A numeric
range uses a hyphen. Binary files, files git does not track, and data
written by others (the test set and the vectors in testdata/, which keep
the names and credits their sources give) are skipped.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DASHES = (chr(0x2014), chr(0x2013))  # em-dash, en-dash
EXEMPT = ("testdata/",)


def offending(paths: list[Path]) -> list[str]:
    """`path:line` for every line holding a dash, in the given files."""
    found = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(dash in line for dash in DASHES):
                found.append(f"{path.relative_to(ROOT)}:{number}")
    return found


def main() -> int:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    found = offending([ROOT / name for name in listed if not name.startswith(EXEMPT)])
    for place in found:
        print(f"dash: {place}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
