#!/usr/bin/env python3
"""Where this repository and the app's checkout are.

The pipeline builds the app's own VisionPipeline.swift into its vision tool,
so the scorer measures exactly what ships, and reads the app's species list
and findings. SUBMERSION_APP names the app's checkout; without it, the
pipeline looks for a clone called `submersion` beside this repository.

Standard library only.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")


def app_root():
    """The app's checkout, or SystemExit saying how to name one."""
    root = os.path.abspath(os.environ.get("SUBMERSION_APP") or os.path.join(HERE, "..", "submersion"))
    if not os.path.isfile(os.path.join(root, "pubspec.yaml")):
        raise SystemExit(f"no app checkout at {root}: clone submersion-app/submersion there, or set SUBMERSION_APP")
    return root


def app_path(*parts):
    """A path inside the app's checkout."""
    return os.path.join(app_root(), *parts)


VISION_PIPELINE = ("packages", "submersion_vision", "darwin", "Classes", "VisionPipeline.swift")
