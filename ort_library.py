#!/usr/bin/env python3
"""ONNX Runtime's macOS library, for scoring the app's ONNX engine on a Mac
(score.py --engine onnx). The app ships ONNX Runtime 1.28 on Android,
Windows and Linux; its Mac build carries none, so the scorer takes the same
version's Mac library out of NuGet's Microsoft.ML.OnnxRuntime package,
checking the package and the library against pinned hashes. Standard
library only."""

import hashlib
import os
import urllib.request
import zipfile

NUPKG_URL = (
    "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/1.28.0/"
    "microsoft.ml.onnxruntime.1.28.0.nupkg"
)
NUPKG_SHA256 = "769d1d3ea8ab6cd69f737c9dd4d4462aa4ad0ccfa106eaf506efc40d7bead5db"
DYLIB_ENTRY = "runtimes/osx-arm64/native/libonnxruntime.dylib"
DYLIB_SHA256 = "244e53902258550e4b3afc3231c3cf8ba860173fc47a1fb572082ed822e6973f"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url, dest):
    request = urllib.request.Request(url, headers={"User-Agent": "species-pack-builder/0.1"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as fh:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)


def macos_library(work, *, fetch=_fetch, package_sha256=NUPKG_SHA256, dylib_sha256=DYLIB_SHA256):
    """The library's path in `<work>/onnxruntime/`, fetched and checked once."""
    folder = os.path.join(work, "onnxruntime")
    os.makedirs(folder, exist_ok=True)
    dylib = os.path.join(folder, "libonnxruntime.dylib")
    if os.path.exists(dylib) and _sha256(dylib) == dylib_sha256:
        return dylib
    package = os.path.join(folder, "onnxruntime.nupkg")
    if not (os.path.exists(package) and _sha256(package) == package_sha256):
        partial = package + ".part"
        fetch(NUPKG_URL, partial)
        os.replace(partial, package)
        if _sha256(package) != package_sha256:
            os.remove(package)
            raise SystemExit(f"{NUPKG_URL} does not match its pinned SHA-256")
    with zipfile.ZipFile(package) as zf:
        data = zf.read(DYLIB_ENTRY)
    if hashlib.sha256(data).hexdigest() != dylib_sha256:
        raise SystemExit(f"{DYLIB_ENTRY} in the package does not match its pinned SHA-256")
    with open(dylib + ".part", "wb") as fh:
        fh.write(data)
    os.replace(dylib + ".part", dylib)
    return dylib
