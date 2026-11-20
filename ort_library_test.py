#!/usr/bin/env python3
"""Unit tests for ort_library.py: the Mac library is taken out of the pinned
package only when both hashes match. Standard library only."""

import hashlib
import importlib.util
import io
import os
import tempfile
import unittest
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ort_library", os.path.join(_HERE, "ort_library.py"))
ort_library = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ort_library)


def _package(dylib=b"dylib"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(ort_library.DYLIB_ENTRY, dylib)
    return buf.getvalue()


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def fetch_with(self, data):
        def fetch(url, dest):
            with open(dest, "wb") as fh:
                fh.write(data)

        return fetch

    def test_takes_the_library_out_of_a_package_that_matches(self):
        package = _package()
        path = ort_library.macos_library(
            self.tmp.name,
            fetch=self.fetch_with(package),
            package_sha256=hashlib.sha256(package).hexdigest(),
            dylib_sha256=hashlib.sha256(b"dylib").hexdigest(),
        )
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(), b"dylib")

    def test_refuses_a_package_that_does_not_match(self):
        with self.assertRaises(SystemExit):
            ort_library.macos_library(
                self.tmp.name,
                fetch=self.fetch_with(_package()),
                package_sha256="0" * 64,
                dylib_sha256=hashlib.sha256(b"dylib").hexdigest(),
            )

    def test_refuses_a_library_that_does_not_match(self):
        package = _package(b"other")
        with self.assertRaises(SystemExit):
            ort_library.macos_library(
                self.tmp.name,
                fetch=self.fetch_with(package),
                package_sha256=hashlib.sha256(package).hexdigest(),
                dylib_sha256=hashlib.sha256(b"dylib").hexdigest(),
            )

    def test_a_library_already_there_is_not_fetched_again(self):
        package = _package()
        kwargs = dict(
            package_sha256=hashlib.sha256(package).hexdigest(),
            dylib_sha256=hashlib.sha256(b"dylib").hexdigest(),
        )
        ort_library.macos_library(self.tmp.name, fetch=self.fetch_with(package), **kwargs)

        def no_fetch(url, dest):
            raise AssertionError("fetched again")

        ort_library.macos_library(self.tmp.name, fetch=no_fetch, **kwargs)


if __name__ == "__main__":
    unittest.main()
