#!/usr/bin/env python3
"""The test vectors this pipeline shares with the app (issue #4769).

naming.py and region_grid.py write testdata/naming_vectors.json and
testdata/region_vectors.json, and the app's Dart tests read their copies in
the app's tool/species_pack/testdata/. The Python and the Dart are only
checked against each other while the two copies are the same file, so this
fails until a regenerated file is copied to the app too. Needs the app's
checkout (app_repo.py).
"""

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import app_repo  # noqa: E402

SHARED = ("naming_vectors.json", "region_vectors.json")
APP_TESTDATA = ("tool", "species_pack", "testdata")


class SharedVectorsTest(unittest.TestCase):
    def test_the_app_tests_against_the_same_vectors(self):
        for name in SHARED:
            with self.subTest(name=name):
                theirs_path = app_repo.app_path(*APP_TESTDATA, name)
                self.assertTrue(os.path.exists(theirs_path), f"the app has no {theirs_path}")
                with open(os.path.join(_HERE, "testdata", name), "rb") as fh:
                    ours = fh.read()
                with open(theirs_path, "rb") as fh:
                    theirs = fh.read()
                self.assertTrue(ours == theirs, f"testdata/{name} differs from the app's; copy it to {theirs_path}")


if __name__ == "__main__":
    unittest.main()
