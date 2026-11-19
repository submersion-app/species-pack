#!/usr/bin/env python3
"""Unit tests for app_repo.py."""

import os
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import app_repo  # noqa: E402


class AppRootTest(unittest.TestCase):
    def test_a_named_checkout_is_used(self):
        with tempfile.TemporaryDirectory() as app:
            open(os.path.join(app, "pubspec.yaml"), "w").close()
            with mock.patch.dict(os.environ, {"SUBMERSION_APP": app}):
                self.assertEqual(app_repo.app_root(), os.path.abspath(app))
                self.assertEqual(app_repo.app_path("a", "b"), os.path.join(os.path.abspath(app), "a", "b"))

    def test_a_folder_that_is_not_the_app_stops_with_a_message(self):
        with tempfile.TemporaryDirectory() as empty:
            with mock.patch.dict(os.environ, {"SUBMERSION_APP": empty}):
                with self.assertRaises(SystemExit) as stop:
                    app_repo.app_root()
        self.assertIn("SUBMERSION_APP", str(stop.exception))

    def test_without_a_name_it_looks_beside_this_repository(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(os.path, "isfile", return_value=True):
            self.assertEqual(app_repo.app_root(), os.path.abspath(os.path.join(_HERE, "..", "submersion")))


if __name__ == "__main__":
    unittest.main()
