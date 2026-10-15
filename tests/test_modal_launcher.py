"""Validate actual Modal decorator configuration without creating cloud resources."""

import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec("modal"), "Modal is not installed")
class LauncherRegistrationTests(unittest.TestCase):
    def test_both_images_register_without_network_or_compute(self):
        import modal
        path = Path(__file__).resolve().parents[1] / "scripts/run_modal.py"
        spec = importlib.util.spec_from_file_location("experiment_launcher", path)
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        volume = modal.Volume.from_name("offline-launch-validation")
        for stage in ("baselines", "digits"):
            with self.subTest(stage=stage):
                app, function = launcher.make_function(stage, volume, 600)
                self.assertIsNotNone(app)
                self.assertIsNotNone(function)
