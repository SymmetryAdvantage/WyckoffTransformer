"""Every argparse script must be able to print its own help.

Nothing else in the suite imports the files under `scripts/`, so a NameError, a missing
import or -- twice during this change -- an unescaped `%` in a help string reaches a user
as a crash on `--help`. argparse runs `help % params` when it formats, so a literal percent
has to be written `%%`; py_compile does not catch it and neither does reading the file.
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

#: The argparse-driven scripts. The files under scripts/diagnostics/ are deliberately
#: excluded: they are top-level scripts that do their work on import, so running them with
#: --help would run them.
SCRIPTS = (
    "scripts/build_lemat_bulk_fmax.py",
    "scripts/cache_a_dataset.py",
    "scripts/cache_a_dataset_reusing.py",
    "scripts/audit_ehull_conditioning.py",
    "scripts/tokenise_a_dataset.py",
    "scripts/train.py",
    "scripts/run_template_reconstruction_study.py",
    "scripts/analyse_template_protocol.py",
    "scripts/run_nep89_protocol_variants.py",
    "scripts/run_pyxtal_tolerance_sweep.py",
    "scripts/measure_prescreen_selector.py",
)


class TestScriptHelp(unittest.TestCase):
    def test_every_script_prints_help(self):
        for script in SCRIPTS:
            path = REPO / script
            with self.subTest(script=script):
                self.assertTrue(path.exists(), f"{script} is missing")
                result = subprocess.run(
                    [sys.executable, str(path), "--help"],
                    cwd=REPO, capture_output=True, text=True, timeout=300)
                self.assertEqual(
                    result.returncode, 0,
                    f"{script} --help failed:\n{result.stderr[-2000:]}")
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()
