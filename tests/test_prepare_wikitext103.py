import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_wikitext103.py"
SPEC = importlib.util.spec_from_file_location("prepare_wikitext103", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareWikiTextTests(unittest.TestCase):
    def test_writes_rows_in_order_and_preserves_blank_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "train.txt"
            MODULE.write_split(
                [{"text": "first article line"}, {"text": ""}, {"text": "third"}],
                target,
                expected_rows=3, min_bytes=0,
            )
            self.assertEqual(target.read_text(encoding="utf-8"), "first article line\n\nthird\n")

    def test_rejects_wrong_split_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                MODULE.write_split([{"text": "line"}], Path(directory) / "train.txt", 2, min_bytes=0)

    def test_rejects_non_string_text(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "no text string"):
                MODULE.write_split([{"text": None}], Path(directory) / "train.txt", 1, min_bytes=0)


if __name__ == "__main__":
    unittest.main()
