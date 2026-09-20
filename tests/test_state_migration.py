from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flybit.state import STATE_SCHEMA, load_state


class StateMigrationTest(unittest.TestCase):
    def test_clean_and_schema_six_startup(self):
        with tempfile.TemporaryDirectory() as tmp, patch("flybit.state.state_dir", return_value=Path(tmp)):
            clean = load_state()
            self.assertEqual(clean.schema, STATE_SCHEMA)
            old = {"schema": 6, "created_at": clean.created_at, "x": 123, "y": 45,
                   "unknown_transient_field": [1] * 100}
            (Path(tmp) / "state.json").write_text(json.dumps(old))
            migrated = load_state()
            self.assertEqual(migrated.schema, STATE_SCHEMA)
            self.assertEqual(migrated.x, 123)
            persisted = json.loads((Path(tmp) / "state.json").read_text())
            self.assertNotIn("unknown_transient_field", persisted)
            self.assertFalse((Path(tmp) / "state.json.tmp").exists())

    def test_corrupt_state_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp, patch("flybit.state.state_dir", return_value=Path(tmp)):
            (Path(tmp) / "state.json").write_text("broken")
            state = load_state()
            self.assertEqual(state.schema, STATE_SCHEMA)
            self.assertTrue((Path(tmp) / "state.json.corrupt").exists())

if __name__ == "__main__": unittest.main()
