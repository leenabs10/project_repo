"""Deployment relocation must preserve raw-data protection and source identity."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.core.paths import raw_data_roots, resolve_legacy_path, workspace_root


class WorkspacePathsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "source"
        self.workspace = Path(self.temporary.name) / "artifacts"
        (self.project / "configs").mkdir(parents=True)

    def configure(self):
        (self.project / "configs/local.json").write_text(
            json.dumps({"workspace_root": str(self.workspace)})
        )

    def test_unconfigured_fixture_retains_legacy_layout(self):
        self.assertEqual(workspace_root(self.project), self.project.resolve())
        self.assertEqual(
            resolve_legacy_path("reports/result.json", self.project),
            self.project / "reports/result.json",
        )

    def test_relocated_assets_use_the_specific_prefix(self):
        self.configure()
        self.assertEqual(
            resolve_legacy_path(
                self.project / ".tools/scene_assets/library/file.png", self.project
            ),
            self.workspace / "assets/scene_assets/library/file.png",
        )
        self.assertEqual(
            resolve_legacy_path(self.project / ".tools/old_operator.py", self.project),
            self.workspace / "archive/tools/old_operator.py",
        )

    def test_both_raw_locations_are_protected(self):
        self.configure()
        self.assertEqual(
            set(raw_data_roots(self.project)),
            {self.project / "data/raw", self.workspace / "data/raw"},
        )

    def test_output_reservation_rejects_both_raw_trees_and_their_parents(self):
        from src.adapters.storage.local import reserve_output

        self.configure()
        with patch("src.core.paths.PROJECT_ROOT", self.project):
            for raw in (self.project / "data/raw", self.workspace / "data/raw"):
                for destination in (raw, raw / "derived", raw.parent):
                    with self.subTest(destination=destination):
                        with self.assertRaises(ValueError):
                            reserve_output(destination, ())
                        self.assertFalse(destination.exists())
            destination = self.workspace / "reports/fresh"
            self.assertEqual(reserve_output(destination, ()), destination)
            with self.assertRaises(FileExistsError):
                reserve_output(destination, ())

    def test_external_paths_are_not_rebound(self):
        self.configure()
        outside = Path(self.temporary.name) / "another_customer/input.png"
        self.assertEqual(resolve_legacy_path(outside, self.project), outside)

    def test_archived_record_is_found_without_modifying_it(self):
        path = self.project / "archive/configs/old.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(b'{"seed":17}')
        self.assertEqual(resolve_legacy_path("configs/old.json", self.project), path)
        self.assertEqual(path.read_bytes(), b'{"seed":17}')

    def test_invalid_local_configuration_does_not_silently_fallback(self):
        (self.project / "configs/local.json").write_text(
            '{"workspace_root":"../relative"}'
        )
        with self.assertRaises(ValueError):
            workspace_root(self.project)

    def test_saved_review_session_is_reopened_without_rewriting_history(self):
        from src.adapters.storage.local import file_hash
        from src.validation.review_server import ReviewStore

        self.configure()
        packet = self.project / "packet"
        packet.mkdir()
        (packet / "references.json").write_text("[]")
        (packet / "profile.json").write_text(
            json.dumps(
                {
                    "status": "prepared",
                    "artifact_sha256": {
                        "references.json": file_hash(packet / "references.json")
                    },
                }
            )
        )
        packet_sha = file_hash(packet / "profile.json")
        state = self.workspace / "review_state"
        state.mkdir(parents=True)
        session = state / "session.json"
        session.write_text(
            json.dumps(
                {
                    "packet_sha256": packet_sha,
                    "source_root": str(self.project / "data/raw"),
                }
            )
        )
        before = session.read_bytes()
        with patch("src.core.paths.PROJECT_ROOT", self.project):
            store = ReviewStore(packet, packet_sha, state, self.workspace / "data/raw")
            self.assertEqual(store.progress()["reviewed"], [])
            self.assertEqual(session.read_bytes(), before)
            with self.assertRaises(ValueError):
                ReviewStore(
                    packet, packet_sha, state, self.workspace / "another_source"
                )


if __name__ == "__main__":
    unittest.main()
