"""Current presentation must preserve bindings, snapshots and confirmation state."""

import json
import threading
import unittest
from urllib.request import urlopen

from src.adapters.storage.local import file_hash
from src.validation.artifacts import verify_snapshot
from src.validation.reference_pages import reference_page
from src.validation.review_server import create_server


class ReferencePageTests(unittest.TestCase):
    def setUp(self):
        from test_benchmark_preparation import BenchmarkPreparationTests

        self.fixture = BenchmarkPreparationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_current_page_keeps_exact_binding_and_no_selected_confirmation(self):
        packet = self.fixture.packet
        entries = {
            entry["binding"]["sample_id"]: entry
            for entry in json.loads((packet / "references.json").read_text())
        }
        entry = entries["train/frame"]
        html = reference_page(packet, entries, entry["folder"] + "/index.html")
        self.assertIn("const source = " + json.dumps(entry["binding"]), html)
        self.assertNotIn("AI draft", html)
        self.assertNotIn("human confirmation", html)
        self.assertNotIn('id="confirmed" type="checkbox" checked', html)
        self.assertIn("reviewer_kind: 'human'", html)

    def test_server_replaces_presentation_without_mutating_packet(self):
        packet = self.fixture.packet
        before = file_hash(packet / "profile.json")
        server = create_server(
            packet, before, self.fixture.base / "current_state", self.fixture.raw, 0
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        for suffix in ("index.html", "guide.html"):
            with urlopen(f"http://127.0.0.1:{server.server_port}/{suffix}") as response:
                page = response.read().decode()
            self.assertNotIn("AI draft", page)
            self.assertNotIn("human review pending", page)
        self.assertEqual(server.review_store.progress()["reviewed"], [])
        self.assertEqual(file_hash(packet / "profile.json"), before)
        verify_snapshot(packet, before)
