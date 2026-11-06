"""Local navigation rejects unsafe selections and refreshes an active server."""

import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from urllib.request import urlopen

from src.validation.report_workspace import render_navigation

PROJECT = Path(__file__).resolve().parents[1]
TEMPLATE = "<p>{{summary}}</p><ul>{{navigation}}</ul>"


class ReportWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = self.root / "reports/scene view/index.html"
        self.report.parent.mkdir(parents=True)
        self.report.write_text("Existing evidence", encoding="utf-8")
        self.configuration = {
            "workspace_root": str(self.root),
            "report_workspace": {
                "port": 8810,
                "summary": "<not a utility claim>",
                "scenes": "scene view/index.html",
                "predictions": "scene view/index.html",
                "links": [
                    {
                        "label": "<Evidence>",
                        "path": "scene view/index.html",
                        "description": "RGB & labels",
                    }
                ],
            },
        }

    def test_navigation_escapes_text_encodes_paths_and_preserves_evidence(self):
        before = self.report.read_bytes()
        rendered = render_navigation(self.configuration, TEMPLATE)
        self.assertIn("&lt;Evidence&gt;", rendered)
        self.assertIn("RGB &amp; labels", rendered)
        self.assertIn("&lt;not a utility claim&gt;", rendered)
        self.assertIn("../scene%20view/index.html", rendered)
        self.assertEqual(self.report.read_bytes(), before)

    def test_invalid_and_missing_selections_fail_before_writing(self):
        for path in ("../outside.html", "C:/outside.html", "/absolute", "bad\\path"):
            configuration = copy.deepcopy(self.configuration)
            configuration["report_workspace"]["links"][0]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                render_navigation(configuration, TEMPLATE)
        configuration = copy.deepcopy(self.configuration)
        configuration["report_workspace"]["scenes"] = "missing.html"
        with self.assertRaises(FileNotFoundError):
            render_navigation(configuration, TEMPLATE)
        for value in (True, 0, 65536, "8810"):
            configuration = copy.deepcopy(self.configuration)
            configuration["report_workspace"]["port"] = value
            with self.subTest(port=value), self.assertRaises(ValueError):
                render_navigation(configuration, TEMPLATE)
        self.assertFalse((self.root / "reports/mvp").exists())

    def test_empty_selection_is_explicit_and_bad_templates_fail(self):
        rendered = render_navigation({"workspace_root": str(self.root)}, TEMPLATE)
        self.assertIn("No report selection is configured", rendered)
        for template in ("{{summary}}", TEMPLATE + "{{navigation}}"):
            with self.assertRaises(ValueError):
                render_navigation(self.configuration, template)

    @unittest.skipUnless(os.name == "nt", "Windows launcher integration")
    def test_launcher_start_reuse_refresh_and_configured_selections(self):
        project = self.root / "project"
        (project / "scripts/templates").mkdir(parents=True)
        (project / "configs").mkdir()
        shutil.copyfile(PROJECT / "scripts/mvp.ps1", project / "scripts/mvp.ps1")
        shutil.copyfile(
            PROJECT / "scripts/templates/mvp_index.html",
            project / "scripts/templates/mvp_index.html",
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        self.configuration["python_executable"] = sys.executable
        self.configuration["report_workspace"]["port"] = port
        config_path = project / "configs/local.json"
        config_path.write_text(json.dumps(self.configuration), encoding="utf-8")
        environment = dict(os.environ, PYTHONPATH=str(PROJECT))

        def launch(action):
            # A detached Windows server can retain an inherited pipe handle.
            # Capture to a regular file so completion never waits for pipe EOF.
            with (project / "launcher-output.txt").open("w+", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(project / "scripts/mvp.ps1"),
                        action,
                        "-NoBrowser",
                    ],
                    cwd=project,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
                log.seek(0)
                output = log.read()
            self.assertEqual(result.returncode, 0, output)
            return output

        try:
            launch("reports")
            session = self.root / "services/reports.json"
            original_session = session.read_bytes()
            self.configuration["report_workspace"]["summary"] = "Refreshed selection"
            config_path.write_text(json.dumps(self.configuration), encoding="utf-8")
            launch("reports")
            self.assertEqual(session.read_bytes(), original_session)
            with urlopen(
                f"http://127.0.0.1:{port}/mvp/index.html", timeout=5
            ) as response:
                self.assertIn(b"Refreshed selection", response.read())
            for action in ("scenes", "predictions"):
                self.assertIn("scene%20view/index.html", launch(action))
            self.assertEqual(self.report.read_text(), "Existing evidence")
        finally:
            if (self.root / "services/reports.json").exists():
                launch("stop")


if __name__ == "__main__":
    unittest.main()
