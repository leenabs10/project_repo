"""Loopback-only native annotation workspace with validated, versioned saves."""

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
from threading import RLock
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from src.core.paths import resolve_legacy_path
from src.adapters.storage.local import reserve_output
from src.validation.artifacts import verify_snapshot, write_json
from src.validation.reference_labels import import_references
from src.validation.reference_pages import reference_page


class ReviewStore:
    """Keep immutable reviewed versions and a replaceable current selection per ID.

    Import validation precedes every write to the selected cohort. This validates
    technical contracts, not the truth of a person's semantic attestation.
    """

    def __init__(self, packet, packet_sha, state, source_root):
        self.packet, self.source_root = (
            Path(packet).resolve(),
            Path(source_root).resolve(),
        )
        self.packet_sha = packet_sha
        profile = verify_snapshot(self.packet, packet_sha)
        if profile["status"] != "prepared":
            raise ValueError("Review packet is not prepared")
        entries = json.loads((self.packet / "references.json").read_text())
        self.entries = {item["binding"]["sample_id"]: item for item in entries}
        self.state = Path(state).resolve()
        if self.state.exists():
            existing = json.loads((self.state / "session.json").read_text())
            # A recorded deployment relocation changes the directory, not the
            # raw payload identity. Keep historical session bytes unchanged and
            # compare their root through the explicit local path resolver.
            if (
                set(existing) != {"packet_sha256", "source_root"}
                or existing["packet_sha256"] != packet_sha
                or resolve_legacy_path(existing["source_root"]) != self.source_root
            ):
                raise ValueError(
                    "Existing review state belongs to another packet/source"
                )
        else:
            reserve_output(self.state, (self.packet, self.source_root))
            write_json(
                self.state / "session.json",
                {"packet_sha256": packet_sha, "source_root": str(self.source_root)},
            )
        self.current = self.state / "current"
        self.current.mkdir(exist_ok=True)
        (self.state / "history").mkdir(exist_ok=True)
        (self.state / "validations").mkdir(exist_ok=True)
        self.lock = RLock()
        self.token = secrets.token_urlsafe(32)

    @staticmethod
    def identity(sample_id):
        """Map a sample ID to a bounded filename without interpreting it as a path."""
        return hashlib.sha256(sample_id.encode()).hexdigest()[:24]

    def progress(self):
        """Return only selected, previously validated human versions."""
        with self.lock:
            reviewed = [
                json.loads(path.read_text())["sample_id"]
                for path in sorted(self.current.glob("*.json"))
            ]
            return {
                "reviewed": reviewed,
                "total": len(self.entries),
                "packet_sha256": self.packet_sha,
            }

    def load(self, sample_id):
        """Read the latest selected version for an ID in this fixed packet."""
        if sample_id not in self.entries:
            raise ValueError("Unknown review sample")
        path = self.current / (self.identity(sample_id) + ".json")
        return json.loads(path.read_text()) if path.exists() else None

    def save(self, submission):
        """Validate one human export, preserve its history, then select it atomically."""
        if (
            not isinstance(submission, dict)
            or submission.get("reviewer_kind") != "human"
        ):
            raise ValueError("This endpoint accepts confirmed reference versions only")
        with self.lock, tempfile.TemporaryDirectory() as temporary:
            inputs = Path(temporary) / "submissions"
            inputs.mkdir()
            payload = json.dumps(
                submission, separators=(",", ":"), allow_nan=False
            ).encode()
            (inputs / "submission.json").write_bytes(payload)
            validation = self.state / "validations" / uuid4().hex
            # Reject malformed masks or stale bindings before changing selection.
            import_references(
                self.packet,
                inputs,
                validation,
                self.packet_sha,
                source_root=self.source_root,
            )
            identity = self.identity(submission["sample_id"])
            digest = hashlib.sha256(payload).hexdigest()
            history = self.state / "history" / identity
            history.mkdir(exist_ok=True)
            version = history / (digest + ".json")
            if not version.exists():
                version.write_bytes(payload)
            destination = self.current / (identity + ".json")
            staging = self.current / (identity + ".tmp")
            staging.write_bytes(payload)
            # A completed atomic replacement cannot expose a half-written review.
            os.replace(staging, destination)
            return {**self.progress(), "submission_sha256": digest}

    def validate_cohort(self):
        """Freeze a validated cohort snapshot; this does not admit synthetic data."""
        with self.lock:
            output = self.state / "cohorts" / uuid4().hex
            result = import_references(
                self.packet,
                self.current,
                output,
                self.packet_sha,
                source_root=self.source_root,
            )
            write_json(
                self.state / "latest_cohort.json",
                {
                    "path": str(output),
                    "reference_counts": result["reference_counts"],
                    "benchmark_reference_ready": result["benchmark_reference_ready"],
                },
            )
            return {
                "reference_counts": result["reference_counts"],
                "benchmark_reference_ready": result["benchmark_reference_ready"],
                "cohort": output.name,
            }


def create_server(packet, packet_sha, state, source_root, port=8768, reviewer_name=""):
    """Serve a fixed packet only on 127.0.0.1, with same-origin token-bound writes."""
    store = ReviewStore(packet, packet_sha, state, source_root)

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            translated = Path(super().translate_path(path)).resolve()
            if not translated.is_relative_to(store.packet):
                return str(store.packet / "__outside_packet_blocked__")
            return str(translated)

        def reply(self, status, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            route = urlsplit(self.path)
            if route.path == "/api/session":
                return self.reply(200, {**store.progress(), "token": store.token})
            if route.path == "/api/review":
                try:
                    query = parse_qs(route.query)
                    result = store.load(query.get("sample_id", [""])[0])
                    return self.reply(
                        200 if result else 404,
                        result or {"error": "No saved reference version"},
                    )
                except ValueError as error:
                    return self.reply(400, {"error": str(error)})
            page = Path(self.translate_path(self.path))
            if page.is_dir():
                page = page / "index.html"
            if page.is_relative_to(store.packet) and page.suffix == ".html":
                html = reference_page(
                    store.packet,
                    store.entries,
                    page.relative_to(store.packet).as_posix(),
                    reviewer_name,
                )
                if html is not None:
                    payload = html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
            return super().do_GET()

        def do_POST(self):
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if (
                self.headers.get("Origin") != origin
                or not secrets.compare_digest(
                    self.headers.get("X-Review-Token", ""), store.token
                )
                or self.headers.get_content_type() != "application/json"
            ):
                return self.reply(
                    403, {"error": "Use the local workspace to save reviewed versions"}
                )
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64 * 1024 * 1024:
                    raise ValueError("Unsupported submission size")
                value = json.loads(self.rfile.read(length))
                if urlsplit(self.path).path == "/api/reviews":
                    return self.reply(200, store.save(value))
                if urlsplit(self.path).path == "/api/validate":
                    return self.reply(200, store.validate_cohort())
                return self.reply(404, {"error": "Unknown route"})
            except (ValueError, KeyError, TypeError) as error:
                return self.reply(400, {"error": str(error)})

    server = ThreadingHTTPServer(
        ("127.0.0.1", port), partial(Handler, directory=str(store.packet))
    )
    server.review_store = store
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--reviewer-name", default="")
    args = parser.parse_args()
    server = create_server(
        args.packet,
        args.packet_sha256,
        args.state,
        args.source_root,
        args.port,
        args.reviewer_name,
    )
    print(
        f"Native reference editor: http://127.0.0.1:{server.server_port}/", flush=True
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
