"""Reopen the selected saved annotation packet from the external workspace.

This command reads deployment configuration, verifies the exact packet and serves
the existing review state. It never creates an attestation or selects a cohort.
"""

import json
from pathlib import Path
import sys
from urllib.error import URLError
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.core.paths import workspace_root
from src.validation.review_server import create_server


def main():
    configuration = json.loads(
        (PROJECT / "configs/local.json").read_text(encoding="utf-8-sig")
    )
    review = configuration.get("annotation_review")
    if not isinstance(review, dict):
        raise ValueError(
            "Select annotation_review in configs/local.json before opening saved annotations"
        )
    workspace = workspace_root(PROJECT)

    def contained(name):
        value = (workspace / review[name]).resolve()
        if not value.is_relative_to(workspace):
            raise ValueError("Annotation path escapes the configured workspace")
        return value

    packet, state, source = (
        contained(name) for name in ("packet", "state", "source_root")
    )
    if state == source or state.is_relative_to(source):
        raise ValueError("Saved state cannot be written inside source data")
    port = review["port"]
    url = f"http://127.0.0.1:{port}/"
    try:
        with urlopen(url + "api/session", timeout=2) as response:
            active = json.load(response)
    except URLError:
        active = None
    if active is not None:
        if active["packet_sha256"] != review["packet_sha256"]:
            raise ValueError(
                "Another annotation packet uses this port; it was left intact"
            )
        print("Already running: " + url, flush=True)
        return
    server = create_server(
        packet,
        review["packet_sha256"],
        state,
        source,
        port,
        review.get("reviewer_name", ""),
    )
    print("Saved annotations: " + url + " — Ctrl+C stops this service", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
