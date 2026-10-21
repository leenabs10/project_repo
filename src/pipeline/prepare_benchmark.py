"""Source-bound benchmark preparation, corrected-target import and admission CLIs."""

import argparse
import json
from pathlib import Path

from src.adapters.datasets.registry import dataset_adapter
from src.validation.dataset_acceptance import prepare_decisions, qualify_dataset
from src.validation.reference_labels import import_references, prepare_references
from src.validation.artifacts import verify_snapshot
from src.validation.draft_review import prepare_draft_review


def main():
    """Dispatch preparation, import and dataset admission as separate operations.

    Preparing an editor does not qualify its labels. Import checks submitted
    provenance; admission also checks the exact generated batch snapshot.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Create native train/development reference candidates"
    )
    prepare.add_argument("--runtime", type=Path, required=True)
    prepare.add_argument("--dataset-id", required=True)
    prepare.add_argument("--task", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, default=Path.cwd())
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--batch", type=Path)
    prepare.add_argument("--batch-sha256")
    importer = commands.add_parser(
        "import", help="Validate source-bound corrected labels"
    )
    importer.add_argument("--packet", type=Path, required=True)
    importer.add_argument("--packet-sha256", required=True)
    importer.add_argument("--submissions", type=Path, required=True)
    importer.add_argument("--source-root", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    qualify = commands.add_parser(
        "qualify", help="Admit only qualified inputs and corrected targets"
    )
    qualify.add_argument("--batch", type=Path, required=True)
    qualify.add_argument("--batch-sha256", required=True)
    qualify.add_argument("--decisions", type=Path, required=True)
    qualify.add_argument("--reviewer", default="")
    qualify.add_argument("--references", type=Path, required=True)
    qualify.add_argument("--references-sha256", required=True)
    qualify.add_argument("--task", type=Path, required=True)
    qualify.add_argument("--source-root", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    decisions = commands.add_parser(
        "decisions", help="Create a snapshot-bound decision template"
    )
    decisions.add_argument("--batch", type=Path, required=True)
    decisions.add_argument("--batch-sha256", required=True)
    decisions.add_argument("--source-root", type=Path, required=True)
    decisions.add_argument("--output", type=Path, required=True)
    review = commands.add_parser(
        "review-drafts", help="Prepare a native annotation reference workspace"
    )
    for flag in ("packet", "drafts", "task", "output", "source-root"):
        review.add_argument("--" + flag, type=Path, required=True)
    review.add_argument("--packet-sha256", required=True)
    review.add_argument("--drafts-sha256", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        configuration = json.loads(args.runtime.read_text(encoding="utf-8"))["dataset"]
        for name in ("root", "manifest"):
            configuration[name] = str((args.workspace / configuration[name]).resolve())
        adapter = dataset_adapter(configuration, args.dataset_id)
        required = ()
        if bool(args.batch) != bool(args.batch_sha256):
            parser.error("Batch path and checksum must be provided together")
        if args.batch:
            profile = verify_snapshot(args.batch, args.batch_sha256)
            if (
                profile["status"] != "passed"
                or profile["manifest_sha256"] != adapter.manifest_sha256
                or profile["source_split"] != "train"
            ):
                raise ValueError(
                    "Reference preparation batch does not match training sources"
                )
            job = json.loads((args.batch / "job.json").read_text(encoding="utf-8"))
            required = tuple(job["source_sample_ids"])
        result = prepare_references(
            adapter, args.task, args.output, required_training_ids=required
        )
    elif args.command == "review-drafts":
        result = prepare_draft_review(
            args.packet,
            args.packet_sha256,
            args.drafts,
            args.drafts_sha256,
            args.task,
            args.output,
            args.source_root,
        )
    elif args.command == "import":
        result = import_references(
            args.packet,
            args.submissions,
            args.output,
            args.packet_sha256,
            source_root=args.source_root,
        )
    elif args.command == "decisions":
        path = prepare_decisions(
            args.batch, args.batch_sha256, args.output, source_root=args.source_root
        )
        result = {"status": "prepared", "decisions": str(path)}
    else:
        result = qualify_dataset(
            args.batch,
            args.batch_sha256,
            args.decisions,
            args.reviewer,
            args.references,
            args.task,
            args.output,
            source_root=args.source_root,
            expected_reference_sha256=args.references_sha256,
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "artifact_sha256"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
