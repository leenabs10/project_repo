"""Self-contained collections of individually accepted opaque scene jobs.

Collection assembly preserves each admission, decision and payload unchanged.
It grants no new appearance approval, training protocol or utility qualification.
"""

import argparse
import json
from pathlib import Path
import shutil

from src.core.paths import workspace_root
from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.training.scene_inputs import load_admitted_scene_inputs
from src.validation.artifacts import artifact_hashes, verify_snapshot, write_json
from src.validation.scene_admission import require_snapshot_digest

SCHEMA_VERSION = "dubaidata.scene-collection/1.0"


def _member_rows(
    root, members, references, references_sha256, real_rows, config, *, fixture
):
    """Revalidate every original admission and rebase only its two input paths."""
    rows = []
    seen = {key: set() for key in ("sample_id", "source_rgb_sha256")}
    digests = []
    jobs = set()
    for member in members:
        if not isinstance(member, dict) or set(member) != {
            "path",
            "profile_sha256",
            "job_sha256",
            "eligible_samples",
        }:
            raise ValueError("Invalid scene collection member declaration")
        digest = member["profile_sha256"]
        require_snapshot_digest(digest)
        if member["path"] != "members/" + digest:
            raise ValueError("Scene collection member path must match its digest")
        if digest in digests or member["job_sha256"] in jobs:
            raise ValueError("Duplicate admission or job in scene collection")
        digests.append(digest)
        jobs.add(member["job_sha256"])
        member_root = source_path(root, member["path"])
        admitted = load_admitted_scene_inputs(
            member_root,
            digest,
            references,
            references_sha256,
            real_rows,
            config,
            fixture=fixture,
        )
        if len(admitted) != member["eligible_samples"] or any(
            row["job_sha256"] != member["job_sha256"] for row in admitted
        ):
            raise ValueError("Scene collection member binding mismatch")
        for original in admitted:
            for key, values in seen.items():
                if original[key] in values:
                    raise ValueError("Duplicate scene collection " + key)
                values.add(original[key])
            row = dict(original)
            for key in ("rgb_path", "target_path"):
                source_path(member_root, row[key])
                row[key] = member["path"] + "/" + row[key]
                source_path(root, row[key])
            rows.append(row)
    if not rows or digests != sorted(digests):
        raise ValueError("Scene collection requires nonempty digest-ordered members")
    return sorted(rows, key=lambda row: row["sample_id"])


def load_scene_collection(
    root,
    expected_sha256,
    references,
    references_sha256,
    real_rows,
    config,
    *,
    fixture=False,
):
    """Return native training rows after full admission and collection checks.

    Paths are relative to the collection root; IDs, labels, units and source
    hashes retain their original meaning. Invalid, stale, duplicate, mixed-cohort
    or nested inputs raise ValueError. No development/test rows are admitted.
    """
    require_snapshot_digest(expected_sha256)
    require_snapshot_digest(references_sha256)
    root = Path(root)
    profile = verify_snapshot(root, expected_sha256)
    reference = verify_snapshot(references, references_sha256)
    if (
        profile.get("schema_version") != SCHEMA_VERSION
        or profile.get("kind") != "scene_collection"
        or profile.get("status") != "passed"
        or profile.get("qualification_status") != "accepted_members_only"
        or profile.get("training_eligible") is not True
        or profile.get("purpose")
        != ("known_fixture_only" if fixture else "qualified_scene_collection")
        or profile.get("reference_profile_sha256") != references_sha256
        or any(
            profile.get(key) != reference.get(key)
            for key in ("task_id", "protocol_sha256", "manifest_sha256")
        )
        or not isinstance(profile.get("members"), list)
    ):
        raise ValueError("Completed scene collection and matching cohort are required")
    rows = _member_rows(
        root,
        profile["members"],
        references,
        references_sha256,
        real_rows,
        config,
        fixture=fixture,
    )
    saved = [
        json.loads(line)
        for line in (root / "eligible_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if rows != saved or len(rows) != profile.get("eligible_samples"):
        raise ValueError("Scene collection manifest differs from accepted members")
    return rows


def create_scene_collection(
    members, references, references_sha256, real_rows, config, output, *, fixture=False
):
    """Copy accepted (path, profile-SHA256) pairs into a fresh portable snapshot.

    Source admissions must already contain exact-batch decisions. The supplied
    real rows must be the verified reference cohort used by the training caller.
    All validation precedes allocation; copying is checked again before success.
    Failed copies remain quarantined with training_eligible=False. Inputs and
    data/raw are protected against output overlap. No training is executed.
    """
    require_snapshot_digest(references_sha256)
    reference = verify_snapshot(references, references_sha256)
    sources = {}
    declarations = []
    all_rows = []
    for path, digest in members:
        require_snapshot_digest(digest)
        if digest in sources:
            raise ValueError("Duplicate admission in scene collection")
        path = Path(path).resolve()
        admitted = load_admitted_scene_inputs(
            path,
            digest,
            references,
            references_sha256,
            real_rows,
            config,
            fixture=fixture,
        )
        sources[digest] = path
        all_rows.extend(admitted)
        declarations.append(
            dict(
                path="members/" + digest,
                profile_sha256=digest,
                job_sha256=admitted[0]["job_sha256"],
                eligible_samples=len(admitted),
            )
        )
    declarations.sort(key=lambda member: member["profile_sha256"])
    for key in ("sample_id", "source_rgb_sha256"):
        if len({row[key] for row in all_rows}) != len(all_rows):
            raise ValueError("Duplicate scene collection " + key)
    if not all_rows or len({m["job_sha256"] for m in declarations}) != len(
        declarations
    ):
        raise ValueError("Scene collection requires nonempty distinct jobs")
    output = reserve_output(
        Path(output),
        (
            *sources.values(),
            Path(references),
            workspace_root() / "data/raw",
        ),
    )
    try:
        for member in declarations:
            source = sources[member["profile_sha256"]]
            destination = output / member["path"]
            original = verify_snapshot(source, member["profile_sha256"])
            # Only bound artifacts belong to this snapshot. Ignore stray files.
            for relative in ("profile.json", *original["artifact_sha256"]):
                target = source_path(destination, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path(source, relative), target)
        rows = _member_rows(
            output,
            declarations,
            references,
            references_sha256,
            real_rows,
            config,
            fixture=fixture,
        )
        (output / "eligible_manifest.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        profile = dict(
            schema_version=SCHEMA_VERSION,
            kind="scene_collection",
            status="passed",
            purpose="known_fixture_only" if fixture else "qualified_scene_collection",
            qualification_status="accepted_members_only",
            training_eligible=True,
            reference_profile_sha256=references_sha256,
            task_id=reference["task_id"],
            protocol_sha256=reference["protocol_sha256"],
            manifest_sha256=reference["manifest_sha256"],
            members=declarations,
            eligible_samples=len(rows),
            held_out_test_evaluated=False,
            physical_sensor_realism_certified=False,
            utility_measured=False,
            artifact_sha256=artifact_hashes(output),
        )
        write_json(output / "profile.json", profile)
        load_scene_collection(
            output,
            file_hash(output / "profile.json"),
            references,
            references_sha256,
            real_rows,
            config,
            fixture=fixture,
        )
        return profile
    except Exception as error:
        write_json(
            output / "profile.json",
            dict(
                status="failed",
                training_eligible=False,
                error=str(error),
            ),
        )
        raise


def main():
    """Assemble a collection using a verified cohort and explicit source digests."""
    from src.training.goods_benchmark import load_references

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member",
        nargs=2,
        action="append",
        required=True,
        metavar=("ADMISSION", "PROFILE_SHA256"),
    )
    for name in ("references", "task", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--references-sha256", required=True)
    args = parser.parse_args()
    rows = load_references(args.references, args.references_sha256, args.task)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = create_scene_collection(
        args.member,
        args.references,
        args.references_sha256,
        rows,
        config,
        args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
