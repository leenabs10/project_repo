"""Warehouse task semantics layered over the dataset-independent scene contract."""

from src.core.scene import validate_object_separation, validate_scene_job


def validate_goods_scene(job):
    """Keep empty transport equipment background and identifiable loads goods.

    Role assignments are authored scene semantics, not inferred object contents.
    Packaging parts may overlap their loads; loads cannot penetrate equipment
    or structural geometry. Neither condition establishes real-world fidelity.
    """
    validate_scene_job(job)
    if job["taxonomy"] != {"0": "background", "1": "identifiable_goods"}:
        raise ValueError("The goods pilot requires its declared binary taxonomy")
    for capture in job["scenes"]:
        for obj in capture["objects"]:
            if obj["role"] not in ("goods", "equipment", "structure"):
                raise ValueError("Unknown role for the warehouse goods scene adapter")
            if obj["class_id"] != int(obj["role"] == "goods"):
                raise ValueError(
                    "Goods and transport equipment must retain distinct labels"
                )
        validate_object_separation(
            capture["objects"], "goods", {"equipment", "structure"}
        )
    return job
