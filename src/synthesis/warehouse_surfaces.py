"""Resolve explicit surface bindings without simulator or deployment paths."""

import copy

from src.adapters.tasks.goods_scene import validate_goods_scene


def bind_surfaces(job, catalog, catalog_sha256, configuration):
    """Return a copy with registered map roles and authored palette mixing frozen.

    Map bytes and catalog integrity are verified by execution preflight. This
    function preserves all geometry, poses, labels and render settings. A material
    binding does not establish photographic origin or appearance qualification.
    """
    if (
        not isinstance(configuration, dict)
        or set(configuration) != {"schema_version", "surfaces"}
        or configuration["schema_version"] != "dubaidata.surface-bindings/1.0"
        or not isinstance(configuration["surfaces"], dict)
        or not configuration["surfaces"]
    ):
        raise ValueError("Unsupported surface-binding configuration")
    resolved = copy.deepcopy(job)
    resolved["asset_catalog_sha256"] = catalog_sha256
    for name, binding in configuration["surfaces"].items():
        if (
            name not in resolved["materials"]
            or not isinstance(binding, dict)
            or set(binding) != {"asset_id", "base_color_mix"}
        ):
            raise ValueError(
                "Surface binding requires a declared material and explicit controls"
            )
        if binding["asset_id"] not in catalog.get("assets", {}):
            raise ValueError("Surface binding references an unregistered asset")
        mix = binding["base_color_mix"]
        if type(mix) not in (int, float) or not 0 <= mix <= 1:
            raise ValueError("Surface palette mix must be finite and bounded")
        asset = catalog["assets"][binding["asset_id"]]
        resolved["materials"][name]["texture_set"] = dict(
            asset_id=binding["asset_id"],
            tile_size_metres=copy.deepcopy(asset["tile_size_metres"]),
            base_color_mix=mix,
            maps={
                role: {key: entry[key] for key in ("path", "sha256", "color_space")}
                for role, entry in asset["maps"].items()
            },
        )
    return validate_goods_scene(resolved)
