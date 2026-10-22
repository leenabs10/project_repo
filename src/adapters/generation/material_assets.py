"""Resolve immutable texture assets outside simulator and customer-specific code."""

import json
from pathlib import Path

from PIL import Image

from src.adapters.storage.local import file_hash, source_path


def verify_material_assets(job, asset_root):
    """Verify catalog, role/color-space bindings and PNG bytes before allocation.

    The asset catalog records origin and license. Hash verification establishes
    integrity against that declared catalog, not automatic legal or visual quality.
    Only color and roughness maps are supported; no inferred normal/displacement.
    """
    textures = [
        surface["texture_set"]
        for surface in job["materials"].values()
        if "texture_set" in surface
    ]
    if not textures:
        return {}
    if asset_root is None:
        raise ValueError("Textured scene requires a configured asset root")
    root = Path(asset_root).resolve()
    catalog_path = root / "catalog.json"
    if file_hash(catalog_path) != job["asset_catalog_sha256"]:
        raise ValueError("Material asset catalog checksum mismatch")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "dubaidata.material-assets/1.0":
        raise ValueError("Unsupported material asset catalog")
    verified = {}
    for texture in textures:
        entry = catalog["assets"].get(texture["asset_id"])
        if (
            entry is None
            or not entry.get("license")
            or not entry.get("source_page")
            or entry["tile_size_metres"] != texture["tile_size_metres"]
        ):
            raise ValueError(
                "Texture asset lacks matching source, license or metric scale"
            )
        for role, binding in texture["maps"].items():
            registered = entry["maps"][role]
            if any(
                registered[key] != binding[key]
                for key in ("path", "sha256", "color_space")
            ):
                raise ValueError("Texture map is not bound to the registered asset")
            path = source_path(root, binding["path"])
            if binding["path"] in verified:
                continue
            if file_hash(path) != binding["sha256"]:
                raise ValueError("Texture map checksum mismatch")
            with Image.open(path) as image:
                if (
                    image.format != "PNG"
                    or min(image.size) < 4
                    or max(image.size) > 8192
                ):
                    raise ValueError("Unsupported texture image geometry or encoding")
                image.verify()
            verified[binding["path"]] = binding["sha256"]
    return verified
