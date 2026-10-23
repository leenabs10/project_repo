"""Verify registered textures and meshes independently of renderer implementations."""

import json
from pathlib import Path

from PIL import Image

from src.adapters.storage.local import file_hash, source_path
from src.adapters.generation.mesh_assets import load_normalized_mesh


def verify_material_assets(job, asset_root):
    """Return verified asset-path checksums after catalog and payload validation.

    Check source/license bindings, texture roles and color spaces, PNG payloads,
    and normalized meshes before renderer allocation. Supported texture maps are
    color and roughness. Checksums establish catalog integrity; appearance and
    license suitability require separate qualification.
    """
    textures = [
        surface["texture_set"]
        for surface in job["materials"].values()
        if "texture_set" in surface
    ]
    meshes = [
        obj["mesh"]
        for scene in job["scenes"]
        for obj in scene["objects"]
        if "mesh" in obj
    ]
    if not textures and not meshes:
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
    for binding in meshes:
        entry = catalog["assets"].get(binding["asset_id"], {})
        if (
            not entry.get("license")
            or not entry.get("source_page")
            or entry.get("mesh") != binding
        ):
            raise ValueError("Mesh lacks a matching registered source/license binding")
        if binding["path"] not in verified:
            load_normalized_mesh(binding, root)
            verified[binding["path"]] = binding["sha256"]
    return verified
