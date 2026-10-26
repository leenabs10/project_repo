"""Convert a restricted opaque glTF mesh into renderer-independent triangle data.

This adapter accepts one untransformed mesh node, one triangle primitive and one
external binary buffer. It never executes source scripts or guesses unsupported
transforms, transparency or accessor layouts. Downloads and license registration
belong to asset preparation; this pure conversion is independently fixture-testable.
"""

import numpy as np


def convert_opaque_gltf(document, binary):
    """Return centered unit arrays, original Z-up extents and conversion evidence.

    Source positions are metres in right-handed glTF Y-up coordinates. A proper
    rotation maps them to Z-up; UV V is flipped to the renderer's bottom-left
    convention. Uniform runtime scaling must use the returned original extents.
    No decimation, inferred surfaces, normal maps or material baking is performed.
    """
    nodes = document.get("nodes", [])
    meshes = document.get("meshes", [])
    buffers = document.get("buffers", [])
    materials = document.get("materials", [])
    if (
        document.get("asset", {}).get("version") != "2.0"
        or document.get("extensionsRequired")
        or document.get("extensionsUsed")
        or len(nodes) != 1
        or set(nodes[0]) - {"mesh", "name", "translation"}
        or nodes[0].get("mesh") != 0
        or len(meshes) != 1
        or len(meshes[0].get("primitives", [])) != 1
        or len(buffers) != 1
        or buffers[0].get("byteLength") != len(binary)
        or len(binary) > 32 * 1024 * 1024
        or len(materials) != 1
        or materials[0].get("alphaMode", "OPAQUE") != "OPAQUE"
        or document.get("animations")
        or document.get("skins")
    ):
        raise ValueError(
            "Unsupported glTF geometry, transform or opaque material layout"
        )
    primitive = meshes[0]["primitives"][0]
    if (
        set(primitive) - {"attributes", "indices", "material", "mode"}
        or primitive.get("mode", 4) != 4
        or primitive.get("material") != 0
        or set(primitive["attributes"]) - {"POSITION", "NORMAL", "TEXCOORD_0"}
        or not {"POSITION", "TEXCOORD_0"} <= primitive["attributes"].keys()
    ):
        raise ValueError(
            "Only a single opaque triangle primitive with explicit UVs is supported"
        )

    def read_accessor(index, kind, components, allowed):
        specification = document["accessors"][index]
        if (
            set(specification)
            - {
                "bufferView",
                "componentType",
                "count",
                "type",
                "min",
                "max",
                "byteOffset",
            }
            or specification.get("type") != kind
            or specification.get("componentType") not in allowed
            or type(specification.get("count")) is not int
            or not 1 <= specification["count"] <= 600000
        ):
            raise ValueError("Unsupported glTF accessor")
        view = document["bufferViews"][specification["bufferView"]]
        if (
            set(view) - {"buffer", "byteOffset", "byteLength", "target"}
            or view.get("buffer") != 0
        ):
            raise ValueError("Interleaved or external glTF buffers are unsupported")
        dtype = np.dtype(
            {5126: "<f4", 5123: "<u2", 5125: "<u4"}[specification["componentType"]]
        )
        local_offset = specification.get("byteOffset", 0)
        start = view.get("byteOffset", 0) + local_offset
        length = specification["count"] * components * dtype.itemsize
        if (
            type(start) is not int
            or start < 0
            or local_offset < 0
            or local_offset + length > view["byteLength"]
            or start + length > len(binary)
        ):
            raise ValueError("glTF accessor exceeds its declared byte range")
        return (
            np.frombuffer(
                binary,
                dtype=dtype,
                count=specification["count"] * components,
                offset=start,
            )
            .copy()
            .reshape(-1, components)
        )

    vertices = read_accessor(
        primitive["attributes"]["POSITION"], "VEC3", 3, {5126}
    ).astype(np.float64)
    translation = np.asarray(nodes[0].get("translation", [0, 0, 0]), dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Invalid glTF translation")
    vertices += translation
    vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
    extent = np.ptp(vertices, axis=0)
    if not np.isfinite(vertices).all() or np.any(extent <= 0) or len(vertices) > 100000:
        raise ValueError("Invalid source mesh bounds or vertex budget")
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = ((vertices - center) / extent).astype(np.float32)
    uv = read_accessor(primitive["attributes"]["TEXCOORD_0"], "VEC2", 2, {5126})
    uv[:, 1] = 1 - uv[:, 1]
    indices = read_accessor(primitive["indices"], "SCALAR", 1, {5123, 5125})
    if len(indices) % 3 or indices.max() >= len(vertices):
        raise ValueError("Invalid source triangle indices")
    triangles = indices.reshape(-1, 3).astype(np.int32)
    points = vertices[triangles]
    keep = (
        np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
        )
        > 1e-12
    )
    triangles = triangles[keep]
    if (
        not len(triangles)
        or len(triangles) > 200000
        or uv.shape != (len(vertices), 2)
        or not np.isfinite(uv).all()
        or uv.min() < -1e-5
        or uv.max() > 1.00001
    ):
        raise ValueError("Invalid source triangles or UV coordinates")
    return (
        dict(vertices=vertices, triangles=triangles, uv=uv),
        dict(
            original_extent_metres=extent.tolist(),
            vertices=len(vertices),
            triangle_count=len(triangles),
            removed_degenerate_triangles=int((~keep).sum()),
            source_axes="glTF right-handed Y-up",
            resolved_axes="right-handed Z-up",
            uv_conversion="v -> 1-v",
            normals="Engine-computed; source normal maps are not applied",
        ),
    )
