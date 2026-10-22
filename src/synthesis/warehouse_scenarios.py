"""Validate bounded warehouse coverage intent before resolving renderer geometry.

These are authored task cases, not a reconstruction of customer scenes. Coordinates
use metres in the existing warehouse world frame. Only floor and the known front
pallet are supported; arbitrary support heights would permit floating loads.
"""

import math
import re


def _fields(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError("Unsupported or missing warehouse coverage fields")


def _numbers(values, length):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("Coverage requires explicit metric vectors")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError("Coverage coordinates and dimensions must be finite numbers")


def validate_coverage(configuration):
    """Reject ambiguous supports, unsafe identifiers and unbounded task requests.

    Final cross-role penetration is checked against resolved scene geometry by the
    goods task adapter. This validation does not certify physics or appearance.
    """
    _fields(configuration, ("schema_version", "cases"))
    if configuration["schema_version"] != "dubaidata.warehouse-coverage/1.0":
        raise ValueError("Unsupported warehouse coverage version")
    cases = configuration["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ValueError("Coverage requires one to eight captures")
    identifiers = set()
    for case in cases:
        _fields(case, ("sample_id", "objective", "camera_x_metres", "loads"))
        identifier = case["sample_id"]
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", identifier)
            or identifier in identifiers
        ):
            raise ValueError("Duplicate or unsafe coverage identifier")
        identifiers.add(identifier)
        if not isinstance(case["objective"], str) or not case["objective"].strip():
            raise ValueError("Every coverage case requires a declared objective")
        _numbers([case["camera_x_metres"]], 1)
        if abs(case["camera_x_metres"]) > 0.75:
            raise ValueError("Camera offset exceeds the bounded aisle pilot")
        if not isinstance(case["loads"], list) or len(case["loads"]) > 2:
            raise ValueError("At most two explicit coverage loads are supported")
        bodies = []
        for load in case["loads"]:
            _fields(
                load, ("support", "center_xy_metres", "dimensions_metres", "material")
            )
            _numbers(load["center_xy_metres"], 2)
            _numbers(load["dimensions_metres"], 3)
            x, y = load["center_xy_metres"]
            width, length, height = load["dimensions_metres"]
            if not (
                0.25 <= width <= 1.1 and 0.25 <= length <= 0.9 and 0.25 <= height <= 1.4
            ):
                raise ValueError("Load dimensions exceed the bounded carton family")
            if load["support"] == "foreground_pallet":
                if [x, y] != [-0.85, -1.6]:
                    raise ValueError(
                        "Pallet load must be centered on its declared support"
                    )
                bottom = 0.22
            elif load["support"] == "floor":
                if not (-1.5 <= x <= 1.5 and 1.2 <= y <= 12):
                    raise ValueError("Floor load is outside the bounded aisle region")
                bottom = 0.0
            else:
                raise ValueError("Unknown load support")
            if load["material"] not in (
                "paper",
                "carton_0",
                "carton_1",
                "carton_2",
                "carton_3",
            ):
                raise ValueError("Load requires a registered carton material")
            # Composite tape/label parts may overlap their own carton. Separate
            # load bodies may not overlap each other, even though both are goods.
            center = (x, y, bottom + height / 2)
            dimensions = (width, length, height)
            for previous_center, previous_dimensions in bodies:
                if all(
                    abs(a - b) < (da + db) / 2 - 0.0001
                    for a, b, da, db in zip(
                        center, previous_center, dimensions, previous_dimensions
                    )
                ):
                    raise ValueError("Coverage load bodies overlap")
            bodies.append((center, dimensions))
    return configuration
