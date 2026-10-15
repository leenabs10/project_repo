"""Reject unavailable backends and unsupported sensor requests before generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendCapabilities:
    """Separate declared output types from implemented common-worker support."""

    backend_id: str
    modalities: frozenset[str]
    worker_implemented: bool
    limitation: str

    def require(self, requested: set[str]) -> None:
        missing = requested - self.modalities
        if missing:
            raise NotImplementedError(
                "Unsupported requested modalities: " + ", ".join(sorted(missing))
            )
        if not self.worker_implemented:
            raise NotImplementedError(self.limitation)


BACKENDS = {
    "real_sensor_conditions": BackendCapabilities(
        "real_sensor_conditions",
        frozenset({"rgb", "depth", "semantic", "lidar"}),
        True,
        "Only RGB gain and depth-mask transfer vary inputs; labels/LiDAR are preserved. Not scene simulation or calibrated sensor noise.",
    ),
    "blenderproc": BackendCapabilities(
        "blenderproc",
        frozenset({"rgb", "depth", "semantic"}),
        False,
        "Common generation worker and rendering compatibility are pending; installation is not integration.",
    ),
    "isaac_sim": BackendCapabilities(
        "isaac_sim",
        frozenset(),
        False,
        "Candidate backend; no implemented/validated adapter or sensor support declared.",
    ),
}
