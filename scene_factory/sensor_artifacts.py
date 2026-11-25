"""On-demand camera artifacts tagged with the same committed session state."""
from __future__ import annotations

import hashlib
import io
import os
import uuid
from pathlib import Path
from typing import Any

from .session_contracts import finite_json


def capture_observation_images(session: Any, request: Any, response: dict) -> dict:
    request = finite_json(request, "observation request")
    if not isinstance(request, dict) or set(request) - {"modalities", "camera"}:
        raise ValueError("observation request accepts only modalities and camera")
    modalities = request.get("modalities", ["state"])
    if not isinstance(modalities, list) or not modalities or any(not isinstance(v, str) for v in modalities):
        raise ValueError("modalities must be a nonempty list of strings")
    if len(set(modalities)) != len(modalities):
        raise ValueError("duplicate observation modalities")
    requested = set(modalities) - {"state"}
    if requested - set(session.capabilities["image_modalities"]):
        raise ValueError("requested image modality is unsupported")
    if request.get("camera", "default") != "default":
        raise ValueError("only the default camera is currently supported")
    if not requested:
        return response
    if session.artifact_root is None:
        raise ValueError("configure session.artifact_root before requesting image references")
    import numpy as np

    root = Path(session.artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    backend = session.env.backend
    before_step = backend.steps
    before_time = getattr(getattr(backend, "_data", None), "time", None)
    simulation = getattr(backend, "_simulation", None)
    before_engine_time = getattr(simulation, "current_time", None)
    calibration = None
    frame = None
    try:
        if session.backend == "isaac":
            frame = backend.capture_rgbd_observation()
            if frame["simulation_step"] != before_step:
                raise RuntimeError("camera and committed state are not aligned")
            arrays = {key: frame[key] for key in requested}
            calibration = {
                "intrinsics": np.asarray(frame["intrinsics"]).tolist(),
                "camera_to_world": np.asarray(frame["extrinsics"]).tolist(),
                "camera_axes": "world", "rendering_frame": frame["rendering_frame"],
            }
        elif session.backend == "mujoco":
            arrays = {"rgb": session.env.render()}
        else:
            raise ValueError("image capture is unsupported")
    finally:
        after_time = getattr(getattr(backend, "_data", None), "time", None)
        after_engine_time = getattr(simulation, "current_time", None)
        if before_step != backend.steps or before_time != after_time or before_engine_time != after_engine_time:
            session._faulted = True
            session._observation = None
            raise RuntimeError("image capture unexpectedly advanced simulation; reset required")
    images = []
    for modality in sorted(requested):
        array = np.asarray(arrays[modality])
        if modality == "rgb" and (array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8):
            raise ValueError("camera RGB must be H x W x 3 uint8")
        if modality == "depth" and (array.ndim != 2 or not np.isfinite(array).any()):
            raise ValueError("camera depth must contain finite depth samples")
        blob = io.BytesIO()
        np.save(blob, array, allow_pickle=False)
        content = blob.getvalue()
        name = f"{session.episode_id}-{session._state_version}-{modality}-{uuid.uuid4().hex[:8]}.npy"
        target = root / name
        temporary = root / (name + ".tmp")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        images.append({
            "modality": modality, "camera": "default", "path": name,
            "url": f"/v1/sessions/{session.session_id}/artifacts/{name}",
            "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content),
            "format": "npy", "dtype": str(array.dtype), "shape": list(array.shape),
            "units": "m" if modality == "depth" else "uint8_srgb",
            "nonfinite_depth_means": "no_return" if modality == "depth" else None,
            "episode_id": session.episode_id, "state_version": session._state_version,
            "step_index": session._step_index, "sim_time": response["sim_time"],
            "calibration": calibration,
        })
    response["images"] = images
    response["unavailable_fields"] = [v for v in response["unavailable_fields"] if v != "image_references"]
    if calibration is None:
        response["unavailable_fields"].append("camera_calibration")
    response["info"]["render_advanced_simulation"] = False
    return finite_json(response)
