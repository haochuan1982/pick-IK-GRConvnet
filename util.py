"""Geometry helpers shared by the pick-and-place pipeline.

Conventions
  * T_a_b is a 4x4 matrix that maps points expressed in frame b into frame a.
  * "base" is the SO101 base_link frame (same frame the hand-eye calibration and URDF use).
  * Camera frame is the RealSense colour optical frame (x right, y down, z forward).
"""
import json
import re
import tempfile
from pathlib import Path

import numpy as np


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def inv_T(T) -> np.ndarray:
    return make_T(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def load_T_base_cam(path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    if data.get("mode") != "eye-to-hand":
        raise ValueError(f"{path}: expected an eye-to-hand calibration, got {data.get('mode')!r}")
    return np.array(data["T_base_cam"], dtype=np.float64)


def urdf_without_meshes(urdf_path) -> str:
    """Temp copy of the URDF with meshes replaced by tiny boxes (kinematics needs no meshes)."""
    text = re.sub(r"<mesh\b[^>]*/>", '<box size="0.001 0.001 0.001"/>', Path(urdf_path).read_text())
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


def deproject(u, v, z, K) -> np.ndarray:
    """Pixel (u, v) with depth z (m) -> 3D point in the camera frame."""
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def transform_point(T, p) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p) + T[:3, 3]


def pixel_to_base_on_plane(u, v, K, T_base_cam, z_plane) -> np.ndarray:
    """Intersect the camera ray through pixel (u, v) with the horizontal plane z = z_plane (base frame)."""
    d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    d = T_base_cam[:3, :3] @ d_cam
    o = T_base_cam[:3, 3]
    if abs(d[2]) < 1e-6:
        raise ValueError("camera ray is parallel to the plane")
    s = (z_plane - o[2]) / d[2]
    if s <= 0:
        raise ValueError("plane is behind the camera")
    return o + s * d


def robust_depth(depth_m, u, v, half=3) -> float:
    """Median of the valid (non-zero) depth values in a (2*half+1)^2 window around (u, v)."""
    h, w = depth_m.shape
    u, v = int(round(u)), int(round(v))
    win = depth_m[max(v - half, 0):min(v + half + 1, h), max(u - half, 0):min(u + half + 1, w)]
    valid = win[win > 0]
    if valid.size == 0:
        raise ValueError(f"no valid depth around pixel ({u}, {v})")
    return float(np.median(valid))


def topdown_pose(position, jaw_axis_xy, jaw_axis="x") -> np.ndarray:
    """Gripper pose with the approach axis (tool +z) pointing straight down.

    jaw_axis_xy: horizontal direction (base frame) along which the jaws open/close.
    jaw_axis:    which tool axis ("x" or "y") is the jaw opening axis of gripper_frame_link.
    """
    a = np.array([jaw_axis_xy[0], jaw_axis_xy[1], 0.0])
    a /= np.linalg.norm(a)
    z = np.array([0.0, 0.0, -1.0])
    if jaw_axis == "x":
        x, y = a, np.cross(z, a)
    elif jaw_axis == "y":
        y = a
        x = np.cross(y, z)
    else:
        raise ValueError(jaw_axis)
    return make_T(np.column_stack([x, y, z]), position)


def flip_jaw(T) -> np.ndarray:
    """Same top-down grasp with the jaw axis reversed (rotate 180 deg about the tool z axis)."""
    R = T[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
    return make_T(R, T[:3, 3])


def min_jerk(s) -> float:
    """Smooth 0..1 time-scaling (zero velocity and acceleration at both ends)."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5
