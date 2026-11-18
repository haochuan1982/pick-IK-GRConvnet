"""GR-ConvNet grasp generation (OpenVINO) and conversion of the 2D grasp to a 3D gripper pose."""
from dataclasses import dataclass

import cv2
import numpy as np
import openvino as ov

import util

CROP = 224           # the network was trained on 224x224 native-resolution crops
WIDTH_SCALE = 150.0  # network width output is width_px / 150


@dataclass
class Grasp2D:
    u: float          # grasp centre, full-image pixels
    v: float
    angle: float      # rad; jaw opening direction in the image is (cos a, sin a) (x right, y down)
    width_px: float
    quality: float


@dataclass
class Grasp3D:
    T_base_grasp: np.ndarray  # tool frame, approach (+z) pointing down
    width_m: float
    center_base: np.ndarray
    grasp2d: Grasp2D


class RoboticGrasp:
    def __init__(self, model_xml, device="CPU", jaw_axis="x"):
        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(model_xml)), device)
        self.jaw_axis = jaw_axis
        print(f"[grasp] {len(self.compiled.outputs)} outputs, assumed order: pos, cos, sin, width")

    # ------------------------------------------------------------------ 2D grasp
    @staticmethod
    def _crop_origin(center, shape):
        h, w = shape[:2]
        x0 = int(np.clip(round(center[0] - CROP / 2), 0, w - CROP))
        y0 = int(np.clip(round(center[1] - CROP / 2), 0, h - CROP))
        return x0, y0

    def predict_2d(self, rgb, depth_m, center, box=None) -> Grasp2D:
        """rgb HxWx3 uint8, depth_m HxW float (0 = invalid), center/box in full-image pixels.

        The 224x224 crop is centred on the detected object; the best grasp is searched
        inside `box` only, so the grasp always lands on the object.
        """
        x0, y0 = self._crop_origin(center, rgb.shape)
        rgb_c = rgb[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255.0
        depth_c = depth_m[y0:y0 + CROP, x0:x0 + CROP].copy()
        valid = depth_c > 0
        if not valid.any():
            raise ValueError("no valid depth in the crop")
        depth_c[~valid] = np.median(depth_c[valid])

        # same normalisation as the original repo (CameraData): zero-mean, depth clipped to [-1, 1]
        rgb_c -= rgb_c.mean()
        depth_c = np.clip(depth_c - depth_c.mean(), -1, 1)
        # NOTE channel order is [depth, R, G, B], as in robotic-grasping's CameraData.get_data
        x = np.concatenate([depth_c[None], rgb_c.transpose(2, 0, 1)], 0)[None].astype(np.float32)

        res = self.compiled(x)
        pos, cos, sin, wid = (res[self.compiled.output(i)][0, 0] for i in range(4))
        q = cv2.GaussianBlur(pos, (0, 0), 2.0)
        ang = cv2.GaussianBlur(0.5 * np.arctan2(sin, cos), (0, 0), 2.0)
        width = cv2.GaussianBlur(wid * WIDTH_SCALE, (0, 0), 1.0)

        mask = np.ones_like(q, bool)
        if box is not None:
            bx1, by1, bx2, by2 = (int(round(b)) for b in box)
            mask[:] = False
            mask[max(by1 - y0, 0):max(by2 - y0, 0), max(bx1 - x0, 0):max(bx2 - x0, 0)] = True
            if not mask.any():
                mask[:] = True
        qm = np.where(mask, q, -np.inf)
        cy, cx = np.unravel_index(np.argmax(qm), qm.shape)
        return Grasp2D(u=float(x0 + cx), v=float(y0 + cy), angle=float(ang[cy, cx]),
                       width_px=float(width[cy, cx]), quality=float(q[cy, cx]))

    # ------------------------------------------------------------------ 2D -> 3D
    def to_3d(self, g: Grasp2D, depth_m, K, T_base_cam) -> Grasp3D:
        """Lift the 2D grasp into a base-frame gripper pose.

        The camera looks obliquely at the table, so the image angle is NOT the gripper yaw.
        Instead the two jaw end points (centre +/- width/2 along the grasp direction) are
        intersected with the horizontal plane through the grasp centre; their separation
        gives the metric width and their direction the yaw in the base frame.
        """
        z = util.robust_depth(depth_m, g.u, g.v)
        center = util.transform_point(T_base_cam, util.deproject(g.u, g.v, z, K))
        d = np.array([np.cos(g.angle), np.sin(g.angle)]) * g.width_px / 2
        a = util.pixel_to_base_on_plane(g.u - d[0], g.v - d[1], K, T_base_cam, center[2])
        b = util.pixel_to_base_on_plane(g.u + d[0], g.v + d[1], K, T_base_cam, center[2])
        axis = (b - a)[:2]
        width_m = float(np.linalg.norm(axis))
        T = util.topdown_pose(center, axis, self.jaw_axis)
        return Grasp3D(T, width_m, center, g)
