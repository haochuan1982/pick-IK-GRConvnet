"""Sample inference with the exported GR-ConvNet OpenVINO IR, using a fake RGB-D image.

Run from the convert directory:
    python infer_grconvnet.py [path/to/model.xml]

The fake input only shows the data flow; the grasp it predicts is meaningless.
"""
import sys

import numpy as np
import openvino as ov

MODEL_XML = sys.argv[1] if len(sys.argv) > 1 else "models/grconvnet-jacquard-v2/model.xml"

core = ov.Core()
model = core.read_model(MODEL_XML)
compiled = core.compile_model(model, "CPU")

# Input shape is NCHW, e.g. [1, 4, 480, 640]. Dimensions may be dynamic (e.g. [?, 4, ?, ?]),
# in which case any size works and we fall back to 480x640.
pshape = model.input(0).partial_shape
print("Model input partial shape:", pshape)


def dim_or(dim, default):
    return dim.get_length() if dim.is_static else default


channels = dim_or(pshape[1], 4)
height = dim_or(pshape[2], 480)
width = dim_or(pshape[3], 640)
print(f"Using input: channels={channels} height={height} width={width}")

# --- Fake RGB-D image -------------------------------------------------------
rng = np.random.default_rng(0)
rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
depth = np.tile(np.linspace(0.3, 0.6, width, dtype=np.float32), (height, 1))  # metres

# --- Preprocess (simplified version of the original repo) -------------------
rgb_f = rgb.astype(np.float32) / 255.0
rgb_f -= rgb_f.mean()
depth_f = np.clip(depth - depth.mean(), -1, 1)

x = np.concatenate([rgb_f.transpose(2, 0, 1), depth_f[None]], axis=0)  # CHW, 4 channels
x = x[None].astype(np.float32)  # NCHW
assert x.shape == (1, channels, height, width), x.shape

# --- Inference ---------------------------------------------------------------
result = compiled(x)
# GenerativeResnet.forward returns (pos, cos, sin, width); check the order if yours differs
pos, cos, sin, wid = (result[compiled.output(i)] for i in range(4))
print("Output shapes:", pos.shape, cos.shape, sin.shape, wid.shape)

# --- Postprocess -------------------------------------------------------------
q_img = pos[0, 0]                          # grasp quality map
angle_img = 0.5 * np.arctan2(sin[0, 0], cos[0, 0])  # grasp angle in radians
width_img = wid[0, 0] * 150.0              # gripper width in pixels (original scale factor)

y, x_px = np.unravel_index(np.argmax(q_img), q_img.shape)
print(
    f"Best grasp: center=({x_px}, {y}) quality={q_img[y, x_px]:.3f} "
    f"angle={np.degrees(angle_img[y, x_px]):.1f} deg width={width_img[y, x_px]:.1f} px"
)
