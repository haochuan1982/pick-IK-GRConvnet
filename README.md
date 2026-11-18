# Pick and place with the SO101 arm

`pick-and-place/main.py` picks a brick with an SO101 arm and drops it at a fixed place pose. It uses a fixed RealSense D435i, a YOLO detector on the NPU, the GR-ConvNet grasp network (OpenVINO) and Pinocchio for kinematics and trajectories.

## Pipeline

1. Move to the home pose (joints and gripper are configurable).
2. Read RGB + depth from the RealSense and detect the brick with YOLO (OpenVINO, NPU, CPU fallback).
3. Pixel + depth + camera intrinsics give the brick position in the camera frame.
4. The hand-eye result (`hand-eye-calibration/hand_eye_result.json`, `T_base_cam`) converts it to the robot base frame.
5. Hover above the brick (tool pointing down). The path is planned with Pinocchio.
6. Run GR-ConvNet on a 224x224 RGB-D crop around the brick. The best grasp (pixel, angle, width) becomes a 4x4 target matrix in the base frame.
7. Rotate to the grasp yaw above the brick, then open the gripper a little wider than the predicted width.
8. Descend in a straight vertical line to the grasp pose and close the gripper.
9. Lift straight up to the home pose's TCP height, move to the place joints, open the gripper to drop the brick, and return home.

Every move is planned and checked before the arm moves. The script asks for confirmation before each one unless you pass `--yes`.

## Files (`pick-and-place/`)

| File | Role |
|---|---|
| `main.py` | The whole pipeline and the command-line options |
| `yoloDetection.py` | `YoloDetector`: OpenVINO YOLO inference, decoding and NMS |
| `roboticGrasp.py` | `RoboticGrasp`: GR-ConvNet inference and 2D grasp to 3D pose |
| `kinematics.py` | `SO101Kinematics`: Pinocchio FK, IK and trajectory planning |
| `armController.py` | `ArmController`: lerobot joint I/O and smooth execution |
| `realsenseCamera.py` | `RealSenseCamera`: colour + depth aligned to colour, live intrinsics |
| `util.py` | Transforms, deprojection, ray-plane intersection, top-down pose |

## Requirements

- An SO101 follower arm, calibrated with LeRobot, and a RealSense D435i fixed above the table.
- A hand-eye calibration of **your** setup (see [hand-eye-calibration.md](hand-eye-calibration.md), use `--mode eye-to-hand --vertical-camera` for a camera looking straight down). Do not reuse another project's result: the transform is specific to one camera mount and one arm.
- The GR-ConvNet IR from [convert-robotic-grasping.md](convert-robotic-grasping.md) in `convert-robotic-grasping/models/grconvnet-jacquard-v2/`.
- A YOLO OpenVINO model that detects bricks. `OpenVINO/YOLO11s-fp16-ov` on Hugging Face is trained on COCO, which has no brick class, so use a brick-trained model (or pass `--class-id` to test with another object). If `metadata.yaml` is next to the `.xml`, class names are read from it.

```bash
pip install -r pick-and-place/requirements.txt   # numpy, opencv, openvino, pin, pyyaml, pyrealsense2, lerobot
```

## Usage

```bash
cd pick-and-place
python main.py --port /dev/ttyACM0 --robot-id my_follower_arm \
    --yolo ~/models/YOLO11s-fp16-ov/yolo11s.xml --class-name brick
```

Debug images are written to `out/`: `detection.jpg` (YOLO box) and `grasp.jpg` (box and chosen grasp). Check them before approving the first motion.

### Options

| Flag | Default | Description |
|---|---|---|
| `--port` | required | Serial port of the arm |
| `--robot-id` | `my_follower_arm` | ID used during LeRobot calibration |
| `--yolo` | required | YOLO OpenVINO IR (`.xml`) |
| `--yolo-device` | `NPU` | OpenVINO device; falls back to CPU if it cannot compile |
| `--class-name` / `--class-id` | `brick` / none | Class to pick; `--class-id` overrides the name |
| `--conf` | `0.4` | YOLO confidence threshold |
| `--grasp-model` | repo model | GR-ConvNet `model.xml` |
| `--calib` | `hand_eye_result.json` | Eye-to-hand calibration |
| `--urdf` | `so101_new_calib.urdf` | SO101 URDF (same one used for calibration) |
| `--home` | `0 -32.7 35.3 90 0` | Home joints in degrees (pan, lift, elbow, flex, roll) |
| `--home-gripper` | `15` | `gripper.pos` at home |
| `--place` | `-63.5 -33.6 27.3 92.8 -0.7` | Place joints in degrees |
| `--place-gripper` | `15` | `gripper.pos` used to release the brick |
| `--hover` | `0.06` | Hover height above the grasp (m); lowered if unreachable |
| `--grasp-depth` | `0.01` | How far below the brick top the TCP goes (m) |
| `--width-margin` | `0.015` | Extra opening over the predicted width (m) |
| `--gripper-max-width` | `0.10` | Jaw opening (m) at `gripper.pos = 100`; measure yours |
| `--gripper-closed` | `0` | `gripper.pos` used to close |
| `--jaw-axis` | `x` | Axis of `gripper_frame_link` along which the jaws open (`x` or `y`) |
| `--min-z` | `0` | TCP safety floor in the base frame (table height) |
| `--speed` | `0.5` | Max joint speed (rad/s) |
| `--infer-first` | off | Run the grasp network on the first frame instead of after hovering |
| `--yes` | off | Skip the confirmation before each motion |


## How the target pose matrix is obtained

The grasp network works in image space. The robot needs a 4x4 matrix `T_base_grasp`: where the gripper tip (`gripper_frame_link`) must be and how it must be oriented, in the robot base frame. `RoboticGrasp.predict_2d` and `RoboticGrasp.to_3d` in `roboticGrasp.py` produce it in these steps.

### 1. 2D grasp from the network

A 224x224 RGB-D crop centred on the YOLO box is the input, with channels ordered `[depth, R, G, B]`. RGB and depth are zero-centred, and depth is clipped to [-1, 1]. The network returns four maps, `Q`, `cos(2θ)`, `sin(2θ)` and `width`, which are blurred slightly.

The best pixel is taken inside the YOLO box, so the grasp always lands on the brick:

```
(u, v)  = argmax Q inside the box          grasp centre, full-image pixels
θ       = atan2(sin2θ, cos2θ) / 2          image angle, rad (jaw direction in the image is (cos θ, sin θ))
w_px    = width * 150                      jaw opening, pixels
```

The network predicts 2θ because a grasp is the same after a 180° turn. `cos` and `sin` are therefore not put into the matrix directly.

### 2. Position (x, y, z)

The depth `z_c` at the grasp pixel is the median of a small window of valid values. With the colour intrinsics `K`, the pixel becomes a point in the camera frame, and the hand-eye result moves it to the base frame:

```
p_cam  = [ (u - cx) * z_c / fx,  (v - cy) * z_c / fy,  z_c ]
p_base = T_base_cam @ [p_cam, 1]
```

`p_base` is the grasp centre on the top surface of the brick. `main.py` lowers its z by `--grasp-depth` so the fingers straddle the brick.

### 3. Orientation (yaw only)

The grasp is top-down, so only the rotation about the vertical axis comes from the network. The image angle is not that yaw, because the camera's axes are not the robot's axes. The yaw is measured on the table instead:

1. Take the two jaw end pixels `(u, v) ± (w_px / 2) * (cos θ, sin θ)`.
2. Intersect each pixel's camera ray with the horizontal plane `z = p_base.z` (base frame). This gives two points A and B on the table.
3. The jaw axis is `a = normalize((B - A)_xy)`, and the metric width is `w_m = |B - A|`.

### 4. Assemble the matrix

The tool z axis (the approach direction) points straight down. The jaw axis (tool x, set with `--jaw-axis`) follows `a`. The tool y axis completes the right-handed frame:

```
z_tool = [0, 0, -1]
x_tool = a                       (jaw axis; use y_tool = a instead for --jaw-axis y)
y_tool = z_tool × x_tool

T_base_grasp = [ x_tool  y_tool  z_tool  p_base ]
               [   0       0       0       1    ]
```

`a` and `-a` describe the same grasp. The planner tries both and keeps the one with the shorter joint path. The open width sent to the gripper is `w_m + --width-margin`.

### 5. From matrix to joint motion

Pinocchio inverse kinematics turns `T_base_grasp` into joint angles for the five arm joints. The arm first rotates to this yaw at hover height, then follows a straight vertical line down to the grasp pose. Each waypoint is checked for reachability before anything moves.

Camera and arm axes do not need to line up, because `T_base_cam` from the hand-eye calibration handles that. A wrong calibration shifts `p_base` and the yaw, so the grasp misses.
