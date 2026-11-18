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

## How the grasp becomes a robot pose

GR-ConvNet outputs four per-pixel maps: quality, cos(2θ), sin(2θ) and width. The best pixel inside the YOLO box gives the grasp centre, angle θ = atan2(sin, cos) / 2 and width (the network's width output times 150, in pixels).

The 2D grasp is lifted to 3D like this:

- The centre pixel plus its depth gives the grasp centre in the base frame.
- The two jaw end points (centre ± width/2 along the grasp direction) are intersected with a horizontal plane at the grasp height. Their separation is the metric width, and their direction is the gripper yaw.
- The pose is a top-down grasp: the tool z axis points down and the jaw axis follows that yaw. The reversed jaw axis is the same grasp, and the shorter joint path is used.

## Safety and limits

- **Top-down only.** GR-ConvNet predicts top-down grasps. The SO101 has 5 joints, so exact top-down poses are reachable only at low heights (about 7 cm above the table at 20 cm radius, 2-3 cm at 30 cm). The hover height is lowered automatically until it is reachable.
- **Path checks.** Straight vertical segments (descend, lift) use Cartesian interpolation with IK at each waypoint. Big moves use joint-space interpolation. Both are checked against `--min-z`, joint jumps and reachability before the arm moves.
- **Occlusion.** The arm hovering over the brick may hide it from the camera. If the brick is not re-detected, the first detection is used. `--infer-first` avoids the problem by running the grasp network before moving.
- **Place height.** The brick is released from the place pose's height (about 9 cm above the table with the default pose).
- Start without `--yes`, with the workspace clear.

## Things to verify on your hardware

- **Calibration.** Project the gripper tip into the camera image, or command a known pose, and check the camera reading matches.
- **Jaw axis.** The default assumes the jaws open along `gripper_frame_link` x. If the gripper is turned 90° at the grasp, use `--jaw-axis y`.
- **Gripper width.** Measure the jaw opening at `gripper.pos = 100` and set `--gripper-max-width`.
- **Table height.** `--min-z 0` assumes the table is at base z = 0.
- **YOLO classes.** The model must have your brick class, or detection will fail.
