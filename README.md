# SO101 Hand-Eye Calibration

`hand_eye_calibration.py` calibrates a camera against a LeRobot SO101 arm using a chessboard and OpenCV's `calibrateHandEye`.

| Mode | Setup | Result |
|------|-------|--------|
| `eye-in-hand` | Camera mounted on the arm (wrist cam), board fixed in the world | `T_gripper_cam` |
| `eye-to-hand` | Camera fixed in the world, board held by / attached to the gripper | `T_base_cam` |

## Requirements

- Python 3.10+
- An SO101 arm that has already been calibrated with LeRobot
- A printed chessboard on a rigid, flat surface
- The SO101 URDF from the [SO-ARM100 repo](https://github.com/TheRobotStudio/SO-ARM100) (e.g. `so101_new_calib.urdf`). The end-effector frame must be named `gripper_frame_link`.

```bash
pip install -r requirements.txt
```

## Camera intrinsics

Pass an `.npz` file with `K` (3x3 camera matrix) and `dist` (distortion coefficients) via `--intrinsics`. Without it, the script uses a rough guess (focal length = image width) and warns you, so calibrate the camera intrinsics first for good results.

```python
np.savez("intrinsics.npz", K=K, dist=dist)
```

## Usage

### 1. Capture and solve

```bash
python hand_eye_calibration.py capture \
  --port /dev/ttyACM0 \
  --robot-id my_follower_arm \
  --urdf so101_new_calib.urdf \
  --camera 0 \
  --mode eye-in-hand \
  --board 9x6 \
  --square 0.025 \
  --intrinsics intrinsics.npz
```

A preview window opens first and marks the board (corners and axes) when it is detected. Torque is disabled so you can pose the arm by hand. Keys work in the preview window or in the terminal running the script:

- `s`: save a sample (needs the board detected; autosaved to the samples file)
- `u`: undo the last sample
- `c`: calculate, available once at least 10 samples are saved
- `q`: quit without solving

### 2. Re-solve from saved samples

```bash
python hand_eye_calibration.py solve --samples samples.json --mode eye-in-hand
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | required | `eye-in-hand` or `eye-to-hand` |
| `--port` | required (capture) | Serial port of the arm |
| `--robot-id` | `my_follower_arm` | ID used during LeRobot calibration |
| `--urdf` | required (capture) | Path to the SO101 URDF |
| `--camera` | `0` | Camera index or device path |
| `--width`, `--height` | 640, 480 | Capture resolution |
| `--board` | `9x6` | Chessboard **inner corners**, columns x rows |
| `--square` | `0.025` | Square size in meters |
| `--intrinsics` | none | `.npz` with `K` and `dist` |
| `--samples` | `samples.json` | Samples file (written by capture, read by solve) |
| `--output` | `hand_eye_result.json` | Result file |
| `--method` | auto | Force `TSAI`, `PARK`, `HORAUD`, `ANDREFF` or `DANIILIDIS` |

## Tips for good results

- Collect at least 10 samples (15-30 is better).
- Vary orientation as well as position. Rotating about a single axis gives a poorly conditioned solve.
- Keep the board fully visible and in focus, and avoid motion blur and glare.
- Use a rigid, flat board and measure the square size accurately.
- Check the consistency table in the output. The translation std of the supposedly constant board pose should be a few millimeters. If it is much larger, re-check the intrinsics, the board size and the URDF.

## Output

`hand_eye_result.json` contains the mode, the chosen method, the 4x4 transform (translation in meters), and the transforms from all methods.

`samples.json` stores the intrinsics and, per sample, the joint angles, `T_base_gripper` (from URDF forward kinematics) and `T_cam_target` (from `solvePnP`).

## Notes

- The arm is in free-drive during capture, so support it while you move it.
- Joint angles are read in degrees (`use_degrees=True`). Forward kinematics uses the five arm joints and ignores the gripper.
- The LeRobot API names (`SO101Follower`, `RobotKinematics`) can differ between versions. If an import fails, adjust it to match your installed version.
