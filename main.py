#!/usr/bin/env python3
"""Pick a brick with the SO101: YOLO (NPU) -> hover -> GR-ConvNet grasp -> Pinocchio trajectory -> grasp -> lift.

Example
  python main.py --port /dev/ttyACM0 --robot-id my_follower_arm \
      --yolo ~/models/YOLO11s-fp16-ov/yolo11s.xml --class-name brick
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import util
from armController import ArmController
from kinematics import PlanningError
from realsenseCamera import RealSenseCamera
from roboticGrasp import RoboticGrasp
from yoloDetection import YoloDetector

ROOT = Path(__file__).resolve().parent.parent


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="serial port of the arm, e.g. /dev/ttyACM0")
    p.add_argument("--robot-id", default="my_follower_arm")
    p.add_argument("--yolo", required=True, help="YOLO OpenVINO IR (.xml), e.g. YOLO11s-fp16-ov/yolo11s.xml")
    p.add_argument("--yolo-device", default="NPU")
    p.add_argument("--class-name", default="brick")
    p.add_argument("--class-id", type=int, default=None, help="overrides --class-name")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--grasp-model", default=str(ROOT / "convert-robotic-grasping/models/grconvnet-jacquard-v2/model.xml"))
    p.add_argument("--calib", default=str(ROOT / "hand-eye-calibration/hand_eye_result.json"))
    p.add_argument("--urdf", default=str(ROOT / "hand-eye-calibration/so101_new_calib.urdf"))
    p.add_argument("--jaw-axis", choices=["x", "y"], default="x",
                   help="axis of gripper_frame_link along which the jaws open (verify on the real arm)")
    p.add_argument("--hover", type=float, default=0.06, help="requested hover height above the grasp (m); lowered if unreachable")
    p.add_argument("--grasp-depth", type=float, default=0.01, help="how far below the brick's top surface the TCP goes (m)")
    p.add_argument("--width-margin", type=float, default=0.015, help="extra jaw opening over the predicted width (m)")
    p.add_argument("--gripper-max-width", type=float, default=0.10,
                   help="jaw opening (m) at lerobot gripper.pos = 100; measure on your gripper")
    p.add_argument("--gripper-closed", type=float, default=0.0, help="gripper.pos used to close")
    p.add_argument("--home", type=float, nargs=5, default=[0.0, -32.747252747252745, 35.34065934065934, 90.0, 0.0],
                   metavar=("PAN", "LIFT", "ELBOW", "FLEX", "ROLL"), help="home joints in degrees (start and end pose)")
    p.add_argument("--home-gripper", type=float, default=15.0, help="gripper.pos at home")
    p.add_argument("--place", type=float, nargs=5, default=[-63.529411764705884, -33.61309274024339, 27.26453357368183, 92.80575539568346, -0.6732263076126372],
                   metavar=("PAN", "LIFT", "ELBOW", "FLEX", "ROLL"), help="place joints in degrees")
    p.add_argument("--place-gripper", type=float, default=15.0, help="gripper.pos used to release the brick")
    p.add_argument("--min-z", type=float, default=0.0, help="safety floor for the TCP in the base frame (table height)")
    p.add_argument("--speed", type=float, default=0.5, help="max joint speed (rad/s)")
    p.add_argument("--infer-first", action="store_true",
                   help="run the grasp network on the first frame instead of after hovering (the arm may occlude the brick)")
    p.add_argument("--out", default="out", help="folder for debug images")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation before each motion")
    return p.parse_args()


def confirm(args, msg):
    if args.yes:
        return
    if input(f"{msg}  [Enter = go, q = abort] ").strip().lower() == "q":
        raise KeyboardInterrupt


def pick_detection(detector, cam, class_id, tries=30):
    for _ in range(tries):
        rgb, depth = cam.read()
        dets = detector.detect(rgb, class_id)
        if dets:
            return rgb, depth, dets[0]
    return None


def width_to_gripper_pos(args, width_m):
    return float(np.clip((width_m + args.width_margin) / args.gripper_max_width * 100.0, 0, 100))


def plan_raised(arm, T, heights, label):
    """Plan to T raised by the largest height in `heights` that is reachable with the tool pointing down."""
    last = None
    for h in heights:
        Th = T.copy()
        Th[2, 3] += h
        try:
            return arm.plan(Th), h
        except (PlanningError, ValueError) as e:
            last = e
    raise PlanningError(f"{label}: no reachable height in {heights} ({last})")


def lift_traj_z(arm, traj):
    return float(arm.kin.fk(traj[-1])[2, 3])


def save_debug(out, name, rgb, det=None, g2d=None):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if det is not None:
        x1, y1, x2, y2 = (int(v) for v in det.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{det.label} {det.conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if g2d is not None:
        c = np.array([g2d.u, g2d.v])
        d = np.array([np.cos(g2d.angle), np.sin(g2d.angle)]) * g2d.width_px / 2
        cv2.line(img, tuple(int(v) for v in c - d), tuple(int(v) for v in c + d), (0, 0, 255), 2)
        cv2.circle(img, (int(c[0]), int(c[1])), 4, (255, 0, 0), -1)
    cv2.imwrite(str(Path(out) / name), img)


def main():
    args = parse()
    Path(args.out).mkdir(exist_ok=True)
    T_base_cam = util.load_T_base_cam(args.calib)

    detector = YoloDetector(args.yolo, args.yolo_device, conf=args.conf)
    class_id = args.class_id if args.class_id is not None else detector.class_id_of(args.class_name)
    grasper = RoboticGrasp(args.grasp_model, "CPU", args.jaw_axis)
    cam = RealSenseCamera()
    cam.warmup()
    arm = ArmController(args.port, args.robot_id, args.urdf, max_joint_speed=args.speed, min_z=args.min_z)
    arm.connect()
    try:
        confirm(args, f"Move to the home pose {np.round(args.home, 1).tolist()}?")
        arm.go_home(args.home, args.home_gripper)

        # 1-2. camera + YOLO detection of the brick
        found = pick_detection(detector, cam, class_id)
        if found is None:
            sys.exit("no brick detected")
        rgb, depth, det = found
        save_debug(args.out, "detection.jpg", rgb, det)

        # 3-4. pixel + depth -> camera xyz -> base xyz
        u, v = det.center
        x1, y1, x2, y2 = det.box
        # depth of the brick's top: median over the central part of the box
        cu, cv_ = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        win = depth[int(y1 + cv_):int(y2 - cv_) + 1, int(x1 + cu):int(x2 - cu) + 1]
        win = win[win > 0]
        if win.size == 0:
            sys.exit("no valid depth on the brick")
        p_base = util.transform_point(T_base_cam, util.deproject(u, v, float(np.median(win)), cam.K))
        print(f"brick: conf {det.conf:.2f}, pixel ({u:.0f}, {v:.0f}), base xyz {np.round(p_base, 3)} m")

        # 5. move above the brick (jaws tangential, tool pointing down), path planned by Pinocchio
        radial = p_base[:2] / max(np.linalg.norm(p_base[:2]), 1e-6)
        T_hover0 = util.topdown_pose(p_base, [-radial[1], radial[0]], args.jaw_axis)
        traj, h = plan_raised(arm, T_hover0, [args.hover, 0.05, 0.04, 0.03, 0.02], "hover")
        print(f"hover {h * 100:.0f} cm above the brick surface; {len(traj)} waypoints")
        confirm(args, "Move above the brick?")
        arm.execute(traj)

        # 6. grasp network on RGB + depth -> 2D grasp -> target matrix
        if not args.infer_first:
            time.sleep(0.5)
            again = pick_detection(detector, cam, class_id, tries=10)
            if again is not None:
                rgb, depth, det = again
            else:
                print("brick not re-detected from the hover pose (occluded?); using the first detection")
        g2d = grasper.predict_2d(rgb, depth, det.center, det.box)
        g3d = grasper.to_3d(g2d, depth, cam.K, T_base_cam)
        save_debug(args.out, "grasp.jpg", rgb, det, g2d)
        print(f"grasp: quality {g2d.quality:.2f}, angle {np.degrees(g2d.angle):.0f} deg (image), "
              f"width {g3d.width_m * 100:.1f} cm, base xyz {np.round(g3d.center_base, 3)} m")
        print("target matrix:\n", np.round(g3d.T_base_grasp, 4))

        T_grasp = g3d.T_base_grasp.copy()
        T_grasp[2, 3] = max(T_grasp[2, 3] - args.grasp_depth, args.min_z)

        # rotate to the grasp yaw above the brick (so the descent is a pure vertical line)
        traj, h = plan_raised(arm, T_grasp, [h, 0.05, 0.04, 0.03, 0.02], "pre-grasp")
        confirm(args, f"Rotate to the grasp yaw at {h * 100:.0f} cm above the brick?")
        arm.execute(traj)

        # 7. open the gripper a little wider than the predicted width
        open_pos = width_to_gripper_pos(args, g3d.width_m)
        print(f"opening gripper to {open_pos:.0f}/100 (predicted width {g3d.width_m * 100:.1f} cm)")
        arm.set_gripper(open_pos)

        # 8. straight-down trajectory to the grasp pose, close the gripper
        traj = arm.plan(T_grasp)
        confirm(args, "Descend and grasp?")
        arm.execute(traj)
        arm.set_gripper(args.gripper_closed, wait=1.0)

        # 9. lift straight up to the TCP height of the home pose, then carry the brick to the place pose
        z_home = float(arm.kin.fk(np.deg2rad(args.home))[2, 3])
        T_lift = T_grasp.copy()
        T_lift[2, 3] = z_home
        try:
            lift_traj = arm.plan(T_lift)
        except (PlanningError, ValueError) as e:
            print(f"cannot lift to the home height z={z_home:.3f} m with the tool pointing down ({e}); lifting less")
            lift_traj, lh = plan_raised(arm, T_grasp, [z_home - T_grasp[2, 3], 0.04, 0.03, 0.02], "lift")
        print(f"lifting to z = {lift_traj_z(arm, lift_traj):.3f} m (home z = {z_home:.3f} m)")
        confirm(args, "Lift?")
        arm.execute(lift_traj)

        confirm(args, f"Move to the place pose {np.round(args.place, 1).tolist()}?")
        arm.go_home(args.place, arm.gripper_cmd)  # joint-space move, keeps holding the brick

        # release
        arm.set_gripper(args.place_gripper, wait=1.0)

        # back to the initial pose
        confirm(args, "Return to the home pose?")
        arm.go_home(args.home, args.home_gripper)
        print("done")
    except KeyboardInterrupt:
        print("aborted")
    finally:
        cam.stop()
        arm.disconnect()


if __name__ == "__main__":
    main()
