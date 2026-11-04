#!/usr/bin/env python3
"""Hand-eye calibration for the LeRobot SO101 arm.

Two setups are supported:
  --mode eye-in-hand : camera mounted on the arm (wrist cam). Solves T_gripper<-cam.
  --mode eye-to-hand : camera fixed in the world, board held by the gripper
                       (or fixed to the gripper). Solves T_base<-cam.

Workflow
  1. Capture: a preview window opens first and marks the board when detected. Torque is
     disabled so you can pose the arm by hand. Press "s" to save a sample (joints + board
     pose), "u" to undo, "c" to calculate once >= 10 samples are saved, "q" to quit.
     Keys work in the preview window or in the terminal. Vary positions AND orientations.
  2. Solve: samples are saved to JSON and cv2.calibrateHandEye is run with all five
     methods; the spread between them is a good sanity check.

  Gripper FK comes from the SO101 URDF (SO-ARM100 repo, e.g. so101_new_calib.urdf) via
  lerobot's RobotKinematics (requires `pip install placo`).

Examples
  python hand_eye_calibration.py capture --port /dev/ttyACM0 --urdf SO101/so101_new_calib.urdf \
      --camera 0 --mode eye-in-hand --board 9x6 --square 0.025 --intrinsics intrinsics.npz
  python hand_eye_calibration.py solve --samples samples.json --mode eye-in-hand

Intrinsics file (.npz) must contain `K` (3x3) and `dist` (N,). If omitted, a rough
pinhole guess is used (focal ~ image width) -- accuracy will suffer.
"""

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
EE_FRAME = "gripper_frame_link"

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


MIN_SAMPLES = 10


# ----------------------------------------------------------------------------- helpers
class TerminalKeys:
    """Non-blocking single-key reads from the terminal (so keys work without window focus)."""

    def __enter__(self):
        self.fd = None
        if sys.stdin.isatty():
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read(self):
        if self.fd is None:
            return None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def urdf_without_meshes(urdf_path: str) -> str:
    """Write a temp copy of the URDF with mesh geometry replaced by tiny boxes.

    Forward kinematics only needs joints/links, and placo fails if any referenced mesh
    file is missing. Returns the temp file path.
    """
    text = Path(urdf_path).read_text()
    text = re.sub(r"<mesh\b[^>]*/>", '<box size="0.001 0.001 0.001"/>', text)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


def hand_eye_park(T_a, T_b) -> np.ndarray:
    """Solve X in  (A_i^-1 A_j) X = X (B_i B_j^-1)  with Park & Martin (1994).

    Same convention as cv2.calibrateHandEye: T_a = gripper->base (or base->gripper for
    eye-to-hand), T_b = target->cam. Returns X (cam->gripper, or cam->base).
    """
    alphas, betas, pairs = [], [], []
    for i in range(len(T_a)):
        for j in range(i + 1, len(T_a)):
            A = inv_T(T_a[i]) @ T_a[j]
            B = T_b[i] @ inv_T(T_b[j])
            alphas.append(cv2.Rodrigues(A[:3, :3])[0].ravel())
            betas.append(cv2.Rodrigues(B[:3, :3])[0].ravel())
            pairs.append((A, B))
    M = sum(np.outer(b, a) for a, b in zip(alphas, betas))
    U, _, Vt = np.linalg.svd(M)
    Rx = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    lhs = np.vstack([A[:3, :3] - np.eye(3) for A, _ in pairs])
    rhs = np.concatenate([Rx @ B[:3, 3] - A[:3, 3] for A, B in pairs])
    tx = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return make_T(Rx, tx)


def sample_mismatch(T_a, T_b) -> np.ndarray:
    """Per-sample median |angle(A_ij) - angle(B_ij)| in degrees over all pairs.

    The rotation angle of a relative motion is the same seen from the robot (A) and from
    the camera (B). A large value flags a bad sample, typically a chessboard whose
    orientation was detected flipped by 180 degrees.
    """
    n = len(T_a)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A = inv_T(T_a[i]) @ T_a[j]
            B = T_b[i] @ inv_T(T_b[j])
            D[i, j] = D[j, i] = abs(rot_angle_deg(A[:3, :3]) - rot_angle_deg(B[:3, :3]))
    return np.array([np.median(np.delete(D[i], i)) for i in range(n)])


def fix_board_flips(T_a, T_b, pattern, square, restarts=30):
    """Undo 180-degree board-orientation flips (corner order reversed by the detector).

    A flipped detection equals the true pose rotated 180 deg about the board centre. Each
    sample is either kept or flipped; pick the combination whose camera-side rotation
    angles best agree with the robot-side ones. Returns (corrected T_b, flipped indices).
    """
    n = len(T_a)
    c = np.array([(pattern[0] - 1) * square / 2, (pattern[1] - 1) * square / 2, 0.0])
    Rz = np.diag([-1.0, -1.0, 1.0])
    F = make_T(Rz, c - Rz @ c)
    variants = [[t, t @ F] for t in T_b]
    ang_a = np.zeros((n, n))
    ang_b = np.zeros((n, n, 2, 2))
    for i in range(n):
        for j in range(i + 1, n):
            ang_a[i, j] = rot_angle_deg((inv_T(T_a[i]) @ T_a[j])[:3, :3])
            for fi in range(2):
                for fj in range(2):
                    B = variants[i][fi] @ inv_T(variants[j][fj])
                    ang_b[i, j, fi, fj] = rot_angle_deg(B[:3, :3])

    def cost(mask):
        return sum(abs(ang_a[i, j] - ang_b[i, j, mask[i], mask[j]])
                   for i in range(n) for j in range(i + 1, n))

    rng = np.random.default_rng(0)
    best_mask, best_cost = [0] * n, np.inf
    for r in range(restarts):
        mask = [0] * n if r == 0 else [int(x) for x in rng.integers(0, 2, n)]
        cur = cost(mask)
        improved = True
        while improved:
            improved = False
            for k in range(n):
                mask[k] ^= 1
                new = cost(mask)
                if new < cur - 1e-9:
                    cur, improved = new, True
                else:
                    mask[k] ^= 1
        if cur < best_cost:
            best_mask, best_cost = list(mask), cur
    # the all-flipped solution is equivalent; report the one with fewer flips
    if sum(best_mask) > n / 2:
        best_mask = [1 - m for m in best_mask]
    return [variants[i][best_mask[i]] for i in range(n)], [i for i, m in enumerate(best_mask) if m]


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load_intrinsics(path: str | None, frame_shape) -> tuple[np.ndarray, np.ndarray]:
    if path:
        data = np.load(path)
        # accept plain K/dist or RealSense-style color_K/color_dist
        k_key, d_key = ("K", "dist") if "K" in data.files else ("color_K", "color_dist")
        if k_key not in data.files:
            sys.exit(f"{path}: expected K/dist or color_K/color_dist, found {data.files}")
        if "color_size" in data.files and tuple(data["color_size"]) != tuple(frame_shape[1::-1]):
            print(f"WARNING: intrinsics are for {tuple(data['color_size'])} but frames are "
                  f"{tuple(frame_shape[1::-1])}; set --width/--height to match.", file=sys.stderr)
        return data[k_key].astype(np.float64), data[d_key].astype(np.float64).ravel()
    h, w = frame_shape[:2]
    print("WARNING: no intrinsics given, using a rough guess (f = image width).", file=sys.stderr)
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    return K, np.zeros(5)


def detect_board(gray, pattern, square, K, dist, max_reproj=2.0):
    """Return (T_cam<-target 4x4 or None, corners or None, status string).

    corners is returned whenever the pattern was found, even if the pose was rejected,
    so the preview can show what happened.
    """
    corners = None
    # sector-based detector: much more robust on small / blurry / low-contrast boards
    found, c = False, None
    if min(pattern) > 2:  # SB detector needs both dimensions >= 3
        found, c = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if found:
        corners = c
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, c = cv2.findChessboardCorners(gray, pattern, flags)
        if found:
            corners = cv2.cornerSubPix(
                gray, c, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4),
            )
    if corners is None:
        return None, None, "pattern not found (check --board = INNER corners)"

    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, corners, "solvePnP failed"
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1))))
    if err > max_reproj:
        return None, corners, f"pose rejected: reproj {err:.2f}px > {max_reproj}px"
    R, _ = cv2.Rodrigues(rvec)
    return make_T(R, tvec), corners, f"reproj {err:.2f}px"


# ----------------------------------------------------------------------------- capture
def capture(args):
    from lerobot.model.kinematics import RobotKinematics
    try:  # lerobot >= 0.4
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:  # older lerobot
        from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

    pattern = tuple(int(x) for x in args.board.lower().split("x"))  # inner corners (cols, rows)
    if min(pattern) < 3:
        sys.exit(f"--board {args.board}: OpenCV needs >= 3 inner corners per side "
                 "(inner corners = squares - 1), so use a board with at least 4x4 squares.")

    kin = RobotKinematics(urdf_path=urdf_without_meshes(args.urdf), target_frame_name=EE_FRAME, joint_names=JOINT_NAMES)

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.robot_id, use_degrees=True))
    robot.connect()
    robot.bus.disable_torque()  # free-drive by hand

    cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        robot.disconnect()
        sys.exit(f"Cannot open camera {args.camera}")

    ok, frame = cap.read()
    if not ok:
        robot.disconnect()
        sys.exit("Camera returned no frame")
    K, dist = load_intrinsics(args.intrinsics, frame.shape)

    samples = []
    print(f"Keys (preview window or this terminal): s = save sample, u = undo, "
          f"c = calculate (needs >= {MIN_SAMPLES} samples), q = quit without solving")

    def save_file():
        Path(args.samples).write_text(
            json.dumps({"K": K.tolist(), "dist": dist.tolist(), "board": list(pattern),
                        "square": args.square, "samples": samples}, indent=2))

    do_solve = False
    try:
        with TerminalKeys() as term:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                T_cam_target, corners, status = detect_board(gray, pattern, args.square, K, dist, args.max_reproj)
                detected = T_cam_target is not None

                # preview: mark detection
                vis = frame.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(vis, pattern, corners, True)
                if detected:
                    rvec, _ = cv2.Rodrigues(T_cam_target[:3, :3])
                    cv2.drawFrameAxes(vis, K, dist, rvec, T_cam_target[:3, 3], args.square * 3)
                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(vis, "DETECTED - press s to save" if detected else "NOT DETECTED",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(vis, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(vis, f"samples: {len(samples)}/{MIN_SAMPLES}"
                            + ("  (c = calculate)" if len(samples) >= MIN_SAMPLES else ""),
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("hand-eye capture", vis)

                key = cv2.waitKey(1) & 0xFF
                tkey = term.read()
                if tkey:
                    key = ord(tkey.lower())

                if key == ord("q"):
                    break
                elif key == ord("u") and samples:
                    samples.pop()
                    save_file()
                    print(f"removed; {len(samples)} samples")
                elif key == ord("s"):
                    if not detected:
                        print("board not detected, sample skipped")
                        continue
                    obs = robot.get_observation()
                    q = np.array([obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float64)  # degrees
                    T_base_gripper = np.asarray(kin.forward_kinematics(q), dtype=np.float64)
                    samples.append({
                        "joints_deg": q.tolist(),
                        "T_base_gripper": T_base_gripper.tolist(),
                        "T_cam_target": T_cam_target.tolist(),
                    })
                    save_file()  # autosave so a crash doesn't lose samples
                    print(f"sample {len(samples)} saved, joints = {np.round(q, 1).tolist()}")
                elif key == ord("c"):
                    if len(samples) < MIN_SAMPLES:
                        print(f"need at least {MIN_SAMPLES} samples, have {len(samples)}")
                        continue
                    do_solve = True
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()

    save_file()
    print(f"{len(samples)} samples in {args.samples}")
    if do_solve:
        solve(args)


# ----------------------------------------------------------------------------- solve
def solve(args):
    data = json.loads(Path(args.samples).read_text())
    samples = data["samples"]
    if len(samples) < 3:
        sys.exit("Need at least 3 samples")

    T_bg = [np.array(s["T_base_gripper"]) for s in samples]
    T_ct = [np.array(s["T_cam_target"]) for s in samples]

    if args.mode == "eye-in-hand":
        # inputs: gripper->base, target->cam   => output: cam->gripper
        A = T_bg
    else:
        # eye-to-hand: feed base->gripper, output: cam->base
        A = [inv_T(T) for T in T_bg]

    pattern = tuple(data.get("board") or [int(x) for x in args.board.lower().split("x")])
    square = data.get("square") or args.square
    T_ct, flipped = fix_board_flips(A, T_ct, pattern, square)
    if flipped:
        print(f"corrected 180-degree board flips in samples {flipped}")

    # drop samples whose board pose is still inconsistent with the robot motion
    mism = sample_mismatch(A, T_ct)
    keep = [i for i, m in enumerate(mism) if m <= args.max_mismatch]
    for i, m in enumerate(mism):
        if i not in keep:
            print(f"sample {i}: rejected, rotation mismatch {m:.1f} deg (noisy board pose or bad sample)")
    if len(keep) < 3:
        sys.exit(f"Only {len(keep)} consistent samples left; recapture with a stable board detection.")
    A = [A[i] for i in keep]
    T_ct = [T_ct[i] for i in keep]
    T_bg = [T_bg[i] for i in keep]
    print(f"using {len(keep)}/{len(mism)} samples")

    results = {}
    if hasattr(cv2, "calibrateHandEye"):
        for name, m in METHODS.items():
            try:
                R, t = cv2.calibrateHandEye(
                    [T[:3, :3] for T in A], [T[:3, 3].reshape(3, 1) for T in A],
                    [T[:3, :3] for T in T_ct], [T[:3, 3].reshape(3, 1) for T in T_ct],
                    method=m,
                )
                results[name] = make_T(R, t)
            except cv2.error as e:
                print(f"{name}: failed ({e.msg.strip()})")
    else:  # e.g. OpenCV 5.0 does not expose calibrateHandEye
        print(f"cv2 {cv2.__version__} has no calibrateHandEye; using built-in Park-Martin solver "
              "(install opencv-python==4.12.0.88 for all five methods).")
        results["PARK"] = hand_eye_park(A, T_ct)

    if not results:
        sys.exit("All methods failed")

    label = "T_gripper_cam" if args.mode == "eye-in-hand" else "T_base_cam"
    print(f"\n=== {label} per method (translation in m, rotation as RPY deg) ===")
    for name, T in results.items():
        rpy = cv2.RQDecomp3x3(T[:3, :3])[0]
        print(f"{name:11s} t = {np.round(T[:3, 3], 4)}  rpy = {np.round(rpy, 2)}")

    # Consistency check: the target should be static in a fixed frame.
    #   eye-in-hand: T_base_target = T_bg @ T_gc @ T_ct   (constant)
    #   eye-to-hand: T_gripper_target = inv(T_bg) @ T_bc @ T_ct (constant)
    print("\n=== Consistency (std of the supposedly-constant target pose across samples) ===")
    best, best_err = None, np.inf
    for name, X in results.items():
        if args.mode == "eye-in-hand":
            poses = [Tbg @ X @ Tct for Tbg, Tct in zip(T_bg, T_ct)]
        else:
            poses = [inv_T(Tbg) @ X @ Tct for Tbg, Tct in zip(T_bg, T_ct)]
        t_std = np.std([p[:3, 3] for p in poses], axis=0)
        ref = poses[0]
        r_dev = np.mean([rot_angle_deg(ref[:3, :3].T @ p[:3, :3]) for p in poses])
        err = float(np.linalg.norm(t_std))
        print(f"{name:11s} trans std = {err * 1000:6.2f} mm   mean rot dev = {r_dev:5.2f} deg")
        if err < best_err:
            best, best_err = name, err

    chosen = args.method if args.method else best
    if chosen not in results:
        sys.exit(f"Method {chosen} unavailable")
    X = results[chosen]
    print(f"\nUsing {chosen}")
    print(f"{label} =\n{np.round(X, 5)}")

    out = {"mode": args.mode, "method": chosen, label: X.tolist(),
           "all_methods": {k: v.tolist() for k, v in results.items()}}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"saved to {args.output}")


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--mode", choices=["eye-in-hand", "eye-to-hand"], required=True)
        sp.add_argument("--samples", default="samples.json")
        sp.add_argument("--output", default="hand_eye_result.json")
        sp.add_argument("--board", default="9x6", help="chessboard INNER corners, COLSxROWS")
        sp.add_argument("--square", type=float, default=0.025, help="square size in meters")
        sp.add_argument("--max-mismatch", type=float, default=10.0,
                        help="drop samples whose camera/robot rotation angles disagree by more than this (deg)")
        sp.add_argument("--method", choices=list(METHODS), help="force a method (default: most consistent)")

    c = sub.add_parser("capture", help="collect samples interactively, then solve")
    common(c)
    c.add_argument("--port", required=True, help="serial port, e.g. /dev/ttyACM0")
    c.add_argument("--robot-id", default="my_follower_arm", help="id used during lerobot calibration")
    c.add_argument("--urdf", required=True, help="path to SO101 URDF")
    c.add_argument("--camera", default="0", help="camera index or device/path")
    c.add_argument("--width", type=int, default=640)
    c.add_argument("--height", type=int, default=480)
    c.add_argument("--intrinsics", help=".npz with K and dist")
    c.add_argument("--max-reproj", type=float, default=2.0, help="reject board poses above this RMS reprojection error (px)")
    c.set_defaults(func=capture)

    s = sub.add_parser("solve", help="solve from a saved samples file")
    common(s)
    s.set_defaults(func=solve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
