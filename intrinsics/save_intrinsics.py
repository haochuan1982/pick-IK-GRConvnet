import argparse
import sys

import numpy as np
import pyrealsense2 as rs


def list_devices():
    devices = []
    for d in rs.context().query_devices():
        devices.append({
            "name": d.get_info(rs.camera_info.name),
            "serial": d.get_info(rs.camera_info.serial_number),
        })
    return devices


def model_of(name):
    # "Intel RealSense D435I" -> "D435I" (exact token, so D435 does not match D435I)
    return name.split()[-1].upper()


def select_serial(args, devices):
    if args.serial:
        if args.serial not in [d["serial"] for d in devices]:
            sys.exit(f"No connected device with serial {args.serial}")
        return args.serial
    matches = [d for d in devices if model_of(d["name"]) == args.model.upper()]
    if not matches:
        sys.exit(f"No connected {args.model} found. Use --list to see devices.")
    if len(matches) > 1:
        sys.exit(f"Multiple {args.model} cameras connected; pick one with --serial.")
    return matches[0]["serial"]


def K(i):
    return np.array([[i.fx, 0, i.ppx],
                     [0, i.fy, i.ppy],
                     [0,    0,     1]])


def main():
    p = argparse.ArgumentParser(description="Save RealSense intrinsics to an .npz file.")
    p.add_argument("--model", choices=["D435i", "D435"], default="D435i",
                   help="camera model to use (default: D435i)")
    p.add_argument("--serial", help="select by serial number instead of model")
    p.add_argument("--list", action="store_true", help="list connected devices and exit")
    p.add_argument("--output", help="output file (default: intrinsics_<model>.npz)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    devices = list_devices()
    if args.list:
        for d in devices:
            print(f"{d['name']}  serial={d['serial']}")
        return
    if not devices:
        sys.exit("No RealSense device connected.")

    serial = select_serial(args, devices)
    name = next(d["name"] for d in devices if d["serial"] == serial)
    output = args.output or f"intrinsics_{model_of(name).lower()}.npz"

    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        np.savez(output,
                 color_K=K(color_intr),
                 color_dist=np.array(color_intr.coeffs),
                 color_size=np.array([color_intr.width, color_intr.height]),
                 depth_K=K(depth_intr),
                 depth_dist=np.array(depth_intr.coeffs),
                 depth_size=np.array([depth_intr.width, depth_intr.height]),
                 depth_scale=profile.get_device().first_depth_sensor().get_depth_scale(),
                 model=name,
                 serial=serial)
        print(f"Saved {output} ({name}, serial {serial})")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
