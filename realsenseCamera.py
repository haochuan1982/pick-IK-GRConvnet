"""RealSense D435i wrapper: colour + depth aligned to colour, intrinsics from the live stream."""
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, serial=None):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=np.float64)
        self.size = (width, height)

    def warmup(self, n=30):
        for _ in range(n):
            self.pipeline.wait_for_frames()

    def read(self):
        """Returns (rgb uint8 HxWx3, depth float32 HxW in metres; 0 = invalid)."""
        frames = self.align.process(self.pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        rgb = np.asarray(color.get_data()).copy()
        depth_m = np.asarray(depth.get_data()).astype(np.float32) * self.depth_scale
        return rgb, depth_m

    def stop(self):
        self.pipeline.stop()
