"""SO101 follower arm: joint I/O through lerobot, planning through Pinocchio."""
import time

import numpy as np

import util
from kinematics import ARM_JOINTS, SO101Kinematics

try:  # lerobot >= 0.4
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
except ImportError:  # older lerobot
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


class ArmController:
    def __init__(self, port, robot_id, urdf_path, hz=30.0, max_joint_speed=0.5, min_z=0.0):
        self.kin = SO101Kinematics(urdf_path)
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, use_degrees=True))
        self.hz, self.max_joint_speed, self.min_z = hz, max_joint_speed, min_z
        self.gripper_cmd = None

    def connect(self):
        self.robot.connect()
        self.gripper_cmd = self.read_obs()["gripper.pos"]

    def disconnect(self):
        self.robot.disconnect()

    def read_obs(self):
        return self.robot.get_observation()

    def get_q(self) -> np.ndarray:
        """Current arm joints in rad (URDF convention)."""
        obs = self.read_obs()
        return np.deg2rad([obs[f"{j}.pos"] for j in ARM_JOINTS])

    def get_pose(self) -> np.ndarray:
        return self.kin.fk(self.get_q())

    def _send(self, q_rad):
        action = {f"{j}.pos": float(d) for j, d in zip(ARM_JOINTS, np.rad2deg(q_rad))}
        action["gripper.pos"] = float(self.gripper_cmd)
        self.robot.send_action(action)

    def set_gripper(self, pos, wait=1.0):
        """pos in lerobot gripper units, 0 (closed) .. 100 (open)."""
        self.gripper_cmd = float(np.clip(pos, 0, 100))
        self._send(self.get_q())
        time.sleep(wait)

    def go_home(self, joints_deg, gripper_pos):
        """Joint-space move to a known joint configuration (degrees, arm joints only) and set the gripper."""
        q_goal = np.deg2rad(joints_deg)
        traj = np.linspace(self.get_q(), q_goal, 41)
        low = min(self.kin.fk(q)[2, 3] for q in traj)
        if low < self.min_z:
            raise ValueError(f"joint path dips to z={low:.3f} m")
        self.execute(traj)
        self.set_gripper(gripper_pos, wait=0.5)

    def plan(self, T_goal, n=40) -> np.ndarray:
        """Plan (and validate) a Cartesian path from the current pose. Does not move the arm."""
        if T_goal[2, 3] < self.min_z:
            raise ValueError(f"target z {T_goal[2, 3]:.3f} m is below the safety floor {self.min_z} m")
        return self.kin.plan_best(self.get_q(), T_goal, n, self.min_z)

    def execute(self, traj):
        """Stream a planned joint path with minimum-jerk timing."""
        traj = np.asarray(traj)
        travel = np.abs(np.diff(traj, axis=0)).sum(0).max()
        duration = max(1.5, 1.9 * travel / self.max_joint_speed)  # 1.875 = peak speed / mean speed
        steps = int(duration * self.hz)
        t0 = time.time()
        for k in range(1, steps + 1):
            f = util.min_jerk(k / steps) * (len(traj) - 1)
            i = min(int(f), len(traj) - 2)
            q = traj[i] + (f - i) * (traj[i + 1] - traj[i])
            self._send(q)
            time.sleep(max(0.0, t0 + k / self.hz - time.time()))
        self._send(traj[-1])
        time.sleep(0.3)
