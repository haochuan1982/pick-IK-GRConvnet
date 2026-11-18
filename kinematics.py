"""SO101 kinematics and trajectory generation with Pinocchio (5 arm joints, gripper locked)."""
import numpy as np
import pinocchio as pin

import util

EE_FRAME = "gripper_frame_link"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class PlanningError(RuntimeError):
    pass


class SO101Kinematics:
    def __init__(self, urdf_path):
        full = pin.buildModelFromUrdf(util.urdf_without_meshes(urdf_path))
        self.model = pin.buildReducedModel(full, [full.getJointId("gripper")], pin.neutral(full))
        names = list(self.model.names)[1:]
        if names != ARM_JOINTS:
            raise RuntimeError(f"unexpected joint order {names}")
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(EE_FRAME)
        self.lo, self.hi = self.model.lowerPositionLimit, self.model.upperPositionLimit

    def fk(self, q) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.fid].homogeneous.copy()

    def ik(self, T_goal, q0, iters=300, tol_p=5e-4, tol_r=2e-3, damp=1e-4, dt=0.5):
        """Damped least-squares IK. Returns (q, position error m, rotation error rad)."""
        goal = pin.SE3(T_goal[:3, :3], T_goal[:3, 3])
        q = np.clip(q0, self.lo, self.hi).copy()
        for _ in range(iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            iMd = self.data.oMf[self.fid].actInv(goal)
            err = pin.log6(iMd).vector
            if np.linalg.norm(err[:3]) < tol_p and np.linalg.norm(err[3:]) < tol_r:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid, pin.LOCAL)
            J = -pin.Jlog6(iMd.inverse()) @ J
            v = -J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), err)
            q = np.clip(pin.integrate(self.model, q, v * dt), self.lo, self.hi)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        err = pin.log6(self.data.oMf[self.fid].actInv(goal)).vector
        return q, float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))

    def plan_linear(self, q_start, T_goal, n=40, max_step=0.35, tol_p=2e-3, tol_r=2e-2):
        """Straight Cartesian line to T_goal (orientation slerped). For moves that keep the
        tool orientation (descend / lift); arbitrary orientations are not reachable by a
        5-DOF arm along the way, so use plan_joint for the large move.

        Every waypoint is solved by IK (warm-started from the previous one). Raises
        PlanningError if a waypoint is unreachable or the joints jump (IK branch change).
        Returns an (n+1, 5) array of joint angles in rad, first row = q_start.
        """
        T0 = self.fk(q_start)
        R0, p0, Rg, pg = T0[:3, :3], T0[:3, 3], T_goal[:3, :3], T_goal[:3, 3]
        w = pin.log3(R0.T @ Rg)
        traj = [np.asarray(q_start, float)]
        for i in range(1, n + 1):
            s = i / n
            T = util.make_T(R0 @ pin.exp3(w * s), p0 + s * (pg - p0))
            q, ep, er = self.ik(T, traj[-1])
            if ep > tol_p or er > tol_r:
                raise PlanningError(f"waypoint {i}/{n} unreachable (pos err {ep*1e3:.1f} mm, rot err {np.degrees(er):.1f} deg)")
            if np.abs(q - traj[-1]).max() > max_step:
                raise PlanningError(f"waypoint {i}/{n}: joint jump {np.degrees(np.abs(q - traj[-1]).max()):.0f} deg")
            traj.append(q)
        return np.array(traj)

    def plan_joint(self, q_start, T_goal, n=40, min_z=0.0, tol_p=2e-3, tol_r=2e-2):
        """IK at the goal, then straight-line interpolation in joint space (checked against the floor)."""
        q_goal, ep, er = self.ik(T_goal, q_start)
        if ep > tol_p or er > tol_r:
            raise PlanningError(f"goal unreachable (pos err {ep*1e3:.1f} mm, rot err {np.degrees(er):.1f} deg)")
        traj = np.linspace(q_start, q_goal, n + 1)
        low = min(self.fk(q)[2, 3] for q in traj)
        if low < min_z:
            raise PlanningError(f"joint-space path dips to z={low:.3f} m (floor {min_z} m)")
        return traj

    def plan_best(self, q_start, T_goal, n=40, min_z=0.0):
        """Plan to T_goal or to the same grasp with the jaw axis reversed; keep the shorter joint path.

        Uses a Cartesian line if the goal has (nearly) the current orientation, else joint space.
        """
        R_rel = self.fk(q_start)[:3, :3].T @ T_goal[:3, :3]
        same_orientation = np.linalg.norm(pin.log3(R_rel)) < np.radians(3)
        best, errors = None, []
        for T in (T_goal, util.flip_jaw(T_goal)):
            try:
                tr = (self.plan_linear(q_start, T, n) if same_orientation
                      else self.plan_joint(q_start, T, n, min_z))
            except PlanningError as e:
                errors.append(str(e))
                continue
            cost = np.abs(np.diff(tr, axis=0)).sum()
            if best is None or cost < best[0]:
                best = (cost, tr)
        if best is None:
            raise PlanningError("; ".join(errors))
        return best[1]
