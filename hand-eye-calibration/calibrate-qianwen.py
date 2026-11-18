# so101_handeye_eye_to_hand.py
import cv2
import numpy as np
import pyrealsense2 as rs
from ikpy.chain import Chain
from lerobot.common.robot_devices.so101 import So101Robot
import time

# ====================== 【请修改这里！】 ======================
URDF_PATH = "./so101.urdf"          # 你的so101 urdf路径
CHESSBOARD_SIZE = (9, 6)            # 棋盘内角点数量
SQUARE_SIZE = 0.025                 # 棋盘格子边长(m)
# 相机内参 fx,fy,cx,cy （你要先用realsense标定得到）
K = np.array([
    [610.0, 0.0, 320.0],
    [0.0, 610.0, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)  # 畸变系数
# ikpy active mask：前5个是臂关节，后面夹爪固定不参与运动学
ACTIVE_MASK = [True, True, True, True, True, False]
# ============================================================

# 1. 加载 SO101 运动学链
arm_chain = Chain.from_urdf_file(
    URDF_PATH,
    active_links_mask=ACTIVE_MASK
)
print("ikpy chain loaded, active joints count:", sum(ACTIVE_MASK))

# 2. 启动 Realsense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

# 3. 初始化 SO101
robot = So101Robot()
robot.connect()
time.sleep(1.0)

# 存储标定数据
R_board2cam_list = []
t_board2cam_list = []
R_flange2base_list = []
t_flange2base_list = []

# 棋盘对象
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0],0:CHESSBOARD_SIZE[1]].T.reshape(-1,2) * SQUARE_SIZE

print("===== 手眼标定采集流程 =====")
print("每按一次 s 采集一组；按 q 结束采集，开始计算")

while True:
    # 获取相机帧
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    if not color_frame:
        continue
    img = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 找棋盘格
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        # 亚像素
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        # PnP 棋盘到相机
        rvec, tvec = cv2.solvePnP(objp, corners_sub, K, DIST_COEFFS)
        R_board2cam, _ = cv2.Rodrigues(rvec)
    cv2.imshow("chessboard", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        if not ret:
            print("当前画面没有识别到棋盘，跳过！")
            continue
        # 读取 SO101 当前关节（lerobot读出顺序：[j1,j2,j3,j4,j5, gripper]）
        q_all = robot.joint_positions
        q_arm = q_all[:5]
        # ikpy forward kinematics 得到法兰 T_flange_base
        T_flange_base = arm_chain.forward_kinematics(q_arm)
        R_flange2base = T_flange_base[:3, :3]
        t_flange2base = T_flange_base[:3, 3].reshape(-1,1)

        R_board2cam_list.append(R_board2cam)
        t_board2cam_list.append(tvec.reshape(-1,1))
        R_flange2base_list.append(R_flange2base)
        t_flange2base_list.append(t_flange2base)
        print(f"✅ 采集一组，当前总数：{len(R_board2cam_list)}")

cv2.destroyAllWindows()
pipeline.stop()
robot.disconnect()

if len(R_board2cam_list) < 6:
    print("❌ 采集组数太少，至少6组以上，推荐10~15组！")
else:
    # OpenCV calibHandEye: 眼在手外公式
    # calibHandEye(R_flange_base, t_flange_base, R_board_cam, t_board_cam, ...)
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_flange2base_list,
        t_gripper2base=t_flange2base_list,
        R_target2cam=R_board2cam_list,
        t_target2cam=t_board2cam_list,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    # 拼成齐次矩阵 T_cam_base
    T_cam2base = np.eye(4)
    T_cam2base[:3, :3] = R_cam2base
    T_cam2base[:3, 3] = t_cam2base.flatten()
    print("\n===== 标定结果 T_cam2base (相机到基座) =====")
    print(T_cam2base)
    np.savetxt("so101_T_cam2base.txt", T_cam2base)
    print("✅ 已保存到 so101_T_cam2base.txt")

