import numpy as np
from pynput import keyboard

from ee_sim_env import make_ee_sim_env

# ------- 调参区 -------
POS_STEP = 0.005        # 米/步
ANG_STEP_DEG = 2.0      # 度/步（姿态）
GRIP_STEP = 0.05        # 夹爪 0~1
# ----------------------

keys_down = set()

def on_press(key):
    try:
        keys_down.add(key.char)
    except Exception:
        keys_down.add(key)

def on_release(key):
    try:
        keys_down.discard(key.char)
    except Exception:
        keys_down.discard(key)
    if key == keyboard.Key.esc:
        return False  # 停止监听

def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))

def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1, 0, 0, 0], dtype=float)
    return q / n

def quat_mul(q1, q2):
    # q = [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)

def quat_from_axis_angle(axis, angle_rad):
    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle_rad / 2.0)
    return np.array([np.cos(angle_rad/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)

def apply_small_rot(q, roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    # 这里使用固定世界轴：X=roll, Y=pitch, Z=yaw（够用且直观）
    q = quat_normalize(q)
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    dq = np.array([1, 0, 0, 0], dtype=float)
    if abs(r) > 1e-9:
        dq = quat_mul(quat_from_axis_angle([1,0,0], r), dq)
    if abs(p) > 1e-9:
        dq = quat_mul(quat_from_axis_angle([0,1,0], p), dq)
    if abs(y) > 1e-9:
        dq = quat_mul(quat_from_axis_angle([0,0,1], y), dq)
    # 新姿态 = dq * q （世界轴旋转）
    return quat_normalize(quat_mul(dq, q))

def main(task='sim_insertion'):
    env = make_ee_sim_env(task)
    ts = env.reset()

    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    img = ax.imshow(ts.observation['images']['angle'])
    ax.set_title("Teleop Camera View")
    plt.show()

    # 目标位姿初值来自 observation（最稳）
    L = ts.observation['mocap_pose_left'].copy()   # [x,y,z,w,x,y,z]
    R = ts.observation['mocap_pose_right'].copy()
    gl, gr = 0.0, 0.0

    # 选择控制模式：both/left/right
    mode = "both"

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n=== Bi-manual Teleop ===")
    print("ESC 退出 | 1=左手 2=右手 3=双手 | 空格=停住(不更新)")
    print("左手平移: W/S=+/-Y  A/D=-/+X  Q/E=+/-Z")
    print("右手平移: I/K=+/-Y  J/L=-/+X  U/O=+/-Z")
    print("左手姿态: T/G=roll+/-  Y/H=pitch+/-  R/F=yaw+/-")
    print("右手姿态: P/;=roll+/-  [/']=pitch+/-  ]/=yaw+/-")
    print("夹爪: 左 Z/X 开/合 | 右 N/M 开/合\n")

    try:
        while True:
            if '1' in keys_down: mode = "left"
            if '2' in keys_down: mode = "right"
            if '3' in keys_down: mode = "both"

            if keyboard.Key.space in keys_down:
                # 空格：冻结，不更新
                continue

            # ---- 左手平移 ----
            if mode in ("left", "both"):
                if 'w' in keys_down: L[1] += POS_STEP
                if 's' in keys_down: L[1] -= POS_STEP
                if 'a' in keys_down: L[0] -= POS_STEP
                if 'd' in keys_down: L[0] += POS_STEP
                if 'q' in keys_down: L[2] += POS_STEP
                if 'e' in keys_down: L[2] -= POS_STEP

            # ---- 右手平移 ----
            if mode in ("right", "both"):
                if 'i' in keys_down: R[1] += POS_STEP
                if 'k' in keys_down: R[1] -= POS_STEP
                if 'j' in keys_down: R[0] -= POS_STEP
                if 'l' in keys_down: R[0] += POS_STEP
                if 'u' in keys_down: R[2] += POS_STEP
                if 'o' in keys_down: R[2] -= POS_STEP

            # ---- 左手姿态 ----
            if mode in ("left", "both"):
                roll = pitch = yaw = 0.0
                if 't' in keys_down: roll += ANG_STEP_DEG
                if 'g' in keys_down: roll -= ANG_STEP_DEG
                if 'y' in keys_down: pitch += ANG_STEP_DEG
                if 'h' in keys_down: pitch -= ANG_STEP_DEG
                if 'r' in keys_down: yaw += ANG_STEP_DEG
                if 'f' in keys_down: yaw -= ANG_STEP_DEG
                if roll or pitch or yaw:
                    L[3:7] = apply_small_rot(L[3:7], roll, pitch, yaw)

            # ---- 右手姿态 ----
            if mode in ("right", "both"):
                roll = pitch = yaw = 0.0
                if 'p' in keys_down: roll += ANG_STEP_DEG
                if ';' in keys_down: roll -= ANG_STEP_DEG
                if '[' in keys_down: pitch += ANG_STEP_DEG
                if "'" in keys_down: pitch -= ANG_STEP_DEG
                if ']' in keys_down: yaw += ANG_STEP_DEG
                if '/' in keys_down: yaw -= ANG_STEP_DEG
                if roll or pitch or yaw:
                    R[3:7] = apply_small_rot(R[3:7], roll, pitch, yaw)

            # ---- 夹爪 ----
            if 'z' in keys_down: gl = clamp01(gl + GRIP_STEP)
            if 'x' in keys_down: gl = clamp01(gl - GRIP_STEP)
            if 'n' in keys_down: gr = clamp01(gr + GRIP_STEP)
            if 'm' in keys_down: gr = clamp01(gr - GRIP_STEP)

            action = np.concatenate([L, [gl], R, [gr]]).astype(np.float32)
            ts = env.step(action)

            img.set_data(ts.observation['images']['angle'])
            plt.pause(0.001)

            # 用 env 回写的 mocap pose 作为下一帧基准（防漂、且与仿真一致）
            L = ts.observation['mocap_pose_left'].copy()
            R = ts.observation['mocap_pose_right'].copy()

    finally:
        listener.stop()

if __name__ == "__main__":
    main("sim_insertion")  # 或 "sim_transfer_cube"
