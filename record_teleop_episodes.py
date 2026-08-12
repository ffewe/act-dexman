import time
import os
import argparse
from collections import defaultdict

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pynput import keyboard

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from ee_sim_env import make_ee_sim_env
from constants import SIM_TASK_CONFIGS

# ====== teleop step sizes ======
POS_STEP = 0.01         # 平移步长
ANG_STEP_DEG = 2.0      # 旋转步长（度）
GRIP_STEP = 0.05        # 夹爪 0~1
RENDER_CAM = "angle"    # teleop 显示用相机名（需与 env observation 里的 images key 对齐）

# ====== keyboard state ======
keys_down = set()
finish_episode = False
discard_episode = False


def on_press(key):
    global finish_episode, discard_episode
    try:
        keys_down.add(key.char.lower())
    except Exception:
        keys_down.add(key)

    # Enter：结束并保存
    if key == keyboard.Key.enter:
        finish_episode = True
    # Backspace：丢弃本条 episode
    if key == keyboard.Key.backspace:
        discard_episode = True


def on_release(key):
    try:
        keys_down.discard(key.char.lower())
    except Exception:
        keys_down.discard(key)

    # Esc：退出监听（原脚本逻辑）
    if key == keyboard.Key.esc:
        return False  # 退出监听


def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1, 0, 0, 0], dtype=float)
    return q / n


def quat_mul(q1, q2):
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
    # 世界轴增量旋转：直观好用
    q = quat_normalize(q)
    r = np.deg2rad(roll_deg)
    p = np.deg2rad(pitch_deg)
    y = np.deg2rad(yaw_deg)
    dq = np.array([1, 0, 0, 0], dtype=float)
    if abs(r) > 1e-9:
        dq = quat_mul(quat_from_axis_angle([1, 0, 0], r), dq)
    if abs(p) > 1e-9:
        dq = quat_mul(quat_from_axis_angle([0, 1, 0], p), dq)
    if abs(y) > 1e-9:
        dq = quat_mul(quat_from_axis_angle([0, 0, 1], y), dq)
    return quat_normalize(quat_mul(dq, q))


def rollout_teleop_in_ee_env(task_name, episode_len, onscreen_render=True):
    """
    在 ee_sim_env 里用键盘遥操，返回 episode (list of ts)
    Enter：结束并保存
    Backspace：丢弃
    Esc：退出整个程序（原实现是停止键盘监听）
    """
    global finish_episode, discard_episode
    finish_episode = False
    discard_episode = False

    env = make_ee_sim_env(task_name)
    ts = env.reset()
    episode = [ts]

    # 相机窗口（teleop 观察）
    if onscreen_render:
        plt.ion()
        fig, ax = plt.subplots()
        img = ax.imshow(ts.observation["images"][RENDER_CAM])
        ax.set_title("Teleop Camera View (Enter=save, Backspace=discard, ESC=quit)")
        plt.show()
    else:
        img = None

    # 目标位姿初值来自 observation（你 teleop_bimanual 就是这么做的）
    L = ts.observation["mocap_pose_left"].copy()
    R = ts.observation["mocap_pose_right"].copy()
    gl, gr = 0.0, 0.0
    mode = "both"

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\n=== Teleop Recording ===")
    print("Enter=结束并保存 | Backspace=丢弃本条 | ESC=退出程序")
    print("1=左手 2=右手 3=双手 | 空格=停住")
    print("左手平移: W/S=+/-Y  A/D=-/+X  Q/E=+/-Z")
    print("右手平移: I/K=+/-Y  J/L=-/+X  U/O=+/-Z")
    print("左手姿态: T/G=roll+/-  Y/H=pitch+/-  R/F=yaw+/-")
    print("右手姿态: P/;=roll+/-  [/']=pitch+/-  ]/=yaw+/-")
    print("夹爪: 左 Z/X 开/合 | 右 N/M 开/合\n")

    try:
        t = 0
        while True:
            # 你按 Enter/Backspace 触发结束/丢弃，就退出录制循环
            if finish_episode or discard_episode:
                break

            if '1' in keys_down:
                mode = "left"
            if '2' in keys_down:
                mode = "right"
            if '3' in keys_down:
                mode = "both"

            if keyboard.Key.space in keys_down:
                # 冻结：不更新
                if onscreen_render and img is not None:
                    img.set_data(ts.observation["images"][RENDER_CAM])
                    plt.pause(0.001)
                continue

            # ---- 左手平移 ----
            if mode in ("left", "both"):
                if 'w' in keys_down:
                    L[1] += POS_STEP
                if 'c' in keys_down:
                    L[1] -= POS_STEP
                if 'a' in keys_down:
                    L[0] -= POS_STEP
                if 'd' in keys_down:
                    L[0] += POS_STEP
                if 'v' in keys_down:
                    L[2] += POS_STEP
                if 'e' in keys_down:
                    L[2] -= POS_STEP

            # ---- 右手平移 ----
            if mode in ("right", "both"):
                if 'i' in keys_down:
                    R[1] += POS_STEP
                if 'k' in keys_down:
                    R[1] -= POS_STEP
                if 'j' in keys_down:
                    R[0] -= POS_STEP
                if 'l' in keys_down:
                    R[0] += POS_STEP
                if 'u' in keys_down:
                    R[2] += POS_STEP
                if 'o' in keys_down:
                    R[2] -= POS_STEP

            # ---- 左手姿态 ----
            if mode in ("left", "both"):
                roll = pitch = yaw = 0.0
                if 't' in keys_down:
                    roll += ANG_STEP_DEG
                if 'g' in keys_down:
                    roll -= ANG_STEP_DEG
                if 'y' in keys_down:
                    pitch += ANG_STEP_DEG
                if 'h' in keys_down:
                    pitch -= ANG_STEP_DEG
                if 'r' in keys_down:
                    yaw += ANG_STEP_DEG
                if 'f' in keys_down:
                    yaw -= ANG_STEP_DEG
                if roll or pitch or yaw:
                    L[3:7] = apply_small_rot(L[3:7], roll, pitch, yaw)

            # ---- 右手姿态 ----
            if mode in ("right", "both"):
                roll = pitch = yaw = 0.0
                if 'p' in keys_down:
                    roll += ANG_STEP_DEG
                if ';' in keys_down:
                    roll -= ANG_STEP_DEG
                if '[' in keys_down:
                    pitch += ANG_STEP_DEG
                if "'" in keys_down:
                    pitch -= ANG_STEP_DEG
                if ']' in keys_down:
                    yaw += ANG_STEP_DEG
                if '/' in keys_down:
                    yaw -= ANG_STEP_DEG
                if roll or pitch or yaw:
                    R[3:7] = apply_small_rot(R[3:7], roll, pitch, yaw)

            # ---- 夹爪 ----
            if 'z' in keys_down:
                gl = clamp01(gl + GRIP_STEP)
            if 'x' in keys_down:
                gl = clamp01(gl - GRIP_STEP)
            if 'n' in keys_down:
                gr = clamp01(gr + GRIP_STEP)
            if 'm' in keys_down:
                gr = clamp01(gr - GRIP_STEP)

            # ee_sim_env 的 action 是 [L(7)+gl, R(7)+gr] 这种（你 teleop 就是这么拼）
            action = np.concatenate([L, [gl], R, [gr]]).astype(np.float32)
            ts = env.step(action)
            episode.append(ts)

            if onscreen_render and img is not None:
                img.set_data(ts.observation["images"][RENDER_CAM])
                plt.pause(0.001)

            # 回写：保持和仿真一致
            L = ts.observation["mocap_pose_left"].copy()
            R = ts.observation["mocap_pose_right"].copy()

            t += 1

    finally:
        listener.stop()
        if onscreen_render:
            plt.close()

    return episode, discard_episode


def replay_and_save_hdf5(task_name, dataset_dir, episode_idx, joint_traj, subtask_info, onscreen_render=False):
    """
    在 sim_env 里 replay joint_traj，并保存成 hdf5（对齐原格式）
    """
    # make env
    env = make_ee_sim_env(task_name)
    render_cam_name = "angle"
    camera_names = ["cam0", "cam1", "cam2"]

    # 复位到同一初始状态（subtask_info）
    ts = env.reset()
    print("[DEBUG] env.reset")
    episode_replay = [ts]

    peg_force_data = []

    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation["images"][render_cam_name])
        plt.ion()

    for t in range(len(joint_traj)):  # note: replay 会比 episode 多 1 个 step 的细节，你们原脚本后面有截断
        action = joint_traj[t]
        ts = env.step(action)
        episode_replay.append(ts)

        # 采 peg_force（若有）
        if "peg_force" in ts.observation:
            peg_force = np.array(ts.observation["peg_force"]).copy()
        else:
            peg_force = np.zeros(3)

        force_analysis = None
        if "force" in ts.observation:
            force_analysis = np.array(ts.observation["force"]).copy()

        peg_force_data.append({
            "timestep": t,
            "peg_force": peg_force.copy(),
            "force_analysis": force_analysis.copy() if force_analysis is not None else None
        })

        if onscreen_render:
            plt_img.set_data(ts.observation["images"][render_cam_name])
            plt.pause(0.02)

    # 保存 peg_force 数据
    if peg_force_data:
        force_npz_path = os.path.join(dataset_dir, f"episode_{episode_idx}_peg_forces.npz")
        timesteps = [entry["timestep"] for entry in peg_force_data]
        peg_forces = np.array([entry["peg_force"] for entry in peg_force_data])
        np.savez(force_npz_path, timesteps=np.array(timesteps), peg_forces=peg_forces)
        print(f"Saved peg force data to {force_npz_path}")

    # 组织 hdf5 数据
    max_timesteps = len(joint_traj)
    data_dict = defaultdict(list)

    # action / qpos / qvel / force / peg_force / images
    for t in range(max_timesteps):
        ts = episode_replay[t]

        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])

        if "force" in ts.observation:
            data_dict["/observations/force"].append(ts.observation["force"])
        else:
            data_dict["/observations/force"].append(np.zeros(12))

        if "peg_force" in ts.observation:
            data_dict["/observations/peg_force"].append(ts.observation["peg_force"])
        else:
            data_dict["/observations/peg_force"].append(np.zeros(3))

        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(ts.observation["images"][cam_name])

        # action
        data_dict["/action"].append(joint_traj[t])

    # 写 HDF5（对齐原格式）
    t0 = time.time()
    dataset_path = os.path.join(dataset_dir, f"episode_new{episode_idx}")
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs["sim"] = True
        obs = root.create_group("observations")
        image = obs.create_group("images")

        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype="uint8",
                                     chunks=(1, 480, 640, 3))
        _ = obs.create_dataset("qpos", (max_timesteps, 14))
        _ = obs.create_dataset("qvel", (max_timesteps, 14))
        _ = obs.create_dataset("force", (max_timesteps, 12))
        _ = obs.create_dataset("peg_force", (max_timesteps, 3))
        _ = root.create_dataset("action", (max_timesteps, 14))

        # 一次性写入
        for name, array in data_dict.items():
            root[name][...] = np.array(array)

    print(f"Saved: {dataset_path}.hdf5  ({time.time() - t0:.1f}s)")


def main(args):
    task_name = args.task_name
    dataset_dir = args.dataset_dir
    num_episodes = args.num_episodes
    onscreen_render = args.onscreen_render

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]["episode_len"]  # 保留读取，但 teleop 不再用它“卡住/停步”

    saved = 0
    episode_idx = 0

    while saved < num_episodes:
        print(f"\n==== teleop episode {episode_idx} (saved {saved}/{num_episodes}) ====")

        episode, discard = rollout_teleop_in_ee_env(task_name, episode_len, onscreen_render=True)
        if discard:
            print("Discarded this episode (Backspace). Not saving.")
            episode_idx += 1
            continue

        # 提取 joint_traj，并替换夹爪关节（完全照抄 record_sim_episodes.py）
        joint_traj = [ts.observation["qpos"] for ts in episode]
        gripper_ctrl_traj = [ts.observation["gripper_ctrl"] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6 + 7] = right_ctrl

        subtask_info = episode[0].observation["env_state"].copy()  # box pose at step0

        replay_and_save_hdf5(task_name, dataset_dir, episode_idx, joint_traj, subtask_info, onscreen_render=onscreen_render)

        saved += 1
        episode_idx += 1

    print(f"\nDone. Saved {saved} episodes to {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--onscreen_render", action="store_true",
                        help="replay 时是否显示 sim_env 的 angle 相机（teleop 阶段默认显示）")
    main(parser.parse_args())
