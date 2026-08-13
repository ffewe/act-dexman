import collections
import os
from pathlib import Path

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT


DIANA_XML_DIR = (
    Path(__file__).parent.resolve()
    / "assert"
    / "mujoco_diana7_updated"
    / "mujoco_diana7_updated"
)
DIANA_SCENE_XML = DIANA_XML_DIR / "scene_tactile.xml"

RIGHT_ARM_JOINTS = [f"right_arm_joint_{i}" for i in range(1, 8)]
LEFT_ARM_JOINTS = [f"left_arm_joint_{i}" for i in range(1, 8)]
RIGHT_HAND_JOINTS = [f"joint{i}" for i in range(1, 12)]
LEFT_HAND_JOINTS = [f"J{i}" for i in range(1, 12)]

# Hand values are direct position-control targets for the 11 actuated joints.
# Open is near zero for this XML; close follows each joint's bending direction.
RIGHT_HAND_OPEN = np.zeros(11)
RIGHT_HAND_CLOSE = np.array([-1.10, 0.20, 0.32, -0.85, -0.85, -0.85, -0.85, -0.85, -0.85, -0.85, -0.85])
LEFT_HAND_OPEN = np.zeros(11)
LEFT_HAND_CLOSE = np.array([-0.55, -0.20, -0.32, -0.85, -1.05, -0.85, -1.05, -0.85, -1.05, -0.85, -1.05])

# A conservative reset pose. These are intentionally exposed for the scripted
# policy so data collection can later use the same 16-D action convention.
DIANA_START_ACTION = np.array(
    [
        0.25, 0.55, -0.15, 1.45, 0.00, -0.65, 0.20, 1.0,
        -0.25, 0.55, 0.15, 1.45, 0.00, 0.65, -0.20, 1.0,
    ],
    dtype=np.float64,
)


def make_diana_sim_env():
    physics = mujoco.Physics.from_xml_path(os.fspath(DIANA_SCENE_XML))
    task = DianaPickPegTask(random=False)
    return control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )


def _named_qpos(physics, joint_names):
    return np.array([physics.named.data.qpos[name] for name in joint_names], dtype=np.float64)


def _named_qvel(physics, joint_names):
    return np.array([physics.named.data.qvel[name] for name in joint_names], dtype=np.float64)


def _set_named_qpos(physics, joint_names, values):
    for name, value in zip(joint_names, values):
        physics.named.data.qpos[name] = value


def _hand_from_switch(value, open_pose, close_pose):
    close_amount = 1.0 - float(np.clip(value, 0.0, 1.0))
    return open_pose + close_amount * (close_pose - open_pose)


class DianaBimanualTask(base.Task):
    action_dim = 16

    def before_step(self, action, physics):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action.shape}")

        right_arm = action[:7]
        right_hand_switch = action[7]
        left_arm = action[8:15]
        left_hand_switch = action[15]

        right_hand = _hand_from_switch(right_hand_switch, RIGHT_HAND_OPEN, RIGHT_HAND_CLOSE)
        left_hand = _hand_from_switch(left_hand_switch, LEFT_HAND_OPEN, LEFT_HAND_CLOSE)

        env_action = np.concatenate([right_arm, left_arm, right_hand, left_hand])
        np.copyto(physics.data.ctrl, env_action)

    def initialize_episode(self, physics):
        with physics.reset_context():
            _set_named_qpos(physics, RIGHT_ARM_JOINTS, DIANA_START_ACTION[:7])
            _set_named_qpos(physics, LEFT_ARM_JOINTS, DIANA_START_ACTION[8:15])
            _set_named_qpos(physics, RIGHT_HAND_JOINTS, RIGHT_HAND_OPEN)
            _set_named_qpos(physics, LEFT_HAND_JOINTS, LEFT_HAND_OPEN)
            physics.named.data.qpos["red_peg_joint"][:3] = np.array([0.80, 0.50, 0.8401])
            physics.named.data.qpos["red_peg_joint"][3:] = np.array([0.7071, 0.0, 0.0, 0.7071])
            physics.named.data.qpos["blue_socket_joint"][:3] = np.array([0.50, -0.50, 0.8401])
            physics.named.data.qpos["blue_socket_joint"][3:] = np.array([0.7071, 0.0, 0.0, 0.7071])
            self.before_step(DIANA_START_ACTION, physics)
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        peg = physics.named.data.qpos["red_peg_joint"].copy()
        socket = physics.named.data.qpos["blue_socket_joint"].copy()
        return np.concatenate([peg, socket])

    @staticmethod
    def get_force(physics):
        names = ["right_wrist_force", "right_wrist_torque", "left_wrist_force", "left_wrist_torque"]
        values = []
        for name in names:
            try:
                sensor_id = physics.model.sensor(name).id
                adr = physics.model.sensor_adr[sensor_id]
                dim = physics.model.sensor_dim[sensor_id]
                values.append(physics.data.sensordata[adr : adr + dim].copy())
            except Exception:
                values.append(np.zeros(3))
        return np.concatenate(values)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        right_arm_qpos = _named_qpos(physics, RIGHT_ARM_JOINTS)
        left_arm_qpos = _named_qpos(physics, LEFT_ARM_JOINTS)
        right_hand_qpos = _named_qpos(physics, RIGHT_HAND_JOINTS)
        left_hand_qpos = _named_qpos(physics, LEFT_HAND_JOINTS)
        right_hand_closed = [float(np.linalg.norm(right_hand_qpos - RIGHT_HAND_OPEN) > 0.1)]
        left_hand_closed = [float(np.linalg.norm(left_hand_qpos - LEFT_HAND_OPEN) > 0.1)]

        obs["qpos"] = np.concatenate([right_arm_qpos, right_hand_closed, left_arm_qpos, left_hand_closed])
        obs["qvel"] = np.concatenate([
            _named_qvel(physics, RIGHT_ARM_JOINTS),
            [0.0],
            _named_qvel(physics, LEFT_ARM_JOINTS),
            [0.0],
        ])
        obs["env_state"] = self.get_env_state(physics)
        obs["force"] = self.get_force(physics)
        obs["images"] = {
            "overview": physics.render(height=480, width=640, camera_id="overview"),
        }
        obs["right_wrist_pos"] = physics.named.data.site_xpos["right_wrist_ft_site"].copy()
        obs["left_wrist_pos"] = physics.named.data.site_xpos["left_wrist_ft_site"].copy()
        obs["peg_pos"] = physics.named.data.xpos["peg"].copy()
        return obs


class DianaPickPegTask(DianaBimanualTask):
    max_reward = 2

    def get_reward(self, physics):
        peg_z = physics.named.data.xpos["peg"][2]
        if peg_z > 0.93:
            return 2
        if peg_z > 0.86:
            return 1
        return 0
