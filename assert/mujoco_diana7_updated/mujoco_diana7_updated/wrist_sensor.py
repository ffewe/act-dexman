"""手腕 6 轴力/力矩传感器读取：加载场景、下发保持姿态、输出 12 通道读数。"""

from pathlib import Path

import mujoco
import numpy as np


SCENE_PATH = Path(__file__).parent / "scene_tactile.xml"
AXES = ("x", "y", "z")
SIDES = ("right", "left")

# sensordata 中要读取的传感器名
SENSORS = {
    "right": {"force": "right_wrist_force", "torque": "right_wrist_torque"},
    "left": {"force": "left_wrist_force", "torque": "left_wrist_torque"},
}

# 双臂位置执行器，顺序对应 HOME_QPOS 的 7 个关节
ARM_ACTUATORS = {
    side: [f"{side}_arm_joint_{i}_ctrl" for i in range(1, 8)] for side in SIDES
}

# 保持姿态（rad），joint_4 限位 0~3.05，其余 ±3.12，joint_2 ±1.57
HOME_QPOS = {
    "right": (0.0, 0.3, 0.0, 1.2, 0.0, 0.6, 0.0),
    "left": (0.0, 0.3, 0.0, 1.2, 0.0, 0.6, 0.0),
}


class WristSensorRig:
    """封装场景加载、双臂保持姿态下发和 12 通道传感器读数。"""

    def __init__(self, scene_path=SCENE_PATH):
        """加载场景，缓存传感器与执行器索引。"""
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.timestep = self.model.opt.timestep
        self._sensor_slices = self._resolve_sensor_slices()
        self._actuator_ids = self._resolve_actuator_ids()
        self.channel_names = self._build_channel_names()

    def _resolve_sensor_slices(self):
        """把每个传感器名解析成 sensordata 上的读取区间。"""
        slices = {}
        for side, entries in SENSORS.items():
            for kind, name in entries.items():
                sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
                if sensor_id < 0:
                    raise ValueError(f"场景中找不到传感器 {name}")
                start = self.model.sensor_adr[sensor_id]
                slices[(side, kind)] = slice(start, start + self.model.sensor_dim[sensor_id])
        return slices

    def _resolve_actuator_ids(self):
        """把双臂执行器名解析成 ctrl 索引列表。"""
        actuator_ids = {}
        for side, names in ARM_ACTUATORS.items():
            ids = []
            for name in names:
                actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if actuator_id < 0:
                    raise ValueError(f"场景中找不到执行器 {name}")
                ids.append(actuator_id)
            actuator_ids[side] = ids
        return actuator_ids

    def _build_channel_names(self):
        """生成 12 个通道的列名，顺序与 read_sensors 一致。"""
        names = []
        for side in SIDES:
            for kind in ("force", "torque"):
                names.extend(f"{side}_{kind}_{axis}" for axis in AXES)
        return names

    def apply_home_pose(self):
        """把双臂 ctrl 和 qpos 同时置到保持姿态，避免起步冲击。"""
        for side, ids in self._actuator_ids.items():
            for actuator_id, target in zip(ids, HOME_QPOS[side]):
                self.data.ctrl[actuator_id] = target
                joint_id = self.model.actuator_trnid[actuator_id, 0]
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = target
        mujoco.mj_forward(self.model, self.data)

    def set_arm_target(self, side, joint_index, value):
        """设置单个手臂关节的位置目标。"""
        self.data.ctrl[self._actuator_ids[side][joint_index]] = value

    def read_sensors(self):
        """读取当前 12 通道力/力矩，顺序与 channel_names 一致。"""
        values = []
        for side in SIDES:
            for kind in ("force", "torque"):
                values.extend(self.data.sensordata[self._sensor_slices[(side, kind)]])
        return np.asarray(values, dtype=float)

    def step(self):
        """推进一个仿真步。"""
        mujoco.mj_step(self.model, self.data)
