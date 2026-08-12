import numpy as np
import torch
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside


def quaternion_to_rotation_matrix(q):
    """
    四元数转旋转矩阵
    q = [x, y, z, w]
    """
    x, y, z, w = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])

class RightForceToPegTransformer:
    """
    右侧力传感器数据转换到peg物体坐标系（只处理力，不处理力矩）
    """
    
    def __init__(self):
        # 传感器到腕部的变换（假设传感器与腕部坐标系对齐）
        self.sensor_to_wrist_rotation = np.eye(3)  # 旋转矩阵
        self.sensor_to_wrist_translation = np.zeros(3)  # 平移向量
    
    def compute_peg_force(self, sensor_force, wrist_pose, peg_pose):
        """
        计算peg物体受到的力
        
        Args:
            sensor_force: [fx, fy, fz] 右侧传感器原始力数据
            wrist_pose: [x, y, z, qx, qy, qz, qw] 右侧腕部位姿
            peg_pose: [x, y, z, qx, qy, qz, qw] peg物体位姿
            
        Returns:
            peg_force: peg物体坐标系中的力 [fx, fy, fz]
        """
        # 1. 传感器力转到腕部坐标系
        # 1. 传感器力转到腕部坐标系
        wrist_force = sensor_force.copy()
        #wrist_force = self.sensor_to_wrist_rotation @ sensor_force
        #wrist_force = self._sensor_to_wrist_force(sensor_force)
        
        # 2. 腕部坐标系转到世界坐标系
        wrist_rotation = quaternion_to_rotation_matrix(wrist_pose[3:])
        world_force = wrist_rotation @ wrist_force
        #world_force = self._wrist_to_world_force(wrist_force, wrist_pose)
        
        # 3. 世界坐标系转到peg物体坐标系
        peg_rotation = quaternion_to_rotation_matrix(peg_pose[3:])
        world_to_peg_rotation = peg_rotation.T
        peg_force = world_to_peg_rotation @ world_force
        #peg_force = self._world_to_peg_force(world_force, peg_pose)
        
        return peg_force
    
    def _sensor_to_wrist_force(self, sensor_force):
        """传感器力转到腕部坐标系"""
        # F_wrist = R_sensor_to_wrist * F_sensor
        return self.sensor_to_wrist_rotation @ sensor_force
    
    def _wrist_to_world_force(self, wrist_force, wrist_pose):
        """腕部力转到世界坐标系"""
        # 获取腕部的旋转矩阵
        wrist_rotation = quaternion_to_rotation_matrix(wrist_pose[3:])
        
        # F_world = R_wrist_to_world * F_wrist
        return wrist_rotation @ wrist_force
    
    def _world_to_peg_force(self, world_force, peg_pose):
        """世界力转到peg物体坐标系"""
        # 获取peg的旋转矩阵
        peg_rotation = quaternion_to_rotation_matrix(peg_pose[3:])
        
        # peg坐标系到世界坐标系的旋转矩阵是peg_rotation
        # 世界坐标系到peg坐标系的旋转矩阵是转置
        world_to_peg_rotation = peg_rotation.T
        
        # F_peg = R_world_to_peg * F_world
        return world_to_peg_rotation @ world_force

def make_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.force_transformer = RightForceToPegTransformer()
        self.peg_force_history = []  # 存储peg受力历史

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)
        self.peg_force_history = []

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    @staticmethod
    def get_force(physics):

        #sensor_id=physics.model.name2id('force_sensor_left','sensor')
        name1 = "force_right"
        name2 = "torque_right"
        name3 = "force_left"
        name4 = "torque_left"
        sensor_id1 = physics.model.sensor(name1).id
        adr1 = sensor_id1 * 3                    
        force1 = physics.data.sensordata[adr1:adr1+3]

        sensor_id2=physics.model.sensor(name2).id
        adr2 = sensor_id2 *3
        force2 = physics.data.sensordata[adr2:adr2+3]

        sensor_id4 = physics.model.sensor(name4).id
        adr4 = sensor_id4 * 3                    
        force4 = physics.data.sensordata[adr4:adr4+3]

        sensor_id3 = physics.model.sensor(name3).id
        adr3 = sensor_id3 * 3                    
        force3 = physics.data.sensordata[adr3:adr3+3]
        #force = np.array([force1, force2, force3, force4])
        force = np.concatenate([force1, force2, force3, force4])
        #force = np.concatenate([force1,force4,force2,force3],axis=0)
        #force = torch.from_numpy(np.concatenate([force1, force2], axis=0)).float()                                        
        return force.copy()              


    # def get_observation(self, physics):
    #     obs = collections.OrderedDict()
    #     obs['qpos'] = self.get_qpos(physics)
    #     obs['qvel'] = self.get_qvel(physics)
    #     obs['force']=self.get_force(physics)
    #     obs['env_state'] = self.get_env_state(physics)
    #     obs['images'] = dict()
    #     obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
    #     obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
    #     obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        

    #     return obs

    # def get_reward(self, physics):
    #     # return whether left gripper is holding the box
    #     raise NotImplementedError


    def get_right_sensor_force(self, physics):
        """获取右侧力传感器数据（只取力，不要力矩）"""
        force_data = self.get_force(physics)
        # 前3个元素是右侧传感器的力 (fx, fy, fz)
        return force_data[:3]
    
    # === 新增方法：获取右侧腕部位姿 ===
    def get_right_wrist_pose(self, physics):
        """获取右侧腕部在世界坐标系中的位姿"""
        body_name = 'vx300s_right/wrist_link'
        body_id = physics.model.name2id(body_name, 'body')
        
        # 位置
        position = physics.named.data.xpos[body_id].copy()
        
        # 四元数 (w, x, y, z) 转 (x, y, z, w)
        quat_wxyz = physics.named.data.xquat[body_id].copy()
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        return np.concatenate([position, quat_xyzw])
    
    # === 新增方法：获取peg物体位姿 ===
    def get_peg_pose(self, physics):
        """获取peg物体在世界坐标系中的位姿"""
        try:
            body_id = physics.model.name2id("red_peg", 'body')
        except:
            # 如果没有body，尝试geom
            geom_id = physics.model.name2id("red_peg", 'geom')
            body_id = physics.model.geom_bodyid[geom_id]
        
        # 位置
        position = physics.named.data.xpos[body_id].copy()
        
        # 四元数
        quat_wxyz = physics.named.data.xquat[body_id].copy()
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        
        return np.concatenate([position, quat_xyzw])
    
    # === 新增方法：计算peg物体受力 ===
    def compute_peg_force_from_right_sensor(self, physics):
        """从右侧传感器计算peg物体受到的力"""
        # 获取右侧传感器力数据
        right_sensor_force = self.get_right_sensor_force(physics)
        
        # 获取右侧腕部位姿
        right_wrist_pose = self.get_right_wrist_pose(physics)
        
        # 获取peg物体位姿
        peg_pose = self.get_peg_pose(physics)
        
        # 计算peg受力
        peg_force = self.force_transformer.compute_peg_force(
            right_sensor_force, right_wrist_pose, peg_pose
        )
        
        # 存储历史
        self.peg_force_history.append({
            'time': physics.time(),
            'peg_force': peg_force.copy(),
            'sensor_force': right_sensor_force.copy(),
            'peg_pose': peg_pose.copy(),
            'wrist_pose': right_wrist_pose.copy()
        })
        
        # 保持数据长度
        if len(self.peg_force_history) > 1000:
            self.peg_force_history.pop(0)
        
        return peg_force
    
    # === 新增方法：获取力分析 ===
    def get_force_analysis(self):
        """获取peg受力分析"""
        if not self.peg_force_history:
            return {}
        
        recent = self.peg_force_history[-1]
        peg_force = recent['peg_force']
        
        analysis = {
            'force_x': peg_force[0],  # peg坐标系x方向力（插入方向）
            'force_y': peg_force[1],  # peg坐标系y方向力（侧向）
            'force_z': peg_force[2],  # peg坐标系z方向力（垂直）
            'force_magnitude': np.linalg.norm(peg_force),
            'insertion_force': abs(peg_force[0]),  # 插入力大小
            'lateral_force': np.sqrt(peg_force[1]**2 + peg_force[2]**2),  # 侧向力大小
            'time': recent['time']
        }
        
        # 判断接触状态
        insertion_force = abs(peg_force[0])
        if insertion_force > 1.0:
            analysis['contact_status'] = 'strong_contact'
        elif insertion_force > 0.2:
            analysis['contact_status'] = 'contact'
        else:
            analysis['contact_status'] = 'no_contact'
        
        return analysis

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['force'] = self.get_force(physics)
        obs['env_state'] = self.get_env_state(physics)
        
        # === 新增：添加peg受力信息 ===
        if hasattr(self, 'force_transformer'):
            try:
                peg_force = self.compute_peg_force_from_right_sensor(physics)
                obs['peg_force'] = peg_force
                
                # 添加力分析
                force_analysis = self.get_force_analysis()
                obs['force_analysis'] = force_analysis
            except Exception as e:
                # 如果出错，返回零值
                obs['peg_force'] = np.zeros(3)
                obs['force_analysis'] = {}
        else:
            obs['peg_force'] = np.zeros(3)
            obs['force_analysis'] = {}
        
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        # obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
        # obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        # 子类实现
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

