"""触觉阵列渲染：把 MuJoCo 接触力分配到 10 个指腹的 4x4 taxel 网格。"""

import mujoco
import numpy as np


# 10 个指腹分组，顺序决定输出张量第一维
PADS = (
    ("right_thumb", "link3"),
    ("right_index", "link5"),
    ("right_middle", "link7"),
    ("right_ring", "link9"),
    ("right_little", "link11"),
    ("left_thumb", "Link3"),
    ("left_index", "Link5"),
    ("left_middle", "Link7"),
    ("left_ring", "Link9"),
    ("left_little", "Link11"),
)

ROWS = 4
COLS = 4
SITE_PATTERN = "{pad}_taxel_{index:02d}"

# 高斯核：sigma 取相邻 taxel 间距的一半（实测列距 3.9mm、行距 2.9mm）
SIGMA_M = 0.0015
CUTOFF_SIGMA = 3.0

# 输出信号处理
FORCE_MAX = 20.0
QUANTIZE_STEP = 0.0
FILTER_TIME_CONSTANT = 0.02


class TactileRenderer:
    """把接触力按高斯核分配到各指腹 taxel，输出 (pad, rows, cols) 强度图。"""

    def __init__(self, model):
        """缓存 taxel site 索引、body 到指腹的映射和核参数。"""
        self._model = model
        self.pad_names = [name for name, _ in PADS]
        self.rows = ROWS
        self.cols = COLS
        self._site_ids = self._resolve_site_ids()
        self._body_to_pad = self._resolve_body_to_pad()
        self._cutoff = SIGMA_M * CUTOFF_SIGMA
        self._filter_alpha = self._resolve_filter_alpha()
        self.image = np.zeros((len(PADS), ROWS, COLS))
        self.unassigned_force = 0.0
        self.contact_force = 0.0

    def _resolve_site_ids(self):
        """把每个指腹的 16 个 taxel site 名解析成 id 数组，行优先排列。"""
        count = ROWS * COLS
        table = np.zeros((len(PADS), count), dtype=int)
        for pad_index, (pad, _) in enumerate(PADS):
            for offset in range(count):
                name = SITE_PATTERN.format(pad=pad, index=offset + 1)
                site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, name)
                if site_id < 0:
                    raise ValueError(f"场景中找不到 taxel site {name}")
                table[pad_index, offset] = site_id
        return table

    def _resolve_body_to_pad(self):
        """建立指腹 body id 到分组下标的映射，用于判定接触归属。"""
        mapping = {}
        for pad_index, (_, body) in enumerate(PADS):
            body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body)
            if body_id < 0:
                raise ValueError(f"场景中找不到指腹刚体 {body}")
            mapping[body_id] = pad_index
        return mapping

    def _resolve_filter_alpha(self):
        """把一阶低通时间常数换算成每步滤波系数，0 表示不滤波。"""
        if FILTER_TIME_CONSTANT <= 0.0:
            return 1.0
        timestep = self._model.opt.timestep
        return timestep / (timestep + FILTER_TIME_CONSTANT)

    def _pad_of_contact(self, contact):
        """返回接触所属的指腹下标，双方都不是指腹时返回 None。"""
        for geom_id in (contact.geom1, contact.geom2):
            pad_index = self._body_to_pad.get(int(self._model.geom_bodyid[geom_id]))
            if pad_index is not None:
                return pad_index
        return None

    def _spread_to_taxels(self, data, pad_index, position, normal_force, raw):
        """把单个接触点的法向力按高斯核摊到该指腹 taxel，返回未分配部分。"""
        offsets = data.site_xpos[self._site_ids[pad_index]] - position
        distances = np.linalg.norm(offsets, axis=1)
        weights = np.where(
            distances < self._cutoff,
            np.exp(-0.5 * (distances / SIGMA_M) ** 2),
            0.0,
        )
        total = weights.sum()
        if total <= 0.0:
            return normal_force
        raw[pad_index] += normal_force * weights / total
        return 0.0

    def _collect_contacts(self, data):
        """遍历当前接触，累加各 taxel 原始受力并统计合力与未分配力。"""
        raw = np.zeros((len(PADS), ROWS * COLS))
        force_buffer = np.zeros(6)
        self.unassigned_force = 0.0
        self.contact_force = 0.0
        for index in range(data.ncon):
            contact = data.contact[index]
            pad_index = self._pad_of_contact(contact)
            if pad_index is None:
                continue
            mujoco.mj_contactForce(self._model, data, index, force_buffer)
            normal_force = float(force_buffer[0])
            if normal_force <= 0.0:
                continue
            self.contact_force += normal_force
            self.unassigned_force += self._spread_to_taxels(
                data, pad_index, contact.pos, normal_force, raw
            )
        return raw

    def _shape_output(self, raw):
        """对原始受力做饱和、量化，并整形成网格图像。"""
        values = np.clip(raw, 0.0, FORCE_MAX)
        if QUANTIZE_STEP > 0.0:
            values = np.floor(values / QUANTIZE_STEP) * QUANTIZE_STEP
        return values.reshape(len(PADS), ROWS, COLS)

    def update(self, data):
        """推进一帧渲染，返回滤波后的 (pad, rows, cols) 触觉强度图。"""
        target = self._shape_output(self._collect_contacts(data))
        self.image += self._filter_alpha * (target - self.image)
        return self.image

    def pad_totals(self):
        """返回各指腹当前强度总和，用于快速查看接触分布。"""
        return self.image.reshape(len(PADS), -1).sum(axis=1)
