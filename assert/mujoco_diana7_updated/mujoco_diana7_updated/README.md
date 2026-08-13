# 双臂 Diana7 + QB SoftHand 触觉仿真

MuJoCo 场景：两台 Diana7 机械臂，各配一只 QB SoftHand，指腹带 160 个触觉 taxel，腕部带六维力/力矩传感器。

## 文件

| 文件 | 说明 |
|---|---|
| `scene_tactile.xml` | 顶层场景（光照、地面、台架），引用机器人模型 |
| `dual_diana7_with_tactile.xml` | 机器人模型：双臂 + 双手 + 160 个 taxel site + 4 个腕部传感器 |
| `wrist_sensor.py` | 加载场景、推进仿真、读取腕部力/力矩 |
| `tactile_render.py` | 把 taxel 接触力渲染成 (10, 4, 4) 触觉图 |
| `inspect_tactile.py` | 指节半透明化，目视检查埋在皮下的 160 个触觉点 |
| `assets/` | STL 网格 |

## 依赖

```bash
pip install mujoco numpy
```

## 用法

```python
from wrist_sensor import WristSensorRig
from tactile_render import TactileRenderer

rig = WristSensorRig()
rig.apply_home_pose()
renderer = TactileRenderer(rig.model)

for _ in range(500):
    rig.step()

img = renderer.update(rig.data)   # (10, 4, 4) 触觉图，单位 N
wrench = rig.read_sensors()       # 12 维：左右手各 3 力 + 3 力矩
print(rig.channel_names)
```

## 触觉图布局

`(10, 4, 4)`：10 个手指（左右手各 5 指），每指 4×4 taxel 阵列，共 160 个。取值为该 taxel 上的法向接触力（N）。

手指顺序：右手 thumb / index / middle / ring / little，然后左手同序。

## 可视化

看触觉点（指节半透明，右手青色 / 左手橙色）：

```bash
python inspect_tactile.py
```

只看场景（site 默认不显示，触觉点被实心指节挡住）：

```bash
python -m mujoco.viewer --mjcf=scene_tactile.xml
```

viewer 里按 `S` 切换 site 显示。
