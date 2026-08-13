"""触觉点目视检查：指节半透明 + 强制显示 site，直接看到埋在皮下的 160 个触觉点"""
import sys
import mujoco
import mujoco.viewer

# 手部指节 body 名前缀（右手 link1..11 小写，左手 Link1..11 大写）
FINGER_PREFIX = "link"
# 指节透明度，越小越透
FINGER_ALPHA = 0.28


def make_fingers_transparent(model, alpha=FINGER_ALPHA):
    """把手部指节 geom 调成半透明，让内部的触觉 site 可见"""
    count = 0
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if name.lower().startswith(FINGER_PREFIX):
            model.geom_rgba[gid] = [0.72, 0.75, 0.78, alpha]
            count += 1
    return count


def count_taxels(model):
    """统计触觉 site 数量"""
    return sum(
        1
        for i in range(model.nsite)
        if "taxel" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "")
    )


def main():
    """加载场景，开启 site 显示并把指节透明化后启动交互查看器"""
    path = sys.argv[1] if len(sys.argv) > 1 else "scene_tactile.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    n_geom = make_fingers_transparent(model)
    print(f"{path}: {count_taxels(model)} 个触觉 site，{n_geom} 个指节 geom 已半透明")
    print("右手青色 / 左手橙色；viewer 里按 S 可切换 site 显示")

    # 阻塞式窗口，自带渲染循环，比 launch_passive 更不容易立刻退出
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
