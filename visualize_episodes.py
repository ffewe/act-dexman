import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from constants import DT

import IPython
e = IPython.embed

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
STATE_NAMES = JOINT_NAMES + ["gripper"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        if 'peg_force' in root['/observations']:
            peg_force_data = root['/observations/peg_force'][()]
        else:
            peg_force_data = None

        if 'force' in root['/observations']:
            force_data = root['/observations/force'][()]
        else:
            force_data = None

        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

            
    return qpos, qvel, action, image_dict,force_data,peg_force_data

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_new{episode_idx}'

    qpos, qvel, action, image_dict ,force_data,peg_force_data = load_hdf5(dataset_dir, dataset_name)

    # if peg_force_data is not None:

    #     save_videos_with_peg_force(image_dict, DT, peg_force_data, 
    #                               video_path=os.path.join(dataset_dir, dataset_name + '_video_with_peg_force.mp4'))

    if force_data is not None:
        save_videos_with_ft(image_dict, DT, force_data, 
                          video_path=os.path.join(dataset_dir, dataset_name + '_video_with_ft.mp4'))
    else:
        save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
    
    visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.png'))
    
# 在 visualize_episodes.py 中添加以下函数

def create_peg_force_overlay(peg_force_history, timestamps, width, height, current_time):
    """创建peg物体受力覆盖层"""
    overlay = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
    
    if len(peg_force_history) < 2:
        # 如果没有足够数据，显示提示
        text = "Waiting for peg force data..."
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        cv2.putText(overlay, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return overlay
    
    # 设置绘图区域
    margin = 20
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    
    # 提取peg力数据（从开始到当前时刻）
    peg_forces = np.array(peg_force_history)  # shape: [timesteps, 3]
    
    # 使用完整的时间范围（从0到当前时刻）
    total_time = timestamps[-1] if timestamps else current_time
    time_range = max(1.0, total_time)  # 至少显示1秒
    
    # 计算力的范围（基于所有数据）
    force_max = max(5.0, np.max(np.abs(peg_forces)) * 1.5)
    
    # 在同一图表中显示所有力分量
    plot_top = margin
    plot_bottom = margin + plot_height
    
    # 颜色定义 - peg力分量
    colors = [
        (255, 0, 0),    # 红色 - Fx (插入方向)
        (0, 255, 0),    # 绿色 - Fy (侧向)
        (0, 0, 255),    # 蓝色 - Fz (垂直)
       
    ]
    
    labels = [
        'Fx (Insertion)',
        'Fy (Lateral)', 
        'Fz (Vertical)',
        
    ]
    
    # 绘制所有力曲线
    for i in range(3):  # 3条分量曲线 + 1条合力曲线
        # 归一化坐标
        x_coords = []
        y_coords = []
        
        for j, t in enumerate(timestamps):
            x = (t / time_range * plot_width) + margin
            force_value = peg_forces[j, i]
            
            # 将力值映射到图表高度
            normalized_force = force_value / force_max  # 归一化到 [-1, 1]
            y = plot_bottom - ((normalized_force + 1) / 2) * plot_height
            x_coords.append(x)
            y_coords.append(y)
        
        # 绘制曲线
        if len(x_coords) > 1:
            points = np.array([x_coords, y_coords]).T.astype(np.int32)
            for j in range(len(points)-1):
                cv2.line(overlay, tuple(points[j]), tuple(points[j+1]), colors[i], 2)
        
        # 添加标签
        cv2.putText(overlay, labels[i], 
                   (int(width - 150), int(margin + 20 + i*20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[i], 1)
    
    # 添加标题
    cv2.putText(overlay, 'PEG Force in Peg Frame (N)', 
               (margin, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 绘制坐标轴
    cv2.line(overlay, (margin, plot_top), (margin, plot_bottom), (0, 0, 0), 2)  # Y轴
    cv2.line(overlay, (margin, (plot_top + plot_bottom)//2), 
             (margin + plot_width, (plot_top + plot_bottom)//2), (0, 0, 0), 1)  # X轴（零线）
    
    # 添加当前力值显示
    if len(peg_forces) > 0:
        current_force = peg_forces[-1]
        force_magnitude = np.linalg.norm(current_force)
        
        # 显示当前力值
        current_text = f'Current: ({current_force[0]:.2f}, {current_force[1]:.2f}, {current_force[2]:.2f}) N'
        magnitude_text = f'Magnitude: {force_magnitude:.2f} N'
        
        cv2.putText(overlay, current_text, 
                   (margin, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(overlay, magnitude_text, 
                   (margin, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        
    # 添加量程信息
    cv2.putText(overlay, f'+{force_max:.1f}N', 
               (5, plot_top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(overlay, f'-{force_max:.1f}N', 
               (5, plot_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    return overlay

def save_videos_with_peg_force(video, dt, peg_force_history, video_path=None):
    """保存带有peg受力图的视频"""
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        
        # 添加peg受力图区域
        peg_force_width = 500
        total_width = w * len(cam_names) + peg_force_width
            
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_width, h))
        
        # 预先计算所有时间戳
        total_timestamps = [i * dt for i in range(len(peg_force_history))]
        
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                images.append(image)
            
            # 拼接所有相机图像
            combined_image = np.concatenate(images, axis=1)
            
            # 如果有力数据，添加peg受力图
            if ts < len(peg_force_history):
                current_time = ts * dt
                
                # 只传递从开始到当前时刻的数据
                current_peg_force_data = peg_force_history[:ts+1]
                current_timestamps = total_timestamps[:ts+1]
                
                peg_force_overlay = create_peg_force_overlay(
                    current_peg_force_data, current_timestamps, 
                    peg_force_width, h, current_time
                )
                
                # 将peg受力图拼接到右侧
                final_image = np.concatenate([combined_image, peg_force_overlay], axis=1)
            else:
                final_image = combined_image
            
            out.write(final_image)
            
        out.release()
        print(f'Saved video with peg force overlay to: {video_path}')
        
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        
        peg_force_width = 500
        total_width = w + peg_force_width
            
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_width, h))
        
        # 预先计算所有时间戳
        total_timestamps = [i * dt for i in range(len(peg_force_history))]
        
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            
            # 如果有力数据，添加peg受力图
            if t < len(peg_force_history):
                current_time = t * dt
                
                current_peg_force_data = peg_force_history[:t+1]
                current_timestamps = total_timestamps[:t+1]
                
                peg_force_overlay = create_peg_force_overlay(
                    current_peg_force_data, current_timestamps, 
                    peg_force_width, h, current_time
                )
                final_image = np.concatenate([image, peg_force_overlay], axis=1)
            else:
                final_image = image
                
            out.write(final_image)
            
        out.release()
        print(f'Saved video with peg force overlay to: {video_path}')

def create_ft_overlay(
        ft_data_history,
        timestamps,
        width,
        height,
        current_time):

    """
    Force overlay visualization.

    Layout:

    +----------------------+----------------------+
    | Left Arm Force       | Right Arm Force      |
    |                      |                      |
    | Fx/Fy/Fz curves      | Fx/Fy/Fz curves      |
    |                      |                      |
    | (Fx,Fy,Fz) N         | (Fx,Fy,Fz) N         |
    | |F| = xx N           | |F| = xx N           |
    +----------------------+----------------------+

    Force:
        Right arm: force[0:3]
        Left arm : force[6:9]
    """

    import cv2
    import numpy as np


    # ==========================
    # background
    # ==========================

    overlay = np.ones(
        (height, width, 3),
        dtype=np.uint8
    ) * 255



    if len(ft_data_history) < 2:

        cv2.putText(
            overlay,
            "Waiting for force data...",
            (30, height//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,0),
            2
        )

        return overlay



    force_history = np.asarray(
        ft_data_history
    )


    # ==========================
    # split force
    # ==========================

    left_force = force_history[:,6:9]

    right_force = force_history[:,0:3]



    # ==========================
    # layout
    # ==========================

    margin = 45

    gap = 40


    block_width = (
        width
        -
        2*margin
        -
        gap
    ) // 2


    curve_height = int(
        height*0.55
    )


    curve_y = 45


    left_x = margin

    right_x = (
        margin
        +
        block_width
        +
        gap
    )



    # ==========================
    # colors BGR
    # ==========================

    colors = [

        (255,0,0),      # Fx blue

        (0,140,255),    # Fy orange

        (0,180,0)       # Fz green

    ]


    labels = [
        "Fx",
        "Fy",
        "Fz"
    ]



    # ==========================
    # draw force curve
    # ==========================

    def draw_force_curve(
            force,
            x0,
            title):


        max_force = max(
            10,
            np.max(
                np.abs(force)
            )*1.2
        )


        y0 = curve_y



        # title

        cv2.putText(
            overlay,
            title,
            (
                x0,
                y0-15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )



        # frame

        cv2.rectangle(
            overlay,
            (
                x0,
                y0
            ),
            (
                x0+block_width,
                y0+curve_height
            ),
            (0,0,0),
            1
        )



        zero_y = int(
            y0+curve_height/2
        )


        # zero line

        cv2.line(
            overlay,
            (
                x0,
                zero_y
            ),
            (
                x0+block_width,
                zero_y
            ),
            (180,180,180),
            1
        )



        # ======================
        # y axis scale
        # ======================

        ticks = [
            max_force,
            max_force/2,
            0,
            -max_force/2,
            -max_force
        ]


        for value in ticks:

            y = int(
                zero_y
                -
                value/max_force
                *
                curve_height/2
            )


            cv2.line(
                overlay,
                (
                    x0-5,
                    y
                ),
                (
                    x0,
                    y
                ),
                (0,0,0),
                1
            )


            cv2.putText(
                overlay,
                f"{value:.0f}",
                (
                    x0-38,
                    y+5
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0,0,0),
                1
            )



        # ======================
        # curves
        # ======================

        for axis in range(3):

            points=[]


            for i in range(len(force)):

                x = int(
                    x0
                    +
                    i/(len(force)-1)
                    *
                    block_width
                )


                y = int(
                    zero_y
                    -
                    force[i,axis]
                    /
                    max_force
                    *
                    curve_height/2
                )


                points.append(
                    (x,y)
                )


            for i in range(
                    len(points)-1):

                cv2.line(
                    overlay,
                    points[i],
                    points[i+1],
                    colors[axis],
                    2
                )



        # ======================
        # legend
        # ======================

        for i in range(3):

            cv2.putText(
                overlay,
                labels[i],
                (
                    x0+10,
                    y0+25+i*22
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colors[i],
                1
            )



        # ======================
        # x axis scale
        # ======================

        if len(timestamps)>1:

            total_time = timestamps[-1]


            for t in np.linspace(
                    0,
                    total_time,
                    5):

                x = int(
                    x0
                    +
                    t/total_time
                    *
                    block_width
                )


                cv2.line(
                    overlay,
                    (
                        x,
                        y0+curve_height
                    ),
                    (
                        x,
                        y0+curve_height+5
                    ),
                    (0,0,0),
                    1
                )


                cv2.putText(
                    overlay,
                    f"{t:.1f}",
                    (
                        x-10,
                        y0+curve_height+20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0,0,0),
                    1
                )



        # ======================
        # current force vector
        # ======================

        current = force[-1]


        magnitude = np.linalg.norm(
            current
        )


        text_y = (
            y0
            +
            curve_height
            +
            55
        )


        cv2.putText(
            overlay,
            f"({current[0]:.2f}, "
            f"{current[1]:.2f}, "
            f"{current[2]:.2f}) N",
            (
                x0,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0,0,0),
            1
        )


        cv2.putText(
            overlay,
            f"|F| = {magnitude:.2f} N",
            (
                x0,
                text_y+25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0,0,0),
            1
        )



    # ==========================
    # draw left/right
    # ==========================

    draw_force_curve(
        left_force,
        left_x,
        "Left Arm Force"
    )


    draw_force_curve(
        right_force,
        right_x,
        "Right Arm Force"
    )



    # ==========================
    # time
    # ==========================

    cv2.putText(
        overlay,
        f"Time: {current_time:.2f}s",
        (
            width-120,
            height-15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80,80,80),
        1
    )


    return overlay

def save_videos(video, dt, video_path=None, ft_data_history=None):
    """保存视频，可选择添加力数据覆盖"""
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        
        # 如果有力数据，在右侧添加力图区域
        if ft_data_history is not None:
            ft_width = 500  # 力图宽度
            total_width = w * len(cam_names) + ft_width
        else:
            print("No force data provided") 
            total_width = w * len(cam_names)
            
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_width, h))
        
        # 预先计算所有时间戳
        if ft_data_history is not None:
            total_timestamps = [i * dt for i in range(len(ft_data_history))]
        else:
            total_timestamps = []
        
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            
            # 拼接所有相机图像
            combined_image = np.concatenate(images, axis=1)
            
            # 如果有力数据，添加力图
            if ft_data_history is not None and ts < len(ft_data_history):
                current_time = ts * dt
                
                # 关键修改：只传递从开始到当前时刻的数据
                current_ft_data = ft_data_history[:ts+1]
                current_timestamps = total_timestamps[:ts+1]
                
                ft_overlay = create_ft_overlay(current_ft_data, current_timestamps, 
                                             ft_width, h, current_time)
                
                # 将力图拼接到右侧
                final_image = np.concatenate([combined_image, ft_overlay], axis=1)
            else:
                final_image = combined_image
            
            out.write(final_image)
            
        out.release()
        print(f'Saved video with force overlay to: {video_path}')
        
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        
        # 如果有力数据，调整宽度
        if ft_data_history is not None:
            ft_width = 500
            total_width = w + ft_width
        else:
            total_width = w
            
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (total_width, h))
        
        # 预先计算所有时间戳
        if ft_data_history is not None:
            print("force exist")
            total_timestamps = [i * dt for i in range(len(ft_data_history))]
        else:
            total_timestamps = []
        
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            
            # 如果有力数据，添加力图
            if ft_data_history is not None and t < len(ft_data_history):
                current_time = t * dt
                
                # 关键修改：只传递从开始到当前时刻的数据
                current_ft_data = ft_data_history[:t+1]
                current_timestamps = total_timestamps[:t+1]
                
                ft_overlay = create_ft_overlay(current_ft_data, current_timestamps, 
                                             ft_width, h, current_time)
                final_image = np.concatenate([image, ft_overlay], axis=1)
            else:
                final_image = image
                
            out.write(final_image)
            
        out.release()
        print(f'Saved video with force overlay to: {video_path}')



def save_videos_with_ft(video, dt, ft_data_history, video_path=None):
    """专门用于保存带有力图的视频"""
    return save_videos(video, dt, video_path, ft_data_history)

# def save_videos(video, dt, video_path=None):
#     if isinstance(video, list):
#         cam_names = list(video[0].keys())
#         h, w, _ = video[0][cam_names[0]].shape
#         w = w * len(cam_names)
#         fps = int(1/dt)
#         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         for ts, image_dict in enumerate(video):
#             images = []
#             for cam_name in cam_names:
#                 image = image_dict[cam_name]
#                 image = image[:, :, [2, 1, 0]] # swap B and R channel
#                 images.append(image)
#             images = np.concatenate(images, axis=1)
#             out.write(images)
#         out.release()
#         print(f'Saved video to: {video_path}')
#     elif isinstance(video, dict):
#         cam_names = list(video.keys())
#         all_cam_videos = []
#         for cam_name in cam_names:
#             all_cam_videos.append(video[cam_name])
#         all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

#         n_frames, h, w, _ = all_cam_videos.shape
#         fps = int(1 / dt)
#         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         for t in range(n_frames):
#             image = all_cam_videos[t]
#             image = image[:, :, [2, 1, 0]]  # swap B and R channel
#             out.write(image)
#         out.release()
#         print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))
