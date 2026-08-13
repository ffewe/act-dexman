import argparse

import matplotlib.pyplot as plt
import numpy as np

from diana_scripted_policy import DianaPickPegPolicy
from diana_sim_env import make_diana_sim_env


def main(args):
    env = make_diana_sim_env()
    ts = env.reset()
    policy = DianaPickPegPolicy()

    plt_img = None
    if args.onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation["images"]["overview"])
        plt.ion()

    rewards = []
    for step in range(args.episode_len):
        action = policy(ts)
        ts = env.step(action)
        rewards.append(ts.reward)

        if step % args.log_every == 0:
            peg_pos = ts.observation["peg_pos"]
            right_wrist = ts.observation["right_wrist_pos"]
            print(
                f"step={step:04d} reward={ts.reward} "
                f"peg_pos={np.round(peg_pos, 4)} "
                f"right_wrist={np.round(right_wrist, 4)}"
            )

        if args.onscreen_render:
            plt_img.set_data(ts.observation["images"]["overview"])
            plt.pause(0.001)

    if args.onscreen_render:
        plt.ioff()
        plt.show()

    print(f"max_reward={np.max(rewards) if rewards else 0}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_len", type=int, default=450)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--onscreen_render", action="store_true")
    main(parser.parse_args())
