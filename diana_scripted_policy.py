import numpy as np

from diana_sim_env import DIANA_START_ACTION


class DianaPickPegPolicy:
    """Joint-space scripted grasp policy with a 16-D ACT-friendly action."""

    def __init__(self):
        self.step_count = 0
        self.trajectory = self._build_trajectory()

    def _build_trajectory(self):
        start = DIANA_START_ACTION.copy()

        # Right arm attempts to reach the peg at x=0.80, y=0.50.
        # Left arm stays open and away from the grasp during this first rollout.
        pre_grasp = start.copy()
        pre_grasp[:7] = np.array([0.75, 0.45, -0.20, 1.35, -0.15, -0.80, 0.20])

        descend = pre_grasp.copy()
        descend[:7] = np.array([0.82, 0.65, -0.25, 1.65, -0.20, -0.95, 0.25])

        close = descend.copy()
        close[7] = 0.0

        lift = close.copy()
        lift[:7] = np.array([0.78, 0.35, -0.15, 1.25, -0.10, -0.75, 0.25])

        hold = lift.copy()
        hold[:7] = np.array([0.65, 0.25, -0.05, 1.20, -0.05, -0.70, 0.20])

        return [
            {"t": 0, "action": start},
            {"t": 80, "action": pre_grasp},
            {"t": 150, "action": descend},
            {"t": 210, "action": close},
            {"t": 300, "action": lift},
            {"t": 380, "action": hold},
            {"t": 450, "action": hold},
        ]

    @staticmethod
    def _interpolate(curr_waypoint, next_waypoint, t):
        t0 = curr_waypoint["t"]
        t1 = next_waypoint["t"]
        if t1 == t0:
            return next_waypoint["action"].copy()
        frac = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
        return curr_waypoint["action"] + frac * (next_waypoint["action"] - curr_waypoint["action"])

    def __call__(self, _ts):
        while len(self.trajectory) > 1 and self.step_count >= self.trajectory[1]["t"]:
            self.trajectory.pop(0)
        action = self._interpolate(self.trajectory[0], self.trajectory[1], self.step_count)
        self.step_count += 1
        return action.copy()
