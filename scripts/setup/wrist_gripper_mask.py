"""Pixels of the wrist camera image that show the robot's own gripper (assets/wrist_gripper_mask.npz).

The fingers and the hand move with the wrist camera, so a motion-compensated late frame (sensor fault
lag_wrist_warp) keeps those pixels as they are. The mask is the robot's instance segmentation over 40 control steps of
random hand motion from a calibration init state (pixels that show the robot in at least half of the frames).

Usage: python scripts/setup/wrist_gripper_mask.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlamech.envs.libero_env import LiberoTaskEnv  # noqa: E402


def main():
    env = LiberoTaskEnv("libero_goal", 1, segmentation=True)
    frame = env.reset(0)
    robot = env.robot_segmentation_ids()
    rng = np.random.default_rng(0)
    a, seen = np.zeros(7), []
    for t in range(40):
        if t % 8 == 0:
            a[:3] = rng.uniform(-0.6, 0.6, 3)
            a[2] = -abs(a[2])
            a[3:6] = rng.uniform(-0.1, 0.1, 3)
        seen.append(np.isin(frame.wrist_seg, list(robot)))
        frame, _ = env.step(a)
    mask = np.mean(seen, axis=0) >= 0.5
    np.savez_compressed(ROOT / "assets" / "wrist_gripper_mask.npz", mask=mask)
    print(f"{mask.mean():.3f} of the wrist image is the gripper")


if __name__ == "__main__":
    main()
