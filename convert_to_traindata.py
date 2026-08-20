"""Convert bare-hand teleoperation CSV recordings into training HDF5 files.

Input layout:
  recordings/bare_hand_teleop_<id>.csv
  recordings/bare_hand_teleop_<id>_frames/frame_000001.jpg

The source frame index is assumed to match the CSV row index. Side-by-side
stereo frames are split into overview_left and overview_right datasets.
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FORCE_COLUMNS = (
    "force_fx_n", "force_fy_n", "force_fz_n", "torque_mx_nm", "torque_my_nm", "torque_mz_nm",
    "left_force_fx_n", "left_force_fy_n", "left_force_fz_n", "left_torque_mx_nm", "left_torque_my_nm", "left_torque_mz_nm",
)
QPOS_LAYOUT = (
    [f"right_arm_joint_{i}_rad" for i in range(1, 8)] + ["right_hand_open"]
    + [f"left_arm_joint_{i}_rad" for i in range(1, 8)] + ["left_hand_open"]
)


def read_rows(csv_path):
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{csv_path}: no rows")
    return rows


def columns(rows, names):
    missing = [name for name in names if name not in rows[0]]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    values = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinity")
    return values


def hand(value):
    value = value.strip().lower()
    if value in ("yes", "true", "open"):
        return 1.0
    if value in ("no", "false", "closed", "close"):
        return 0.0
    return float(value)


def qpos(rows):
    result = []
    for row in rows:
        result.append(
            [*[float(row[f"right_arm_joint_{i}_rad"]) for i in range(1, 8)], hand(row["right_hand_open"]),
             *[float(row[f"left_arm_joint_{i}_rad"]) for i in range(1, 8)], hand(row["left_hand_open"])]
        )
    return np.asarray(result, dtype=np.float64)


def tactile(rows):
    sides = []
    for side in ("right", "left"):
        fingers = []
        for finger in FINGERS:
            fingers.append(columns(rows, [f"{side}_tactile_{finger}_taxel_{i:02d}_raw" for i in range(1, 17)]))
        sides.append(np.stack(fingers, axis=1))
    return np.stack(sides, axis=1)


def source_frames(csv_path):
    directory = csv_path.with_name(f"{csv_path.stem}_frames")
    paths = sorted(directory.glob("frame_*"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not paths:
        raise ValueError(f"No frames in {directory}")
    return paths


def interpolate(time, values, new_time):
    return np.column_stack([np.interp(new_time, time, values[:, i]) for i in range(values.shape[1])])


def nearest_indices(time, new_time):
    right = np.clip(np.searchsorted(time, new_time), 0, len(time) - 1)
    left = np.clip(right - 1, 0, len(time) - 1)
    return np.where(np.abs(new_time - time[left]) <= np.abs(time[right] - new_time), left, right)


def images(paths, indices):
    with Image.open(paths[0]) as image:
        full_width, height = image.size
    if full_width % 2:
        raise ValueError(f"Stereo image width must be even, got {full_width}")
    width = full_width // 2
    left = np.empty((len(indices), height, width, 3), dtype=np.uint8)
    right = np.empty_like(left)
    for destination, source in enumerate(indices):
        with Image.open(paths[source]) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if array.shape != (height, full_width, 3):
            raise ValueError(f"Unexpected image shape in {paths[source]}")
        left[destination], right[destination] = array[:, :width], array[:, width:]
    return left, right


def convert(csv_path, output_dir, hz):
    rows = read_rows(csv_path)
    source_time = columns(rows, ["elapsed_s"]).reshape(-1)
    if len(rows) < 2 or np.any(np.diff(source_time) <= 0):
        raise ValueError("elapsed_s must be strictly increasing")
    paths = source_frames(csv_path)
    if len(paths) != len(rows):
        raise ValueError(f"CSV rows ({len(rows)}) and frames ({len(paths)}) do not match")

    raw_qpos = qpos(rows)
    if not np.isfinite(raw_qpos).all():
        raise ValueError("qpos contains NaN or infinity")
    arm_qvel = columns(rows, [*[f"right_arm_joint_{i}_angular_vel_rad_s" for i in range(1, 8)], *[f"left_arm_joint_{i}_angular_vel_rad_s" for i in range(1, 8)]])
    raw_qvel = np.column_stack([arm_qvel[:, :7], np.zeros(len(rows)), arm_qvel[:, 7:], np.zeros(len(rows))])
    raw_force, raw_tactile = columns(rows, FORCE_COLUMNS), tactile(rows)

    dt = 1.0 / hz
    time = np.arange(source_time[0], source_time[-1] + dt / 2, dt)
    index = nearest_indices(source_time, time)
    out_qpos = interpolate(source_time, raw_qpos, time).astype(np.float32)
    out_qvel = interpolate(source_time, raw_qvel, time).astype(np.float32)
    out_force = interpolate(source_time, raw_force, time).astype(np.float32)
    left, right = images(paths, index)
    action = np.vstack([out_qpos[1:], out_qpos[-1:]])

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{csv_path.stem}.hdf5"
    with h5py.File(output, "w") as root:
        root.attrs["sim"] = False
        root.attrs["control_hz"] = hz
        root.attrs["qpos_layout"] = json.dumps(QPOS_LAYOUT)
        root.attrs["action_semantics"] = "action[t] = qpos[t+1]; final action repeats final qpos"
        root.attrs["image_alignment"] = "frame N corresponds to CSV row N; resampled frames selected by nearest elapsed_s"
        root.attrs["stereo_layout"] = "source image split at horizontal midpoint"
        root.create_dataset("action", data=action, compression="gzip", compression_opts=1)
        root.create_dataset("timestamps/control_s", data=time - time[0])
        root.create_dataset("timestamps/source_row_index", data=index.astype(np.int64))
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=out_qpos, compression="gzip", compression_opts=1)
        obs.create_dataset("qvel", data=out_qvel, compression="gzip", compression_opts=1)
        obs.create_dataset("force", data=out_force, compression="gzip", compression_opts=1)
        dset = obs.create_dataset("tactile_taxels", data=raw_tactile[index].astype(np.float32), compression="gzip", compression_opts=1)
        dset.attrs["layout"] = "[time, side(right,left), finger, taxel]"
        image_group = obs.create_group("images")
        image_group.create_dataset("overview_left", data=left, chunks=(1, *left.shape[1:]), compression="gzip", compression_opts=1)
        image_group.create_dataset("overview_right", data=right, chunks=(1, *right.shape[1:]), compression="gzip", compression_opts=1)
    return output, len(time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--target_hz", type=float, default=30.0)
    args = parser.parse_args()
    if args.target_hz <= 0:
        parser.error("--target_hz must be positive")
    csv_paths = sorted(args.source_dir.glob("bare_hand_teleop_*.csv"))
    if not csv_paths:
        parser.error("No bare_hand_teleop_*.csv files found")
    for path in csv_paths:
        output, count = convert(path, args.output_dir, args.target_hz)
        print(f"Saved {output} ({count} steps)")


if __name__ == "__main__":
    main()
