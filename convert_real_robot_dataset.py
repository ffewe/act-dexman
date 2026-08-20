"""Convert real-robot teleoperation recordings into ACT HDF5 episodes.

Edit the configuration block below, then run this file without command-line
arguments.  Each source recording can be either:
  * one ZIP containing one CSV and one MP4; or
  * a CSV and MP4 with the same file stem in SOURCE_DIR.
"""

import csv
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np


# ============================== Configuration ==============================
# Put raw CSV+MP4 pairs or ZIP recording packages in this folder.
SOURCE_DIR = Path(r"D:\real_robot_raw_data")
# Converted episode_new<N>.hdf5 files and JSON reports are written here.
OUTPUT_DIR = Path(r"D:\act_real_robot_dataset")

TARGET_HZ = 30.0
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480

# The CSV time zero maps to this time in the MP4:
# video_time_s = elapsed_s + VIDEO_OFFSET_S.
VIDEO_OFFSET_S = 0.0

# CSV values are used as: mapped_joint = raw_joint * scale + offset.
# Keep these defaults only when the CSV and robot controller use identical
# joint order, joint direction, and zero positions.
RIGHT_JOINT_SCALE = np.ones(7, dtype=np.float32)
RIGHT_JOINT_OFFSET_RAD = np.zeros(7, dtype=np.float32)
LEFT_JOINT_SCALE = np.ones(7, dtype=np.float32)
LEFT_JOINT_OFFSET_RAD = np.zeros(7, dtype=np.float32)

# Hand convention stored in qpos and action: 0.0 = closed, 1.0 = open.
OPEN_WORDS = {"1", "true", "yes", "open", "opened"}
CLOSED_WORDS = {"0", "false", "no", "close", "closed"}

# Do not silently overwrite an existing converted episode.
OVERWRITE_EXISTING = False

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class Recording:
    name: str
    csv_path: Path
    video_path: Path
    temporary_dir: Optional[Path] = None


def require_columns(rows, names, recording_name):
    missing = [name for name in names if name not in rows]
    if missing:
        raise ValueError(f"{recording_name}: missing CSV columns: {missing}")


def numeric_column(rows, name):
    try:
        values = np.asarray([float(value) for value in rows[name]], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"Column {name!r} contains a non-numeric value") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"Column {name!r} contains NaN or infinity")
    return values


def hand_column(rows, name):
    output = []
    for raw in rows[name]:
        value = raw.strip().lower()
        if value in OPEN_WORDS:
            output.append(1.0)
        elif value in CLOSED_WORDS:
            output.append(0.0)
        else:
            try:
                output.append(float(value) >= 0.5)
            except ValueError as exc:
                raise ValueError(f"Unexpected hand value {raw!r} in {name}") from exc
    return np.asarray(output, dtype=np.float64)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV has no header")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            if any(value.strip() for value in row.values() if value is not None):
                for name in columns:
                    columns[name].append(row[name])
    if not columns or not next(iter(columns.values())):
        raise ValueError(f"{path}: CSV has no data rows")
    return columns


def strictly_increasing_unique(time_s, *arrays):
    """Sort rows by time and keep the final sample for duplicate timestamps."""
    order = np.argsort(time_s, kind="stable")
    time_s = time_s[order]
    arrays = [array[order] for array in arrays]
    keep = np.r_[time_s[1:] > time_s[:-1], True]
    return (time_s[keep], *(array[keep] for array in arrays))


def interpolate(time_s, values, target_s, categorical=False):
    if categorical:
        indices = np.searchsorted(time_s, target_s, side="left")
        indices = np.clip(indices, 0, len(time_s) - 1)
        previous = np.clip(indices - 1, 0, len(time_s) - 1)
        use_previous = np.abs(target_s - time_s[previous]) <= np.abs(time_s[indices] - target_s)
        return values[np.where(use_previous, previous, indices)]
    flat = values.reshape(len(time_s), -1)
    output = np.empty((len(target_s), flat.shape[1]), dtype=np.float64)
    for index in range(flat.shape[1]):
        output[:, index] = np.interp(target_s, time_s, flat[:, index])
    return output.reshape((len(target_s), *values.shape[1:]))


def csv_arrays(rows, recording_name):
    right_joint_names = [f"right_arm_joint_{index}_rad" for index in range(1, 8)]
    left_joint_names = [f"left_arm_joint_{index}_rad" for index in range(1, 8)]
    force_names = [
        "force_fx_n", "force_fy_n", "force_fz_n", "torque_mx_nm", "torque_my_nm", "torque_mz_nm",
        "left_force_fx_n", "left_force_fy_n", "left_force_fz_n",
        "left_torque_mx_nm", "left_torque_my_nm", "left_torque_mz_nm",
    ]
    tcp_suffixes = ("x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad")
    required = ["elapsed_s", "right_hand_open", "left_hand_open", *right_joint_names, *left_joint_names, *force_names]
    for side in ("left", "right"):
        required.extend(f"{side}_arm_tcp_{suffix}" for suffix in tcp_suffixes)
        required.extend(f"{side}_arm_cmd_{suffix}" for suffix in tcp_suffixes)
        required.extend(f"{side}_tactile_{finger}" for finger in FINGERS)
        required.extend(f"{side}_tactile_{finger}_taxel_{taxel:02d}_raw" for finger in FINGERS for taxel in range(1, 17))
    require_columns(rows, required, recording_name)

    time_s = numeric_column(rows, "elapsed_s")
    right_arm = np.column_stack([numeric_column(rows, name) for name in right_joint_names])
    left_arm = np.column_stack([numeric_column(rows, name) for name in left_joint_names])
    right_arm = right_arm * RIGHT_JOINT_SCALE + RIGHT_JOINT_OFFSET_RAD
    left_arm = left_arm * LEFT_JOINT_SCALE + LEFT_JOINT_OFFSET_RAD
    qpos = np.column_stack([right_arm, hand_column(rows, "right_hand_open"), left_arm, hand_column(rows, "left_hand_open")])

    force = np.column_stack([numeric_column(rows, name) for name in force_names])
    tactile = []
    tcp_measured = []
    tcp_command = []
    for side in ("left", "right"):
        finger = np.column_stack([numeric_column(rows, f"{side}_tactile_{name}") for name in FINGERS])
        taxels = np.stack([
            np.column_stack([numeric_column(rows, f"{side}_tactile_{finger_name}_taxel_{taxel:02d}_raw") for taxel in range(1, 17)])
            for finger_name in FINGERS
        ], axis=1)
        tactile.extend([taxels, finger])
        tcp_measured.append(np.column_stack([numeric_column(rows, f"{side}_arm_tcp_{suffix}") for suffix in tcp_suffixes]))
        tcp_command.append(np.column_stack([numeric_column(rows, f"{side}_arm_cmd_{suffix}") for suffix in tcp_suffixes]))

    return strictly_increasing_unique(time_s, qpos, force, *tactile, *tcp_measured, *tcp_command)


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return fps, count


def decode_frames(path, frame_indices):
    capture = cv2.VideoCapture(str(path))
    frames = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise ValueError(f"Could not decode frame {frame_index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    return np.asarray(frames, dtype=np.uint8)


def discover_recordings(source_dir):
    recordings = []
    for archive in sorted(source_dir.rglob("*.zip")):
        temp_dir = Path(tempfile.mkdtemp(prefix="act_conversion_"))
        with zipfile.ZipFile(archive) as package:
            csv_entries = [entry for entry in package.infolist() if entry.filename.lower().endswith(".csv")]
            video_entries = [entry for entry in package.infolist() if Path(entry.filename).suffix.lower() in {".mp4", ".avi", ".mov"}]
            if len(csv_entries) != 1 or len(video_entries) != 1:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(f"{archive}: expected exactly one CSV and one video")
            package.extract(csv_entries[0], temp_dir)
            package.extract(video_entries[0], temp_dir)
        recordings.append(Recording(archive.stem, temp_dir / csv_entries[0].filename, temp_dir / video_entries[0].filename, temp_dir))

    csv_files = {path.relative_to(source_dir).with_suffix(""): path for path in source_dir.rglob("*.csv")}
    video_files = {}
    for suffix in ("*.mp4", "*.avi", "*.mov"):
        for path in source_dir.rglob(suffix):
            video_files[path.relative_to(source_dir).with_suffix("")] = path
    for stem in sorted(csv_files.keys() & video_files.keys(), key=str):
        recordings.append(Recording(stem.name, csv_files[stem], video_files[stem]))
    return recordings


def next_episode_index(output_dir):
    existing = []
    for path in output_dir.glob("episode_new*.hdf5"):
        suffix = path.stem.removeprefix("episode_new")
        if suffix.isdigit():
            existing.append(int(suffix))
    return max(existing, default=-1) + 1


def write_episode(path, arrays, images, source, report):
    (time_s, qpos, qvel, force, left_taxels, left_finger, right_taxels, right_finger,
     left_tcp, right_tcp, left_cmd, right_cmd, action) = arrays
    with h5py.File(path, "w", rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs["sim"] = False
        root.attrs["control_hz"] = TARGET_HZ
        root.attrs["action_convention"] = "right_arm_7,right_hand,left_arm_7,left_hand"
        root.attrs["hand_convention"] = "0=closed,1=open"
        root.attrs["source_csv"] = str(source.csv_path)
        root.attrs["source_video"] = str(source.video_path)
        root.attrs["video_offset_s"] = VIDEO_OFFSET_S
        root.attrs["conversion_version"] = "1"

        timestamps = root.create_group("timestamps")
        timestamps.create_dataset("control_s", data=time_s.astype(np.float64))
        observations = root.create_group("observations")
        observations.create_dataset("qpos", data=qpos.astype(np.float32))
        observations.create_dataset("qvel", data=qvel.astype(np.float32))
        observations.create_dataset("force", data=force.astype(np.float32))
        images_group = observations.create_group("images")
        images_group.create_dataset("overview", data=images, compression="gzip", compression_opts=4, chunks=(1, OUTPUT_HEIGHT, OUTPUT_WIDTH, 3))

        tactile = observations.create_group("tactile")
        for side, taxels, finger in (("left", left_taxels, left_finger), ("right", right_taxels, right_finger)):
            side_group = tactile.create_group(side)
            side_group.create_dataset("taxels", data=taxels.astype(np.float32))
            side_group.create_dataset("finger", data=finger.astype(np.float32))
        combined = tactile.create_group("combined")
        combined.create_dataset("taxels", data=np.concatenate([left_taxels, right_taxels], axis=1).astype(np.float32))
        combined.create_dataset("finger", data=np.concatenate([left_finger, right_finger], axis=1).astype(np.float32))

        tcp_group = observations.create_group("tcp_measured")
        tcp_group.create_dataset("left", data=left_tcp.astype(np.float32))
        tcp_group.create_dataset("right", data=right_tcp.astype(np.float32))
        commands = root.create_group("commands").create_group("tcp")
        commands.create_dataset("left", data=left_cmd.astype(np.float32))
        commands.create_dataset("right", data=right_cmd.astype(np.float32))
        root.create_dataset("action", data=action.astype(np.float32))
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def convert_recording(recording, output_path):
    rows = read_csv(recording.csv_path)
    (source_time, source_qpos, source_force, left_taxels, left_finger, right_taxels, right_finger,
     left_tcp, right_tcp, left_cmd, right_cmd) = csv_arrays(rows, recording.name)
    if len(source_time) < 3:
        raise ValueError(f"{recording.name}: need at least three unique CSV samples")

    video_fps, video_count = video_metadata(recording.video_path)
    video_duration = (video_count - 1) / video_fps
    source_grid = np.arange(source_time[0], source_time[-1] + 0.5 / TARGET_HZ, 1.0 / TARGET_HZ)
    video_time = source_grid + VIDEO_OFFSET_S
    valid = (video_time >= 0.0) & (video_time <= video_duration)
    grid = source_grid[valid]
    if len(grid) < 2:
        raise ValueError(f"{recording.name}: CSV and video do not overlap for two control samples")

    qpos = interpolate(source_time, source_qpos, grid)
    force = interpolate(source_time, source_force, grid)
    left_taxels = interpolate(source_time, left_taxels, grid)
    left_finger = interpolate(source_time, left_finger, grid)
    right_taxels = interpolate(source_time, right_taxels, grid)
    right_finger = interpolate(source_time, right_finger, grid)
    left_tcp = interpolate(source_time, left_tcp, grid)
    right_tcp = interpolate(source_time, right_tcp, grid)
    left_cmd = interpolate(source_time, left_cmd, grid)
    right_cmd = interpolate(source_time, right_cmd, grid)
    qpos[:, [7, 15]] = interpolate(source_time, source_qpos[:, [7, 15]], grid, categorical=True)

    frame_indices = np.rint((grid + VIDEO_OFFSET_S) * video_fps).astype(int)
    images = decode_frames(recording.video_path, frame_indices)

    # The final state has no recorded next-state action label, so remove it.
    action = qpos[1:]
    qpos = qpos[:-1]
    force = force[:-1]
    left_taxels, left_finger = left_taxels[:-1], left_finger[:-1]
    right_taxels, right_finger = right_taxels[:-1], right_finger[:-1]
    left_tcp, right_tcp = left_tcp[:-1], right_tcp[:-1]
    left_cmd, right_cmd = left_cmd[:-1], right_cmd[:-1]
    images = images[:-1]
    control_time = grid[:-1]
    qvel = np.zeros_like(qpos)
    arm_indices = [*range(7), *range(8, 15)]
    qvel[:, arm_indices] = np.gradient(qpos[:, arm_indices], control_time, axis=0)

    report = {
        "recording": recording.name,
        "source_rows": int(len(source_time)),
        "output_steps": int(len(control_time)),
        "source_duration_s": float(source_time[-1] - source_time[0]),
        "output_hz": TARGET_HZ,
        "video_fps": video_fps,
        "video_duration_s": video_duration,
        "video_offset_s": VIDEO_OFFSET_S,
        "discarded_outside_video_steps": int((~valid).sum()),
        "qpos_min_rad": qpos.min(axis=0).tolist(),
        "qpos_max_rad": qpos.max(axis=0).tolist(),
        "max_abs_joint_step_rad": float(np.abs(np.diff(qpos[:, [*range(7), *range(8, 15)]], axis=0)).max(initial=0.0)),
        "max_abs_force": float(np.abs(force).max(initial=0.0)),
    }
    arrays = (control_time, qpos, qvel, force, left_taxels, left_finger, right_taxels, right_finger,
              left_tcp, right_tcp, left_cmd, right_cmd, action)
    write_episode(output_path, arrays, images, recording, report)


def main():
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Set SOURCE_DIR to an existing folder: {SOURCE_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recordings = discover_recordings(SOURCE_DIR)
    if not recordings:
        raise FileNotFoundError("No CSV+MP4 pairs or ZIP recording packages found in SOURCE_DIR")

    episode_index = next_episode_index(OUTPUT_DIR)
    failures = []
    for recording in recordings:
        output_path = OUTPUT_DIR / f"episode_new{episode_index}.hdf5"
        try:
            if output_path.exists() and not OVERWRITE_EXISTING:
                raise FileExistsError(f"Refusing to overwrite {output_path}")
            convert_recording(recording, output_path)
            print(f"Converted {recording.name} -> {output_path.name}")
            episode_index += 1
        except Exception as exc:
            failures.append({"recording": recording.name, "error": str(exc)})
            print(f"FAILED {recording.name}: {exc}")
        finally:
            if recording.temporary_dir:
                shutil.rmtree(recording.temporary_dir, ignore_errors=True)

    (OUTPUT_DIR / "conversion_summary.json").write_text(json.dumps({"failures": failures}, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Conversion finished with {len(failures)} failure(s); see conversion_summary.json")


if __name__ == "__main__":
    main()
