"""
Raw real-robot dataset -> standardized HDF5 archive.

This script is NOT an ACT training-data converter.

Its purpose is to archive real-robot recordings faithfully so that
different training datasets (ACT or other models) can be generated later.

Input layout:

SOURCE_DIR/
    episode_0001/
        data.csv
        images/
            000000.png
            000001.png
            ...

    episode_0002/
        data.csv
        images/
            000000.png
            000001.png
            ...

Output layout:

OUTPUT_DIR/
    dataset_metadata.json

    episode_0001/
        data.hdf5
        images/
            000000.png
            000001.png
            ...
        source_data.csv

    episode_0002/
        data.hdf5
        images/
            ...

Important design principles:

1. Do NOT define action.
2. Do NOT remove the final frame.
3. Do NOT resample the raw 30 Hz data.
4. Do NOT resize images.
5. Do NOT normalize values.
6. Do NOT binarize the original gripper values.
7. Preserve original timestamps.
8. Preserve original CSV.
9. Preserve raw image files.
10. Store structured data in HDF5 for later processing.
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from PIL import Image


# ============================================================================
# Configuration
# ============================================================================

# Raw dataset directory.
SOURCE_DIR = Path(r"D:\real_robot_raw_data")

# Output archive directory.
OUTPUT_DIR = Path(r"D:\real_robot_archive")

# Expected robot sampling frequency.
EXPECTED_HZ = 30.0

# Allowed timing deviation.
# Example:
# 30 Hz -> dt = 0.033333 s
TIMESTAMP_TOLERANCE = 0.005

# CSV filename inside each episode.
CSV_NAME = "data.csv"

# Image directory inside each episode.
IMAGE_DIR_NAME = "images"

# Accepted image extensions.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# Copy original CSV into output episode directory.
COPY_SOURCE_CSV = True

# Copy original images into output archive.
# True = archive is self-contained.
COPY_SOURCE_IMAGES = True

# If False, existing episode output will raise an error.
OVERWRITE_EXISTING = False

# qpos convention.
#
# IMPORTANT:
# This is only a naming/layout convention for the archive.
# It does NOT define ACT action.
QPOS_LAYOUT = [
    "right_arm_joint_1",
    "right_arm_joint_2",
    "right_arm_joint_3",
    "right_arm_joint_4",
    "right_arm_joint_5",
    "right_arm_joint_6",
    "right_arm_joint_7",
    "right_gripper",
    "left_arm_joint_1",
    "left_arm_joint_2",
    "left_arm_joint_3",
    "left_arm_joint_4",
    "left_arm_joint_5",
    "left_arm_joint_6",
    "left_arm_joint_7",
    "left_gripper",
]

FINGERS = (
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
)

TCP_SUFFIXES = (
    "x_m",
    "y_m",
    "z_m",
    "rx_rad",
    "ry_rad",
    "rz_rad",
)

FORCE_COLUMNS = (
    "force_fx_n",
    "force_fy_n",
    "force_fz_n",
    "torque_mx_nm",
    "torque_my_nm",
    "torque_mz_nm",

    "left_force_fx_n",
    "left_force_fy_n",
    "left_force_fz_n",
    "left_torque_mx_nm",
    "left_torque_my_nm",
    "left_torque_mz_nm",
)


# ============================================================================
# Utility functions
# ============================================================================

def read_csv(path: Path) -> Dict[str, List[str]]:
    """
    Read CSV while preserving the original string representation.

    The original CSV is also copied to the output archive, so this parser
    is only responsible for generating structured numeric datasets.
    """

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV has no header.")

        columns = {
            name: []
            for name in reader.fieldnames
        }

        for row in reader:
            # Ignore completely empty rows.
            if not any(
                value is not None and value.strip()
                for value in row.values()
            ):
                continue

            for name in columns:
                value = row.get(name)

                if value is None:
                    value = ""

                columns[name].append(value)

    if not columns:
        raise ValueError(f"{path}: empty CSV.")

    num_rows = len(next(iter(columns.values())))

    if num_rows == 0:
        raise ValueError(f"{path}: CSV contains no data rows.")

    return columns


def require_columns(
    rows: Dict[str, List[str]],
    columns: List[str],
    episode_name: str,
):
    missing = [
        column
        for column in columns
        if column not in rows
    ]

    if missing:
        raise ValueError(
            f"{episode_name}: missing CSV columns:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )


def numeric_column(
    rows: Dict[str, List[str]],
    name: str,
) -> np.ndarray:

    values = []

    for index, value in enumerate(rows[name]):

        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(
                f"Column '{name}' contains a non-numeric value "
                f"at row {index}: {value!r}"
            ) from exc

        values.append(number)

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"Column '{name}' contains NaN or infinity."
        )

    return values


def raw_column(
    rows: Dict[str, List[str]],
    name: str,
) -> np.ndarray:

    return np.asarray(
        rows[name],
        dtype=object,
    )


def validate_timestamps(
    timestamps: np.ndarray,
    episode_name: str,
) -> Dict[str, float]:

    if len(timestamps) < 2:
        raise ValueError(
            f"{episode_name}: at least two timestamps are required."
        )

    if not np.isfinite(timestamps).all():
        raise ValueError(
            f"{episode_name}: timestamp contains NaN/Inf."
        )

    dt = np.diff(timestamps)

    if np.any(dt <= 0):
        bad = np.where(dt <= 0)[0]

        raise ValueError(
            f"{episode_name}: timestamps are not strictly increasing. "
            f"Problem near indices: {bad[:10].tolist()}"
        )

    median_dt = float(np.median(dt))
    mean_dt = float(np.mean(dt))
    min_dt = float(np.min(dt))
    max_dt = float(np.max(dt))

    measured_hz = 1.0 / median_dt

    expected_dt = 1.0 / EXPECTED_HZ

    timing_error = abs(
        median_dt - expected_dt
    )

    if timing_error > TIMESTAMP_TOLERANCE:

        print(
            f"[WARNING] {episode_name}: "
            f"measured median frequency = "
            f"{measured_hz:.3f} Hz, "
            f"expected = {EXPECTED_HZ:.3f} Hz"
        )

    return {
        "num_samples": len(timestamps),
        "duration_s": float(
            timestamps[-1] - timestamps[0]
        ),
        "mean_dt_s": mean_dt,
        "median_dt_s": median_dt,
        "min_dt_s": min_dt,
        "max_dt_s": max_dt,
        "measured_hz": measured_hz,
    }


def find_images(image_dir: Path) -> List[Path]:

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise ValueError(
            f"No images found in {image_dir}"
        )

    # Prefer numeric filename ordering.
    def sort_key(path: Path):

        try:
            return (
                0,
                int(path.stem),
            )
        except ValueError:
            return (
                1,
                path.name,
            )

    images.sort(key=sort_key)

    return images


def validate_images(
    image_paths: List[Path],
    expected_count: int,
    episode_name: str,
) -> Tuple[int, int]:

    if len(image_paths) != expected_count:

        raise ValueError(
            f"{episode_name}: image count does not match CSV rows.\n"
            f"CSV rows   : {expected_count}\n"
            f"Images     : {len(image_paths)}"
        )

    first_size = None

    for index, path in enumerate(image_paths):

        try:
            with Image.open(path) as image:
                width, height = image.size

                if first_size is None:
                    first_size = (
                        width,
                        height,
                    )

                if (
                    width,
                    height,
                ) != first_size:

                    raise ValueError(
                        f"{episode_name}: image resolution mismatch.\n"
                        f"Frame 0: {first_size}\n"
                        f"Frame {index}: {(width, height)}\n"
                        f"File: {path}"
                    )

        except Exception as exc:
            raise ValueError(
                f"{episode_name}: cannot read image "
                f"{path}"
            ) from exc

    return first_size


def read_hand_values(
    rows: Dict[str, List[str]],
    name: str,
) -> np.ndarray:

    """
    Preserve the original semantic value while converting common
    numeric representations to float.

    IMPORTANT:
    We do NOT threshold continuous values here.

    Example:
        0.12 -> 0.12
        0.47 -> 0.47
        0.91 -> 0.91

    'open' / 'closed' are converted to 1 / 0.
    """

    output = []

    open_words = {
        "open",
        "opened",
        "true",
        "yes",
    }

    closed_words = {
        "close",
        "closed",
        "false",
        "no",
    }

    for index, raw in enumerate(rows[name]):

        value = raw.strip().lower()

        if value in open_words:
            output.append(1.0)

        elif value in closed_words:
            output.append(0.0)

        else:

            try:
                number = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{name}: invalid hand value at "
                    f"row {index}: {raw!r}"
                ) from exc

            if not np.isfinite(number):
                raise ValueError(
                    f"{name}: NaN/Inf at row {index}"
                )

            output.append(number)

    return np.asarray(
        output,
        dtype=np.float64,
    )


# ============================================================================
# CSV -> structured arrays
# ============================================================================

def parse_robot_data(
    rows: Dict[str, List[str]],
    episode_name: str,
):

    right_joint_names = [
        f"right_arm_joint_{i}_rad"
        for i in range(1, 8)
    ]

    left_joint_names = [
        f"left_arm_joint_{i}_rad"
        for i in range(1, 8)
    ]

    required = [
        "elapsed_s",

        "right_hand_open",
        "left_hand_open",

        *right_joint_names,
        *left_joint_names,

        *FORCE_COLUMNS,
    ]

    for side in (
        "left",
        "right",
    ):

        required.extend(
            f"{side}_arm_tcp_{suffix}"
            for suffix in TCP_SUFFIXES
        )

        required.extend(
            f"{side}_arm_cmd_{suffix}"
            for suffix in TCP_SUFFIXES
        )

        required.extend(
            f"{side}_tactile_{finger}"
            for finger in FINGERS
        )

        required.extend(
            f"{side}_tactile_{finger}"
            f"_taxel_{taxel:02d}_raw"
            for finger in FINGERS
            for taxel in range(1, 17)
        )

    require_columns(
        rows,
        required,
        episode_name,
    )

    # ----------------------------------------------------------------------
    # Timestamp
    # ----------------------------------------------------------------------

    timestamp = numeric_column(
        rows,
        "elapsed_s",
    )

    # ----------------------------------------------------------------------
    # qpos
    # ----------------------------------------------------------------------

    right_arm = np.column_stack([
        numeric_column(rows, name)
        for name in right_joint_names
    ])

    left_arm = np.column_stack([
        numeric_column(rows, name)
        for name in left_joint_names
    ])

    right_gripper = read_hand_values(
        rows,
        "right_hand_open",
    )

    left_gripper = read_hand_values(
        rows,
        "left_hand_open",
    )

    qpos = np.column_stack([
        right_arm,
        right_gripper,
        left_arm,
        left_gripper,
    ])

    # ----------------------------------------------------------------------
    # qvel
    #
    # IMPORTANT:
    # We do NOT calculate qvel here.
    #
    # If your CSV has real qvel columns, they can be added later.
    # ----------------------------------------------------------------------

    qvel = None

    # ----------------------------------------------------------------------
    # Force / torque
    # ----------------------------------------------------------------------

    force = np.column_stack([
        numeric_column(rows, name)
        for name in FORCE_COLUMNS
    ])

    # ----------------------------------------------------------------------
    # TCP measured
    # ----------------------------------------------------------------------

    tcp_measured = {}

    for side in (
        "left",
        "right",
    ):

        tcp_measured[side] = np.column_stack([
            numeric_column(
                rows,
                f"{side}_arm_tcp_{suffix}",
            )
            for suffix in TCP_SUFFIXES
        ])

    # ----------------------------------------------------------------------
    # TCP command
    # ----------------------------------------------------------------------

    tcp_command = {}

    for side in (
        "left",
        "right",
    ):

        tcp_command[side] = np.column_stack([
            numeric_column(
                rows,
                f"{side}_arm_cmd_{suffix}",
            )
            for suffix in TCP_SUFFIXES
        ])

    # ----------------------------------------------------------------------
    # Gripper raw values
    # ----------------------------------------------------------------------

    gripper = np.column_stack([
        right_gripper,
        left_gripper,
    ])

    # ----------------------------------------------------------------------
    # Tactile
    # ----------------------------------------------------------------------

    tactile = {}

    for side in (
        "left",
        "right",
    ):

        # Per-finger summary values.
        finger_values = np.column_stack([
            numeric_column(
                rows,
                f"{side}_tactile_{finger}",
            )
            for finger in FINGERS
        ])

        # 16 taxels per finger.
        #
        # Shape:
        #
        #   [T, 5, 16]
        #
        # T = number of samples
        #

        taxels = np.stack([
            np.column_stack([
                numeric_column(
                    rows,
                    f"{side}_tactile_{finger}"
                    f"_taxel_{taxel:02d}_raw",
                )
                for taxel in range(1, 17)
            ])
            for finger in FINGERS
        ], axis=1)

        tactile[side] = {
            "finger": finger_values,
            "taxels": taxels,
        }

    return {
        "timestamp": timestamp,
        "qpos": qpos,
        "qvel": qvel,
        "gripper": gripper,
        "force": force,
        "tcp_measured": tcp_measured,
        "tcp_command": tcp_command,
        "tactile": tactile,
    }


# ============================================================================
# HDF5 writing
# ============================================================================

def write_string_dataset(
    group,
    name: str,
    values,
):

    dtype = h5py.string_dtype(
        encoding="utf-8"
    )

    group.create_dataset(
        name,
        data=np.asarray(
            values,
            dtype=object,
        ),
        dtype=dtype,
    )


def write_hdf5(
    output_path: Path,
    data,
    image_paths: List[Path],
    episode_name: str,
    timing_report: Dict[str, float],
    image_size: Tuple[int, int],
):

    timestamp = data["timestamp"]
    qpos = data["qpos"]
    gripper = data["gripper"]
    force = data["force"]

    with h5py.File(
        output_path,
        "w",
    ) as root:

        # ==================================================================
        # Root metadata
        # ==================================================================

        root.attrs["dataset_type"] = (
            "raw_robot_archive"
        )

        root.attrs["sim"] = False

        root.attrs["expected_hz"] = EXPECTED_HZ

        root.attrs["measured_hz"] = (
            timing_report["measured_hz"]
        )

        root.attrs["episode_name"] = episode_name

        root.attrs["num_samples"] = len(timestamp)

        root.attrs["duration_s"] = (
            timing_report["duration_s"]
        )

        root.attrs["image_width"] = image_size[0]

        root.attrs["image_height"] = image_size[1]

        root.attrs["image_count"] = len(
            image_paths
        )

        root.attrs["image_format"] = (
            image_paths[0].suffix.lower().lstrip(".")
        )

        root.attrs["conversion_version"] = "2.0"

        root.attrs["contains_action"] = False

        # ==================================================================
        # timestamps
        # ==================================================================

        timestamps = root.create_group(
            "timestamps"
        )

        timestamps.create_dataset(
            "control_s",
            data=timestamp.astype(
                np.float64
            ),
        )

        # ==================================================================
        # observations
        # ==================================================================

        observations = root.create_group(
            "observations"
        )

        observations.create_dataset(
            "qpos",
            data=qpos.astype(
                np.float64
            ),
        )

        observations.create_dataset(
            "gripper",
            data=gripper.astype(
                np.float64
            ),
        )

        observations.create_dataset(
            "force",
            data=force.astype(
                np.float64
            ),
        )

        # ------------------------------------------------------------------
        # qpos metadata
        # ------------------------------------------------------------------

        observations["qpos"].attrs[
            "layout"
        ] = json.dumps(
            QPOS_LAYOUT
        )

        observations["qpos"].attrs[
            "units"
        ] = "joint_rad_and_gripper_native_units"

        observations["gripper"].attrs[
            "layout"
        ] = json.dumps([
            "right_gripper",
            "left_gripper",
        ])

        observations["gripper"].attrs[
            "note"
        ] = (
            "Original gripper values; "
            "not binarized."
        )

        # ------------------------------------------------------------------
        # qvel
        # ------------------------------------------------------------------

        if data["qvel"] is not None:

            observations.create_dataset(
                "qvel",
                data=data["qvel"].astype(
                    np.float64
                ),
            )

        # ------------------------------------------------------------------
        # TCP measured
        # ------------------------------------------------------------------

        tcp_group = observations.create_group(
            "tcp_measured"
        )

        for side in (
            "left",
            "right",
        ):

            dataset = tcp_group.create_dataset(
                side,
                data=data[
                    "tcp_measured"
                ][side].astype(
                    np.float64
                ),
            )

            dataset.attrs[
                "layout"
            ] = json.dumps([
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
            ])

        # ------------------------------------------------------------------
        # Tactile
        # ------------------------------------------------------------------

        tactile_group = observations.create_group(
            "tactile"
        )

        for side in (
            "left",
            "right",
        ):

            side_group = tactile_group.create_group(
                side
            )

            side_group.create_dataset(
                "finger",
                data=data[
                    "tactile"
                ][side]["finger"].astype(
                    np.float64
                ),
            )

            side_group.create_dataset(
                "taxels",
                data=data[
                    "tactile"
                ][side]["taxels"].astype(
                    np.float64
                ),
            )

        # ==================================================================
        # commands
        # ==================================================================

        commands = root.create_group(
            "commands"
        )

        tcp_commands = commands.create_group(
            "tcp"
        )

        for side in (
            "left",
            "right",
        ):

            dataset = tcp_commands.create_dataset(
                side,
                data=data[
                    "tcp_command"
                ][side].astype(
                    np.float64
                ),
            )

            dataset.attrs[
                "layout"
            ] = json.dumps([
                "x",
                "y",
                "z",
                "rx",
                "ry",
                "rz",
            ])

        # ==================================================================
        # images
        #
        # IMPORTANT:
        # We do NOT put 3840x1080 RGB frames into HDF5 here.
        #
        # HDF5 stores the image file paths.
        # The original PNG/JPG files remain in the episode directory.
        # ==================================================================

        images_group = root.create_group(
            "images"
        )

        image_paths_relative = [
            f"images/{path.name}"
            for path in image_paths
        ]

        write_string_dataset(
            images_group,
            "overview",
            image_paths_relative,
        )

        images_group["overview"].attrs[
            "fps"
        ] = EXPECTED_HZ

        images_group["overview"].attrs[
            "width"
        ] = image_size[0]

        images_group["overview"].attrs[
            "height"
        ] = image_size[1]

        images_group["overview"].attrs[
            "color"
        ] = "RGB"

        # ==================================================================
        # raw information
        #
        # Store original gripper strings as well.
        # ==================================================================

        raw_group = root.create_group(
            "raw"
        )

        raw_gripper = raw_group.create_group(
            "gripper"
        )

        write_string_dataset(
            raw_gripper,
            "right_hand_open",
            raw_column(
                rows_global,
                "right_hand_open",
            ),
        )

        write_string_dataset(
            raw_gripper,
            "left_hand_open",
            raw_column(
                rows_global,
                "left_hand_open",
            ),
        )

        # ==================================================================
        # No action!
        # ==================================================================

        root.attrs[
            "action_definition"
        ] = "NOT_DEFINED_IN_RAW_ARCHIVE"


# ============================================================================
# Dataset metadata
# ============================================================================

def write_dataset_metadata(
    output_dir: Path,
    episodes: List[Dict],
):

    metadata = {
        "dataset_type": "raw_robot_archive",
        "version": "2.0",

        "description": (
            "Standardized archive of real-robot "
            "teleoperation recordings. "
            "This dataset is model-agnostic and "
            "does not define an action space."
        ),

        "expected_robot_hz": EXPECTED_HZ,

        "image": {
            "source": "individual_image_files",
            "resize": False,
            "normalization": False,
        },

        "qpos_layout": QPOS_LAYOUT,

        "gripper": {
            "stored_as_observation": True,
            "preserve_raw_value": True,
            "binarization": False,
        },

        "action": {
            "defined": False,
            "note": (
                "Action will be defined later "
                "when constructing a model-specific "
                "training dataset."
            ),
        },

        "episodes": episodes,
    }

    path = (
        output_dir /
        "dataset_metadata.json"
    )

    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ============================================================================
# Episode conversion
# ============================================================================

def convert_episode(
    episode_dir: Path,
    output_dir: Path,
):

    episode_name = episode_dir.name

    csv_path = (
        episode_dir /
        CSV_NAME
    )

    image_dir = (
        episode_dir /
        IMAGE_DIR_NAME
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{episode_name}: "
            f"CSV not found: {csv_path}"
        )

    print()
    print("=" * 70)
    print(
        f"Processing {episode_name}"
    )
    print("=" * 70)

    # ----------------------------------------------------------------------
    # Read CSV
    # ----------------------------------------------------------------------

    rows = read_csv(
        csv_path
    )

    global rows_global
    rows_global = rows

    num_rows = len(
        next(iter(rows.values()))
    )

    print(
        f"CSV rows: {num_rows}"
    )

    # ----------------------------------------------------------------------
    # Parse robot data
    # ----------------------------------------------------------------------

    data = parse_robot_data(
        rows,
        episode_name,
    )

    timestamp = data["timestamp"]

    if len(timestamp) != num_rows:
        raise ValueError(
            f"{episode_name}: timestamp length mismatch."
        )

    # ----------------------------------------------------------------------
    # Validate timestamps
    # ----------------------------------------------------------------------

    timing_report = validate_timestamps(
        timestamp,
        episode_name,
    )

    print(
        f"Measured frequency: "
        f"{timing_report['measured_hz']:.3f} Hz"
    )

    print(
        f"Duration: "
        f"{timing_report['duration_s']:.3f} s"
    )

    # ----------------------------------------------------------------------
    # Find images
    # ----------------------------------------------------------------------

    image_paths = find_images(
        image_dir
    )

    print(
        f"Images: {len(image_paths)}"
    )

    image_size = validate_images(
        image_paths,
        num_rows,
        episode_name,
    )

    print(
        f"Image size: "
        f"{image_size[0]} x "
        f"{image_size[1]}"
    )

    # ----------------------------------------------------------------------
    # Output directory
    # ----------------------------------------------------------------------

    episode_output = (
        output_dir /
        episode_name
    )

    if episode_output.exists():

        if not OVERWRITE_EXISTING:

            raise FileExistsError(
                f"Output already exists: "
                f"{episode_output}"
            )

        shutil.rmtree(
            episode_output
        )

    episode_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Copy original CSV
    # ----------------------------------------------------------------------

    if COPY_SOURCE_CSV:

        shutil.copy2(
            csv_path,
            episode_output /
            "source_data.csv",
        )

    # ----------------------------------------------------------------------
    # Copy original images
    # ----------------------------------------------------------------------

    archive_image_dir = (
        episode_output /
        "images"
    )

    archive_image_dir.mkdir(
        exist_ok=True
    )

    if COPY_SOURCE_IMAGES:

        archived_image_paths = []

        for index, source_image in enumerate(
            image_paths
        ):

            suffix = (
                source_image
                .suffix
                .lower()
            )

            target = (
                archive_image_dir /
                f"{index:06d}{suffix}"
            )

            shutil.copy2(
                source_image,
                target,
            )

            archived_image_paths.append(
                target
            )

    else:

        archived_image_paths = image_paths

    # ----------------------------------------------------------------------
    # Write HDF5
    # ----------------------------------------------------------------------

    hdf5_path = (
        episode_output /
        "data.hdf5"
    )

    write_hdf5(
        hdf5_path,
        data,
        archived_image_paths,
        episode_name,
        timing_report,
        image_size,
    )

    # ----------------------------------------------------------------------
    # Episode metadata
    # ----------------------------------------------------------------------

    episode_metadata = {
        "episode_name": episode_name,
        "num_samples": num_rows,
        "duration_s": timing_report[
            "duration_s"
        ],
        "measured_hz": timing_report[
            "measured_hz"
        ],
        "image_count": len(
            archived_image_paths
        ),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "hdf5": "data.hdf5",
        "source_csv": "source_data.csv",
        "action_defined": False,
    }

    (
        episode_output /
        "episode_metadata.json"
    ).write_text(
        json.dumps(
            episode_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[OK] {episode_name}"
    )

    return episode_metadata


# ============================================================================
# Main
# ============================================================================

def main():

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"SOURCE_DIR does not exist:\n"
            f"{SOURCE_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Discover episodes
    # ----------------------------------------------------------------------

    episodes = [
        path
        for path in sorted(
            SOURCE_DIR.iterdir()
        )
        if path.is_dir()
    ]

    if not episodes:

        raise RuntimeError(
            f"No episode directories found "
            f"in {SOURCE_DIR}"
        )

    print(
        f"Found {len(episodes)} episode(s)."
    )

    # ----------------------------------------------------------------------
    # Convert
    # ----------------------------------------------------------------------

    episode_reports = []

    for episode_dir in episodes:

        try:

            report = convert_episode(
                episode_dir,
                OUTPUT_DIR,
            )

            episode_reports.append(
                report
            )

        except Exception as exc:

            print()
            print(
                f"[ERROR] "
                f"{episode_dir.name}: "
                f"{exc}"
            )

            # Continue with other episodes.
            continue

    # ----------------------------------------------------------------------
    # Dataset metadata
    # ----------------------------------------------------------------------

    write_dataset_metadata(
        OUTPUT_DIR,
        episode_reports,
    )

    print()
    print("=" * 70)
    print("Conversion finished.")
    print(
        f"Successful episodes: "
        f"{len(episode_reports)} / "
        f"{len(episodes)}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()