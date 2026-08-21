"""Quest 3 bare-hand teleoperation runner.

This is the bare-hand counterpart to ``combined_teleop.py``.  Wrist poses still
use teleop3's existing Unity UDP protocol, while finger commands arrive from
QuestBareHandTeleop.cs on a separate UDP port for each hand.
"""

import argparse
from collections import deque
from datetime import datetime
import math
from pathlib import Path
import queue
import shutil
import socket
import struct
import sys
import threading
import time

import combined_teleop as base
import connect2 as hand_driver
from read_m4313m1a import M4313M1AReader


SCRIPT_DIR = Path(__file__).resolve().parent
FINGER_MAGIC = b"QHND"
FINGER_VERSION = 1
FINGER_PACKET = struct.Struct("<4sBBH6f")
DEFAULT_LEFT_FINGER_PORT = 5015
DEFAULT_RIGHT_FINGER_PORT = 5016
DEFAULT_HAND_COMMAND_TIMEOUT = 0.35
DEFAULT_LEFT_FORCE_TORQUE_PORT = "COM6"
DEFAULT_RIGHT_FORCE_TORQUE_PORT = "COM4"
HAND_CONTROL_HZ = 60.0
DEFAULT_BARE_HAND_RECORD_HZ = 30.0
DEFAULT_RECORD_IMAGE_SCALE = 0.5
DEFAULT_RECORD_IMAGE_QUALITY = 80
HANDSHAKE_INTERVAL = 1.0
FORCE_EXIT_IDLE_TIMEOUT = 2.0
FIXED_THUMB_ROTATION_DEG = 90


class DualM4313M1AReader:
    """Read both arm force/torque sensors as one recorder auxiliary source."""

    RIGHT_COLUMNS = tuple(M4313M1AReader.COLUMNS)
    LEFT_COLUMNS = tuple(f"left_{column}" for column in RIGHT_COLUMNS)
    COLUMNS = RIGHT_COLUMNS + LEFT_COLUMNS

    def __init__(
        self,
        right_port,
        left_port,
        baudrate,
        stop_event,
        stale_after=0.5,
    ):
        self.right = M4313M1AReader(
            right_port, baudrate, stop_event, stale_after=stale_after
        )
        self.left = M4313M1AReader(
            left_port, baudrate, stop_event, stale_after=stale_after
        )

    def start(self):
        try:
            self.right.start()
            self.left.start()
        except Exception:
            self.stop()
            raise

    def snapshot(self, now=None):
        right_values = self.right.snapshot(now)
        left_values = self.left.snapshot(now)
        if right_values is None or left_values is None:
            return None
        return list(right_values) + list(left_values)

    def stop(self):
        self.left.stop()
        self.right.stop()


class RecordingFrameSynchronizer:
    """Issue one image-frame token for every successfully written state sample."""

    def __init__(self):
        self._condition = threading.Condition()
        self._targets = deque()
        self._requested = 0

    def request_frame(self, target_time=None):
        if target_time is None:
            target_time = time.monotonic()
        with self._condition:
            self._targets.append(float(target_time))
            self._requested += 1
            self._condition.notify_all()

    def next_target(self):
        with self._condition:
            if not self._targets:
                return None
            return self._targets[0]

    def consume_frame_request(self):
        with self._condition:
            if not self._targets:
                return None
            target_time = self._targets.popleft()
            if not self._targets:
                self._condition.notify_all()
            return target_time

    def wait_until_drained(self, timeout):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._targets:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def reset(self):
        with self._condition:
            if self._targets:
                raise RuntimeError(
                    "cannot reset image synchronization with pending frames"
                )
            self._targets.clear()
            self._requested = 0

    @property
    def requested(self):
        with self._condition:
            return self._requested

    @property
    def pending(self):
        with self._condition:
            return len(self._targets)


class Zed720CameraStreamer(base.CameraStreamer):
    """ZED Mini streamer with synchronized episode-frame recording."""

    FRAME_QUEUE_SIZE = 16
    FRAME_WRITER_COUNT = 2

    def __init__(
        self,
        port,
        stop_event,
        jpeg_quality=90,
        frame_directory=None,
        frame_sink=None,
        frame_start_event=None,
        frame_synchronizer=None,
        record_image_scale=DEFAULT_RECORD_IMAGE_SCALE,
        record_image_quality=DEFAULT_RECORD_IMAGE_QUALITY,
    ):
        super().__init__(
            port,
            stop_event,
            jpeg_quality,
            video_path=frame_directory or getattr(frame_sink, "path", None),
            video_start_event=frame_start_event,
        )
        self._frame_lock = threading.Lock()
        self._frame_result_lock = threading.Lock()
        self.frame_directory = (
            Path(frame_directory) if frame_directory is not None else None
        )
        self.frame_sink = frame_sink
        self.frame_start_event = frame_start_event
        self.frame_synchronizer = frame_synchronizer
        if not 0.0 < record_image_scale <= 1.0:
            raise ValueError(
                "record_image_scale must be greater than zero and at most 1"
            )
        if not 1 <= record_image_quality <= 100:
            raise ValueError("record_image_quality must be from 1 to 100")
        self.record_image_scale = record_image_scale
        self.record_image_quality = record_image_quality
        self.frames_queued = 0
        self.frames_written = 0
        self._previous_frame = None
        self._previous_frame_time = None
        self._frame_queue = queue.Queue(maxsize=self.FRAME_QUEUE_SIZE)
        self._frame_workers = []
        self._frame_writer_error = None

    @staticmethod
    def configure_init_parameters(sl, init_params):
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE

    def start(self):
        try:
            self.cv2 = base.importlib.import_module("cv2")
            self.sl = base.importlib.import_module("pyzed.sl")
        except ImportError as exc:
            raise RuntimeError(
                "camera dependencies are unavailable; run --check-dependencies"
            ) from exc

        self.zed = self.sl.Camera()
        init_params = self.sl.InitParameters()
        self.configure_init_parameters(self.sl, init_params)

        error = self.zed.open(init_params)
        if error != self.sl.ERROR_CODE.SUCCESS:
            self.zed.close()
            self.zed = None
            raise RuntimeError(f"ZED camera open failed: {error}")

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setblocking(False)
            self.server_socket.bind(("0.0.0.0", self.port))
            self.server_socket.listen(1)
        except Exception:
            self.stop()
            raise

        self._start_frame_workers()

        self.thread = threading.Thread(
            target=self._run, name="zed-1080p-streamer", daemon=True
        )
        self.thread.start()
        print(
            f"[Check 1/3] Camera ready; HD1080 @ 30 FPS stream listening "
            f"on TCP {self.port}."
        )

    def _start_frame_workers(self):
        if self._frame_workers:
            return
        self._frame_workers = [
            threading.Thread(
                target=self._frame_writer,
                name=f"zed-frame-writer-{index + 1}",
                daemon=True,
            )
            for index in range(self.FRAME_WRITER_COUNT)
        ]
        for worker in self._frame_workers:
            worker.start()

    def _frame_writer(self):
        while True:
            item = self._frame_queue.get()
            try:
                if item is None:
                    return
                frame_number, path, frame = item
                if self.frame_sink is not None:
                    self.frame_sink.append_image(frame_number, frame)
                else:
                    ok = self.cv2.imwrite(
                        str(path),
                        frame,
                        [
                            int(self.cv2.IMWRITE_JPEG_QUALITY),
                            self.record_image_quality,
                        ],
                    )
                    if not ok:
                        raise RuntimeError(f"could not write image {path}")
                with self._frame_result_lock:
                    self.frames_written += 1
            except Exception as exc:
                with self._frame_result_lock:
                    if self._frame_writer_error is None:
                        self._frame_writer_error = exc
                print(f"[Camera] Image recording failure: {exc}", file=sys.stderr)
                self.stop_event.set()
            finally:
                self._frame_queue.task_done()

    def _prepare_recording_frame(self, frame):
        crop_top = frame.shape[0] // 3
        image = frame[crop_top:, :]
        if self.record_image_scale < 1.0:
            height, width = image.shape[:2]
            output_size = (
                max(1, round(width * self.record_image_scale)),
                max(1, round(height * self.record_image_scale)),
            )
            return self.cv2.resize(
                image,
                output_size,
                interpolation=self.cv2.INTER_AREA,
            )
        return image.copy()

    def _write_video_frame(self, frame):
        """Queue the frame nearest the next state-sample timestamp."""
        with self._frame_lock:
            if self.frame_directory is None and self.frame_sink is None:
                return
            capture_time = time.monotonic()
            target_time = (
                self.frame_synchronizer.next_target()
                if self.frame_synchronizer is not None
                else capture_time
            )
            if target_time is None:
                self._previous_frame = frame.copy()
                self._previous_frame_time = capture_time
                return
            if capture_time < target_time:
                self._previous_frame = frame.copy()
                self._previous_frame_time = capture_time
                return
            try:
                previous_frame = self._previous_frame
                previous_time = self._previous_frame_time
                if (
                    previous_frame is not None
                    and abs(previous_time - target_time)
                    <= abs(capture_time - target_time)
                ):
                    selected_frame = previous_frame
                    # Keep this current frame available for a later target.
                    self._previous_frame = frame.copy()
                    self._previous_frame_time = capture_time
                else:
                    selected_frame = frame
                    self._previous_frame = None
                    self._previous_frame_time = None
                consumed_target = (
                    self.frame_synchronizer.consume_frame_request()
                    if self.frame_synchronizer is not None
                    else capture_time
                )
                if consumed_target is None:
                    return
                frame_number = self.frames_queued + 1
                if frame_number == 1 and self.frame_directory is not None:
                    if self.frame_directory.exists():
                        if not self.frame_directory.is_dir():
                            raise RuntimeError(
                                f"image path is not a directory: "
                                f"{self.frame_directory}"
                            )
                        if any(self.frame_directory.iterdir()):
                            raise RuntimeError(
                                f"image directory is not empty: "
                                f"{self.frame_directory}"
                            )
                    else:
                        self.frame_directory.mkdir(parents=True)
                image = self._prepare_recording_frame(selected_frame)
                frame_path = (
                    self.frame_directory / f"frame_{frame_number:06d}.jpg"
                    if self.frame_directory is not None
                    else None
                )
                self._frame_queue.put((frame_number, frame_path, image))
                self.frames_queued = frame_number
            except Exception as exc:
                with self._frame_result_lock:
                    if self._frame_writer_error is None:
                        self._frame_writer_error = exc
                print(f"[Camera] Image recording failure: {exc}", file=sys.stderr)
                self.stop_event.set()

    def finish_recording(self):
        """Finish the current image sequence without stopping camera streaming."""
        if self.frame_synchronizer is not None:
            thread_is_alive = self.thread is not None and self.thread.is_alive()
            drain_timeout = 2.0 if thread_is_alive else 0.0
            if not self.frame_synchronizer.wait_until_drained(drain_timeout):
                error = RuntimeError(
                    "image synchronization timed out with "
                    f"{self.frame_synchronizer.pending} pending frame(s)"
                )
                with self._frame_result_lock:
                    if self._frame_writer_error is None:
                        self._frame_writer_error = error
                print(f"[Camera] {error}", file=sys.stderr)
        if self.frame_start_event is not None:
            self.frame_start_event.clear()
        with self._frame_lock:
            sink = self.frame_sink
            path = self.frame_directory or getattr(sink, "path", None)
            queued_frames = self.frames_queued
            self.frame_directory = None
            self.video_path = None
            self.video_start_event = None
            self._previous_frame = None
            self._previous_frame_time = None
        self._frame_queue.join()
        with self._frame_lock:
            self.frame_sink = None
        with self._frame_result_lock:
            written_frames = self.frames_written
        requested_frames = (
            self.frame_synchronizer.requested
            if self.frame_synchronizer is not None
            else queued_frames
        )
        if written_frames != requested_frames:
            error = RuntimeError(
                "image/state alignment mismatch: "
                f"{written_frames} image(s) for {requested_frames} state sample(s)"
            )
            with self._frame_result_lock:
                if self._frame_writer_error is None:
                    self._frame_writer_error = error
            print(f"[Camera] {error}", file=sys.stderr)
        was_created = path is not None and written_frames > 0
        if was_created:
            print(f"[Record] Finished {written_frames} synchronized image(s) for {path}")
        return path, was_created

    @property
    def frame_writer_error(self):
        with self._frame_result_lock:
            return self._frame_writer_error

    def start_recording(self, frame_directory=None, frame_start_event=None, frame_sink=None):
        """Arm a new image sequence while keeping the camera thread alive."""
        with self._frame_lock:
            if self.frame_directory is not None or self.frame_sink is not None:
                raise RuntimeError("cannot start image recording while one is active")
            self.frame_directory = (
                Path(frame_directory) if frame_directory is not None else None
            )
            self.frame_sink = frame_sink
            self.frame_start_event = frame_start_event
            self.video_path = self.frame_directory or getattr(frame_sink, "path", None)
            self.video_start_event = frame_start_event
            self.frames_queued = 0
            self._previous_frame = None
            self._previous_frame_time = None
            with self._frame_result_lock:
                self.frames_written = 0
            if self.frame_synchronizer is not None:
                self.frame_synchronizer.reset()

    def stop(self):
        super().stop()
        self._frame_queue.join()
        for _worker in self._frame_workers:
            self._frame_queue.put(None)
        for worker in self._frame_workers:
            worker.join(timeout=2.0)
        self._frame_workers = []


class QuestHandSubsystem:
    """Drive the serial robot hands through native firmware force control."""

    def __init__(
        self,
        args,
        sides,
        stop_event,
        record_hand_angles=False,
        recording_loss_event=None,
        teleop_enabled_event=None,
        recording_loss_settled_event=None,
    ):
        self.args = args
        self.sides = sides
        self.stop_event = stop_event
        self.record_hand_angles = record_hand_angles
        self.recording_loss_event = recording_loss_event
        self.recording_loss_settled_event = recording_loss_settled_event
        self.teleop_enabled_event = teleop_enabled_event or threading.Event()
        if teleop_enabled_event is None:
            self.teleop_enabled_event.set()
        self.all_hands_were_active = False
        self.recording_loss_reported = False
        self.module = None
        self.hands = {}
        self.command_sockets = {}
        self.command_targets = {}
        self.last_packet_time = {side: None for side in sides}
        self.command_active = {side: False for side in sides}
        self.quest_command_ready_time = {side: None for side in sides}
        self.last_robot_command = {
            side: [FIXED_THUMB_ROTATION_DEG, 0, 0, 0, 0, 0]
            for side in sides
        }
        # Keep the Quest command as an ACT action signal. Native force control
        # may intentionally stop forwarding every intermediate finger pose.
        self.latest_quest_angles = {
            side: [FIXED_THUMB_ROTATION_DEG, 0, 0, 0, 0, 0]
            for side in sides
        }
        # The CSV action signal is intentionally binary.  Keep this separate
        # from ``latest_quest_angles`` because the angles are still needed by
        # the native force-control trigger/release logic.
        self.latest_quest_hand_open = {side: True for side in sides}
        self.native_force_active = {side: False for side in sides}
        self.native_force_attempted = {side: False for side in sides}
        self.latest_measured_angles = {side: [""] * 6 for side in sides}
        self.measured_angle_valid = {side: False for side in sides}
        self.state_lock = threading.Lock()
        self.tactile_socket = None
        self.threads = []

    @staticmethod
    def decode_packet(data, expected_side):
        """Return six degrees values, or None for a malformed/wrong-side packet."""
        if len(data) != FINGER_PACKET.size:
            return None
        try:
            magic, version, side_id, reserved, *angles = FINGER_PACKET.unpack(data)
        except struct.error:
            return None
        expected_side_id = 1 if expected_side == "right" else 0
        if (
            magic != FINGER_MAGIC
            or version != FINGER_VERSION
            or side_id != expected_side_id
            or reserved != 0
            or not all(math.isfinite(value) and 0.0 <= value <= 90.0 for value in angles)
        ):
            return None
        return angles

    @staticmethod
    def build_robot_command(angles):
        """Clamp finger commands and keep the thumb rotation fixed at 90 degrees."""
        command = [
            int(round(max(0.0, min(90.0, value))))
            for value in angles
        ]
        command[0] = FIXED_THUMB_ROTATION_DEG
        return command

    def _send_command(self, side, controller, command):
        """Start or release the hand firmware's native force grasp."""
        force_config = hand_driver.get_force_control_config(side == "right")
        trigger_angle = getattr(self.args, f"{side}_native_force_trigger_angle")
        release_angle = getattr(self.args, f"{side}_native_force_release_angle")
        grasp_angle = sum(command[1:]) / 5.0
        if self.native_force_active[side]:
            if grasp_angle <= release_angle:
                stopped = controller.stop_force_control()
                if stopped:
                    self._wait_for_force_idle(side, controller)
                    # The firmware exits force mode by moving to zero. Wait for
                    # that motion, then make the configured initial pose final.
                    self._initialize_force_position(side, controller)
                    self.native_force_active[side] = False
                    self.native_force_attempted[side] = False
                    print(f"[Hand] {side.upper()} native force grasp released.")
                else:
                    print(
                        f"[Hand] {side.upper()} native force release was rejected; "
                        "will retry."
                    )
            return

        if grasp_angle < trigger_angle:
            self.native_force_attempted[side] = False
            return

        if self.native_force_attempted[side]:
            return
        self.native_force_attempted[side] = True
        # Native force control always starts from the configured side-specific
        # FORCE_CONTROL_INITIAL_ANGLES, never from the current Quest gesture.
        initial_angles = list(force_config["initial_angles"])
        configured = controller.configure_force_control(
            initial_angles=initial_angles,
            maximum_angles=force_config["maximum_angles"],
            thresholds=getattr(self.args, f"{side}_native_force_threshold"),
            speed=getattr(self.args, f"{side}_native_force_speed"),
            enables=force_config["enables"],
        )
        if configured and controller.start_force_control():
            self.native_force_active[side] = True
            self.last_robot_command[side] = command
            print(f"[Hand] {side.upper()} native force grasp started.")
        else:
            print(
                f"[Hand] {side.upper()} native force grasp was rejected; "
                "holding the current position."
            )

    def _open_hand(self, side, controller):
        if self.native_force_active[side]:
            if not controller.stop_force_control():
                print(
                    f"[Hand] {side.upper()} native force release was rejected; "
                    "will retry."
                )
                return False
            self._wait_for_force_idle(side, controller)
        self._initialize_force_position(side, controller)
        self.native_force_active[side] = False
        self.native_force_attempted[side] = False
        self.quest_command_ready_time[side] = None
        self.last_robot_command[side] = [FIXED_THUMB_ROTATION_DEG, 0, 0, 0, 0, 0]
        with self.state_lock:
            self.latest_quest_hand_open[side] = True
        return True

    def _initialize_force_position(self, side, controller):
        """Move one hand to its configured force-control initial angles."""
        config = hand_driver.get_force_control_config(side == "right")
        initial_angles = list(config["initial_angles"])
        controller.send_angles(initial_angles)
        self.last_robot_command[side] = initial_angles
        print(
            f"[Hand] {side.upper()} initialized to native force angles "
            f"{initial_angles}."
        )

    @staticmethod
    def _wait_for_force_idle(side, controller):
        waiter = getattr(controller, "wait_for_force_control_idle", None)
        if callable(waiter) and not waiter(timeout=FORCE_EXIT_IDLE_TIMEOUT):
            print(
                f"[Hand] {side.upper()} force-control motion did not report "
                "idle before the initial pose reset."
            )

    def _begin_quest_session(self, side, controller, now):
        """Initialize one hand when its Quest command stream becomes active."""
        if self.native_force_active[side]:
            if not controller.stop_force_control():
                print(
                    f"[Hand] {side.upper()} could not exit the previous force "
                    "grasp; initial gesture pending."
                )
                return False
            self._wait_for_force_idle(side, controller)
            self.native_force_active[side] = False
            self.native_force_attempted[side] = False

        config = hand_driver.get_force_control_config(side == "right")
        hold_seconds = max(0.0, float(config["quest_start_hold_seconds"]))
        self._initialize_force_position(side, controller)
        self.quest_command_ready_time[side] = now + hold_seconds
        print(
            f"[Hand] {side.upper()} Quest session started; holding initial "
            f"gesture for {hold_seconds:.2f} s."
        )
        return True

    def _handle_quest_angles(self, side, controller, angles, now):
        """Handle one valid Quest packet and initialize newly active sessions."""
        command = self.build_robot_command(angles)
        with self.state_lock:
            self.latest_quest_angles[side] = list(command)
            self._update_quest_hand_state_locked(side, command)
        if not self.command_active[side]:
            if not self._begin_quest_session(side, controller, now):
                self.last_packet_time[side] = now
                return
            self.command_active[side] = True
        elif now >= self.quest_command_ready_time[side]:
            self._send_command(side, controller, command)
        self.last_packet_time[side] = now

    def _observe_quest_angles(self, side, angles, now):
        """Drain fresh input while teleoperation is paused without moving a hand."""
        command = self.build_robot_command(angles)
        with self.state_lock:
            self.latest_quest_angles[side] = list(command)
            self._update_quest_hand_state_locked(side, command)
        self.last_packet_time[side] = now

    def _update_quest_hand_state_locked(self, side, command):
        """Apply trigger/release hysteresis and update the binary hand state."""
        grasp_angle = sum(command[1:]) / 5.0
        trigger_angle = getattr(self.args, f"{side}_native_force_trigger_angle")
        release_angle = getattr(self.args, f"{side}_native_force_release_angle")
        if self.latest_quest_hand_open[side]:
            if grasp_angle >= trigger_angle:
                self.latest_quest_hand_open[side] = False
        elif grasp_angle <= release_angle:
            self.latest_quest_hand_open[side] = True

    def _update_combined_tracking_state(self):
        if self.recording_loss_event is None or len(self.sides) != 2:
            return
        if all(self.command_active[side] for side in self.sides):
            self.all_hands_were_active = True
            return
        if (
            self.all_hands_were_active
            and not self.recording_loss_reported
            and not any(self.command_active[side] for side in self.sides)
        ):
            self.recording_loss_reported = True
            self.teleop_enabled_event.clear()
            self.recording_loss_event.set()
            print(
                "[Record] Both Quest hand streams were lost; "
                "teleoperation paused."
            )

    def reset_recording_loss_cycle(self):
        """Allow a later pair of active hand streams to trigger another prompt."""
        self.all_hands_were_active = False
        self.recording_loss_reported = False

    def start(self):
        try:
            self.module = base.importlib.import_module("connect2")
        except ImportError as exc:
            raise RuntimeError(
                "robot-hand dependencies are unavailable; run --check-dependencies"
            ) from exc

        self.module.reset_tactile_data()
        serial_ports = {
            "left": self.args.left_hand_port,
            "right": self.args.right_hand_port,
        }
        finger_ports = {
            "left": self.args.left_finger_port,
            "right": self.args.right_finger_port,
        }

        try:
            for side in self.sides:
                controller = self.module.SerialHandController(
                    serial_ports[side],
                    is_right=(side == "right"),
                    baudrate=self.args.hand_baud,
                )
                if self.record_hand_angles and not callable(
                    getattr(controller, "read_angles", None)
                ):
                    raise RuntimeError(
                        "recording requires SerialHandController.read_angles()"
                    )
                if not controller.connect():
                    raise RuntimeError(
                        f"{side} robot hand connection failed on {serial_ports[side]}"
                    )
                self.hands[side] = controller
                if not all(
                    callable(getattr(controller, method, None))
                    for method in (
                        "configure_force_control",
                        "start_force_control",
                        "stop_force_control",
                    )
                ):
                    raise RuntimeError(
                        "connect2.SerialHandController lacks native force-control APIs"
                    )
                command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                command_socket.setblocking(False)
                command_socket.bind(("0.0.0.0", 0))
                self.command_sockets[side] = command_socket
                self.command_targets[side] = (
                    self.args.unity_ip,
                    finger_ports[side],
                )

            self.tactile_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.tactile_socket.settimeout(0.5)
            self.tactile_socket.bind(("0.0.0.0", self.args.tactile_port))

            self.threads = [
                threading.Thread(
                    target=self._control_loop,
                    name="quest-hand-control",
                ),
                threading.Thread(
                    target=self._tactile_loop,
                    name="tactile-server",
                ),
            ]
            for thread in self.threads:
                thread.start()
        except Exception:
            self._close_unstarted_resources()
            raise

        side_names = ", ".join(side.upper() for side in self.sides)
        port_names = ", ".join(
            f"{side} UDP {finger_ports[side]}" for side in self.sides
        )
        print(
            f"[Check 2/3] Robot hand ready ({side_names}); Quest input {port_names}, "
            f"tactile UDP {self.args.tactile_port}."
        )

    def _close_unstarted_resources(self):
        base.close_socket(self.tactile_socket)
        self.tactile_socket = None
        for command_socket in self.command_sockets.values():
            base.close_socket(command_socket)
        self.command_sockets.clear()
        for side, hand in self.hands.items():
            self._open_hand(side, hand)
            hand.close()
        self.hands.clear()

    def _send_handshakes(self):
        for side, command_socket in self.command_sockets.items():
            try:
                command_socket.sendto(b"CONNECT", self.command_targets[side])
            except OSError:
                if not self.stop_event.is_set():
                    pass

    def _receive_latest(self, side):
        latest = None
        command_socket = self.command_sockets[side]
        expected_port = self.command_targets[side][1]
        while True:
            try:
                data, address = command_socket.recvfrom(128)
            except BlockingIOError:
                break
            except OSError:
                if not self.stop_event.is_set():
                    break
                return None
            if address[1] != expected_port:
                continue
            decoded = self.decode_packet(data, side)
            if decoded is not None:
                latest = decoded
        return latest

    def _control_loop(self):
        period = 1.0 / HAND_CONTROL_HZ
        next_handshake = 0.0
        try:
            while not self.stop_event.is_set():
                loop_start = time.monotonic()
                if loop_start >= next_handshake:
                    self._send_handshakes()
                    next_handshake = loop_start + HANDSHAKE_INTERVAL

                for side, controller in self.hands.items():
                    angles = self._receive_latest(side)
                    if angles is not None:
                        if self.teleop_enabled_event.is_set():
                            self._handle_quest_angles(
                                side, controller, angles, loop_start
                            )
                        else:
                            self._observe_quest_angles(side, angles, loop_start)
                    elif self._command_timed_out(side, loop_start):
                        self.command_active[side] = False
                        self._update_combined_tracking_state()
                        if self.args.open_on_tracking_loss:
                            if self._open_hand(side, controller):
                                print(
                                    f"[Hand] {side.upper()} Quest stream timed out; "
                                    "robot hand opened."
                                )
                        else:
                            print(
                                f"[Hand] {side.upper()} Quest stream timed out; "
                                "robot hand held."
                            )

                    self._poll_hand_feedback(side, controller)

                self._update_combined_tracking_state()

                if (
                    self.recording_loss_event is not None
                    and self.recording_loss_event.is_set()
                    and self.recording_loss_settled_event is not None
                ):
                    # Timeout/open-hand messages are complete before the main
                    # thread displays the recording decision prompt.
                    self.recording_loss_settled_event.set()

                remaining = period - (time.monotonic() - loop_start)
                if remaining > 0:
                    self.stop_event.wait(remaining)
        finally:
            for side, hand in self.hands.items():
                self._open_hand(side, hand)
            for hand in self.hands.values():
                hand.close()
            print("[Hand] Serial ports closed.")
            if self.recording_loss_settled_event is not None:
                self.recording_loss_settled_event.set()

    def _command_timed_out(self, side, now):
        last_packet = self.last_packet_time[side]
        return (
            self.command_active[side]
            and last_packet is not None
            and now - last_packet > self.args.hand_command_timeout
        )

    def _poll_hand_feedback(self, side, hand):
        if self.record_hand_angles:
            measured_angles = hand.read_angles()
            if measured_angles is not None:
                with self.state_lock:
                    self.latest_measured_angles[side] = list(measured_angles)
                    self.measured_angle_valid[side] = True
        hand.read_tactile()

    def _tactile_loop(self):
        print(f"[Tactile] Serving 10 floats on UDP {self.args.tactile_port}.")
        while not self.stop_event.is_set():
            try:
                data, address = self.tactile_socket.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            if data:
                payload = struct.pack("10f", *self.module.get_shared_data())
                try:
                    self.tactile_socket.sendto(payload, address)
                except OSError:
                    if not self.stop_event.is_set():
                        raise

    def snapshot(self, side):
        with self.state_lock:
            angles = list(self.latest_measured_angles[side])
        tactile = self.module.get_shared_data()
        offset = 5 if side == "right" else 0
        return angles, tactile[offset : offset + 5]

    def command_snapshot(self, side):
        """Return the latest Quest finger pose for ACT action recording."""
        with self.state_lock:
            return list(self.latest_quest_angles[side])

    def hand_state_snapshot(self, side):
        """Return the latest Quest hand state as a CSV-friendly yes/no value."""
        with self.state_lock:
            return ["yes" if self.latest_quest_hand_open[side] else "no"]

    def raw_tactile_snapshot(self, side):
        return self.module.get_shared_taxels(is_right=(side == "right"))

    def has_valid_tactile(self, side):
        return self.module.has_tactile_data(is_right=(side == "right"))

    def has_valid_angles(self, side):
        with self.state_lock:
            return self.measured_angle_valid[side]

    def stop(self):
        base.close_socket(self.tactile_socket)
        self.tactile_socket = None
        for command_socket in self.command_sockets.values():
            base.close_socket(command_socket)
        for thread in self.threads:
            thread.join(timeout=2.0)
        self.threads.clear()
        print("[Hand] Data sockets closed.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run ZED streaming, Quest 3 bare-hand control, selected serial robot "
            "hand(s) in native force-control mode, and selected Diana arm(s)."
        )
    )
    parser.add_argument("-l", "--left", action="store_true", help="use only the left side")
    parser.add_argument("-r", "--right", action="store_true", help="use only the right side")
    recording = parser.add_mutually_exclusive_group()
    recording.add_argument(
        "--record",
        nargs="?",
        const="",
        default="",
        metavar="HDF5_PATH",
        help=(
            "record synchronized data directly to HDF5 (default: automatic filename)"
        ),
    )
    recording.add_argument(
        "--no-record",
        dest="record",
        action="store_const",
        const=None,
        help="disable recording for this run",
    )
    parser.add_argument(
        "--convert-recording",
        metavar="CSV_PATH",
        help="convert an existing CSV plus synchronized frames to HDF5 and exit",
    )
    parser.add_argument(
        "--convert-output",
        metavar="HDF5_PATH",
        help="optional HDF5 output path for --convert-recording",
    )
    parser.add_argument(
        "--record-hz",
        type=float,
        default=DEFAULT_BARE_HAND_RECORD_HZ,
        help="HDF5 recording rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--force-torque-port",
        default=DEFAULT_RIGHT_FORCE_TORQUE_PORT,
        help="right M4313M1A sensor and recording port (default: COM4)",
    )
    parser.add_argument(
        "--left-force-torque-port",
        default=DEFAULT_LEFT_FORCE_TORQUE_PORT,
        help="left M4313M1A sensor and recording port (default: COM6)",
    )
    parser.add_argument(
        "--force-torque-baud",
        type=int,
        default=230400,
        help="M4313M1A baud rate (default: 230400)",
    )
    parser.add_argument(
        "--force-torque-timeout",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="maximum age of a force/torque sample (default: 0.5)",
    )
    for side in ("left", "right"):
        config = hand_driver.get_force_control_config(side == "right")
        parser.add_argument(
            f"--{side}-native-force-threshold",
            nargs=5,
            type=int,
            default=config["thresholds"],
            metavar=("THUMB", "INDEX", "MIDDLE", "RING", "PINKY"),
            help=f"{side} firmware force thresholds, each 20..1000",
        )
        parser.add_argument(
            f"--{side}-native-force-speed",
            nargs=6,
            type=int,
            default=config["speeds"],
            metavar=("TR", "TB", "INDEX", "MIDDLE", "RING", "PINKY"),
            help=f"{side} per-axis force-grasp speeds, each 20..200",
        )
        parser.add_argument(
            f"--{side}-native-force-trigger-angle",
            type=float,
            default=config["trigger_angle"],
            metavar="DEGREES",
            help=f"{side} mean finger bend that starts a native grasp",
        )
        parser.add_argument(
            f"--{side}-native-force-release-angle",
            type=float,
            default=config["release_angle"],
            metavar="DEGREES",
            help=f"{side} mean finger bend that releases a native grasp",
        )
    parser.add_argument("--camera-port", type=int, default=base.DEFAULT_CAMERA_PORT)
    parser.add_argument(
        "--jpeg-quality",
        type=base.jpeg_quality,
        default=90,
        metavar="1..100",
        help="JPEG quality for the TCP camera stream",
    )
    parser.add_argument(
        "--record-image-scale",
        type=float,
        default=DEFAULT_RECORD_IMAGE_SCALE,
        metavar="SCALE",
        help="recorded image scale after cropping (default: 0.5)",
    )
    parser.add_argument(
        "--record-image-quality",
        type=base.jpeg_quality,
        default=DEFAULT_RECORD_IMAGE_QUALITY,
        metavar="1..100",
        help="legacy JPEG quality for CSV conversion recordings (default: 80)",
    )
    parser.add_argument("--left-hand-port", default=base.DEFAULT_LEFT_HAND_PORT)
    parser.add_argument("--right-hand-port", default=base.DEFAULT_RIGHT_HAND_PORT)
    parser.add_argument("--hand-baud", type=int, default=base.DEFAULT_BAUD_RATE)
    parser.add_argument("--tactile-port", type=int, default=base.DEFAULT_TACTILE_PORT)
    parser.add_argument("--unity-ip", default="127.0.0.1")
    parser.add_argument("--left-arm-ip", default="192.168.11.60")
    parser.add_argument("--right-arm-ip", default="192.168.11.61")
    parser.add_argument("--left-pose-port", type=int, default=5005)
    parser.add_argument("--right-pose-port", type=int, default=5006)
    parser.add_argument("--left-finger-port", type=int, default=DEFAULT_LEFT_FINGER_PORT)
    parser.add_argument("--right-finger-port", type=int, default=DEFAULT_RIGHT_FINGER_PORT)
    parser.add_argument(
        "--hand-command-timeout",
        type=float,
        default=DEFAULT_HAND_COMMAND_TIMEOUT,
        metavar="SECONDS",
        help="stop accepting a hand command after this tracking gap (default: 0.35)",
    )
    tracking_loss = parser.add_mutually_exclusive_group()
    tracking_loss.add_argument(
        "--open-on-tracking-loss",
        dest="open_on_tracking_loss",
        action="store_true",
        default=True,
        help="open the robot hand when Quest packets time out (default)",
    )
    tracking_loss.add_argument(
        "--hold-on-tracking-loss",
        dest="open_on_tracking_loss",
        action="store_false",
        help="hold the last robot-hand position when Quest packets time out",
    )
    parser.add_argument(
        "--check-dependencies",
        action="store_true",
        help="report software/local SDK dependencies without opening hardware",
    )
    args = parser.parse_args(argv)
    if args.record_hz <= 0:
        parser.error("--record-hz must be greater than zero")
    if args.force_torque_baud <= 0:
        parser.error("--force-torque-baud must be greater than zero")
    if args.force_torque_timeout <= 0:
        parser.error("--force-torque-timeout must be greater than zero")
    if args.hand_command_timeout <= 0:
        parser.error("--hand-command-timeout must be greater than zero")
    if not 0.0 < args.record_image_scale <= 1.0:
        parser.error("--record-image-scale must be greater than zero and at most 1")
    for side in ("left", "right"):
        thresholds = getattr(args, f"{side}_native_force_threshold")
        speeds = getattr(args, f"{side}_native_force_speed")
        release_angle = getattr(args, f"{side}_native_force_release_angle")
        trigger_angle = getattr(args, f"{side}_native_force_trigger_angle")
        if any(not 20 <= value <= 1000 for value in thresholds):
            parser.error(
                f"--{side}-native-force-threshold values must be from 20 to 1000"
            )
        if len(speeds) != 6 or any(not 20 <= value <= 200 for value in speeds):
            parser.error(
                f"--{side}-native-force-speed must contain six values from 20 to 200"
            )
        if not 0.0 <= release_angle < trigger_angle <= 90.0:
            parser.error(
                f"{side} native force angles must satisfy "
                "0 <= release < trigger <= 90"
            )
    for name in (
        "camera_port",
        "tactile_port",
        "left_pose_port",
        "right_pose_port",
        "left_finger_port",
        "right_finger_port",
    ):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"--{name.replace('_', '-')} must be from 1 to 65535")
    return args


def selected_sides(args):
    if args.left and not args.right:
        return ["left"]
    if args.right and not args.left:
        return ["right"]
    return ["left", "right"]


class Hdf5EpisodeRecorder(base.CsvRecorder):
    """Append synchronized robot state and camera frames directly to HDF5."""

    SIDES = ("right", "left")
    FINGERS = ("thumb", "index", "middle", "ring", "pinky")

    def __init__(
        self,
        path,
        rate_hz,
        hand_subsystem,
        arms,
        diana_api,
        auxiliary_sensor,
        recording_started_event=None,
        control_hz=30.0,
    ):
        try:
            import h5py
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("direct recording requires h5py and numpy") from exc
        super().__init__(
            Path(path),
            rate_hz,
            hand_subsystem,
            arms,
            diana_api,
            include_raw_taxels=True,
            include_command_pose=True,
            auxiliary_sensor=auxiliary_sensor,
            recording_started_event=recording_started_event,
            include_hand_commands=True,
            hand_command_mode="binary",
            act_mode=True,
            include_joint_angular_velocity=True,
        )
        self.h5py = h5py
        self.np = np
        self.control_hz = float(control_hz)
        self.file_lock = threading.Lock()
        self.datasets = {}
        self.sample_count = 0
        self.image_count = 0
        self.pending_images = {}
        self.previous_hand_state = {side: None for side in self.SIDES}
        self.previous_timestamp = None

    @staticmethod
    def _hand_state(value):
        normalized = str(value).strip().lower()
        return 1.0 if normalized in {"yes", "true", "1", "open"} else 0.0

    def _create_state_dataset(self, name, tail_shape, dtype, chunk_rows=256):
        shape = (0,) + tuple(tail_shape)
        maxshape = (None,) + tuple(tail_shape)
        chunks = (chunk_rows,) + tuple(tail_shape)
        self.datasets[name] = self.file.create_dataset(
            name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype
        )

    def _start_hdf5(self, now):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.h5py.File(self.path, "w")
        self.file.attrs["sim"] = False
        self.file.attrs["control_hz"] = self.control_hz
        self.file.attrs["episode_length"] = 0
        self.file.attrs["qpos_dim"] = 16
        self.file.attrs["image_height"] = 0
        self.file.attrs["image_width"] = 0
        self.file.attrs["image_encoding"] = "rgb"
        observations = self.file.create_group("observations")
        images = observations.create_group("images")
        self._create_state_dataset("observations/qpos", (16,), "f4")
        self._create_state_dataset("observations/qvel", (16,), "f4")
        self._create_state_dataset("observations/force", (12,), "f4")
        self._create_state_dataset("observations/tactile", (2, 5, 16), "f4", 32)
        self._create_state_dataset("action", (16,), "f4")
        self._create_state_dataset("timestamp", (), "f8", 1024)
        self.image_group = images
        self.start_monotonic = now
        self.last_flush = now
        print(f"[Record] Valid data received; writing directly to {self.path}")

    def _append_state_locked(self, name, value):
        dataset = self.datasets[name]
        dataset.resize(self.sample_count + 1, axis=0)
        dataset[self.sample_count] = value

    def _sample_arrays(self, rows, elapsed):
        side_values = {side: values for side, values in rows}
        qpos = []
        qvel = []
        tactile = self.np.zeros((2, 5, 16), dtype=self.np.float32)
        dt = (
            max(elapsed - self.previous_timestamp, 1e-6)
            if self.previous_timestamp is not None
            else 1.0 / self.control_hz
        )
        for side_index, side in enumerate(self.SIDES):
            values = side_values.get(side)
            if values is None:
                tactile_values = [0.0] * 80
                hand_value = "no"
                joints = [0.0] * 7
                joint_velocities = [0.0] * 7
            else:
                # CsvRecorder._collect_rows emits tactile aggregates, raw
                # taxels, binary hand state, joints, and joint velocities.
                tactile_values = values[5:85]
                hand_value = values[85]
                joints = [float(value) for value in values[86:93]]
                joint_velocities = [float(value) for value in values[93:100]]
            hand = self._hand_state(hand_value)
            qpos.extend(joints + [hand])
            previous_hand = self.previous_hand_state[side]
            hand_velocity = 0.0 if previous_hand is None else (hand - previous_hand) / dt
            qvel.extend(joint_velocities + [hand_velocity])
            self.previous_hand_state[side] = hand
            tactile[side_index] = self.np.asarray(
                tactile_values, dtype=self.np.float32
            ).reshape(5, 16)
        force = list(self._last_auxiliary_values[:12])
        force.extend([0.0] * (12 - len(force)))
        self.previous_timestamp = elapsed
        return (
            self.np.asarray(qpos, dtype=self.np.float32),
            self.np.asarray(qvel, dtype=self.np.float32),
            self.np.asarray(force, dtype=self.np.float32),
            tactile,
        )

    def sample_if_due(self, now):
        if now < self.next_sample:
            return False
        while self.next_sample <= now:
            self.next_sample += self.period
        rows = self._collect_rows(now)
        if rows is None:
            return False
        with self.file_lock:
            if self.file is None:
                self._start_hdf5(now)
            elapsed = now - self.start_monotonic
            qpos, qvel, force, tactile = self._sample_arrays(rows, elapsed)
            self._append_state_locked("observations/qpos", qpos)
            self._append_state_locked("observations/qvel", qvel)
            self._append_state_locked("observations/force", force)
            self._append_state_locked("observations/tactile", tactile)
            self._append_state_locked("action", qpos)
            self._append_state_locked("timestamp", elapsed)
            self.sample_count += 1
            self.file.attrs.modify("episode_length", self.sample_count)
            if now - self.last_flush >= 1.0:
                self.file.flush()
                self.last_flush = now
        if self.recording_started_event is not None:
            self.recording_started_event.set()
        self.last_sample_time = now
        return True

    def _create_image_datasets_locked(self, left, right):
        if left.shape != right.shape:
            raise ValueError("left and right overview images must have equal shapes")
        height, width, channels = left.shape
        if channels != 3:
            raise ValueError("overview images must have three RGB channels")
        chunks = (1, height, width, channels)
        for name in ("overview_left", "overview_right"):
            self.datasets[f"observations/images/{name}"] = self.image_group.create_dataset(
                name,
                shape=(0, height, width, channels),
                maxshape=(None, height, width, channels),
                chunks=chunks,
                dtype="u1",
            )
        self.file.attrs.modify("image_height", height)
        self.file.attrs.modify("image_width", width)

    def _write_ready_images_locked(self):
        while self.image_count + 1 in self.pending_images:
            frame_number = self.image_count + 1
            left, right = self.pending_images.pop(frame_number)
            if "observations/images/overview_left" not in self.datasets:
                self._create_image_datasets_locked(left, right)
            for name, image in (("overview_left", left), ("overview_right", right)):
                dataset = self.datasets[f"observations/images/{name}"]
                if image.shape != dataset.shape[1:]:
                    raise ValueError("recorded camera image dimensions changed")
                dataset.resize(self.image_count + 1, axis=0)
                dataset[self.image_count] = image
            self.image_count += 1

    def append_image(self, frame_number, bgr_image):
        image = self.np.asarray(bgr_image)
        midpoint = image.shape[1] // 2
        if image.ndim != 3 or image.shape[2] != 3 or midpoint <= 0:
            raise ValueError("camera frame must be an HxWx3 side-by-side image")
        if image.shape[1] - midpoint != midpoint:
            raise ValueError("side-by-side camera frame width must be even")
        rgb = image[:, :, ::-1].copy()
        with self.file_lock:
            if self.file is None:
                raise RuntimeError("received an image before the HDF5 episode started")
            if frame_number > self.sample_count:
                raise RuntimeError("received an image without a matching state sample")
            self.pending_images[int(frame_number)] = (
                rgb[:, :midpoint].copy(),
                rgb[:, midpoint:].copy(),
            )
            self._write_ready_images_locked()

    @property
    def has_data(self):
        with self.file_lock:
            return self.file is not None and self.sample_count > 0

    def close(self):
        error = None
        with self.file_lock:
            if self.file is None:
                print("[Record] No file created because valid data was not received.")
                return
            self._write_ready_images_locked()
            if self.sample_count != self.image_count:
                error = RuntimeError(
                    f"HDF5 state/image mismatch: {self.sample_count} state sample(s), "
                    f"{self.image_count} image(s)"
                )
            self.file.attrs.modify("episode_length", self.sample_count)
            self.file.flush()
            self.file.close()
            self.file = None
        print(f"[Record] Closed {self.path} ({self.sample_count} samples)")
        if error is not None:
            raise error


def recording_path(value):
    if value:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else (Path.cwd() / path).resolve()
        return path if path.suffix.lower() == ".hdf5" else path.with_suffix(".hdf5")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRIPT_DIR / "recordings" / f"bare_hand_teleop_{timestamp}.hdf5"


def session_recording_path(record_value, session_number):
    path = recording_path(record_value)
    if session_number <= 0:
        return path
    return path.with_name(f"{path.stem}_session{session_number}{path.suffix}")


def frame_recording_directory(csv_path):
    """Return the per-episode image directory beside its CSV file."""
    csv_path = Path(csv_path)
    return csv_path.parent / f"{csv_path.stem}_frames"


def hdf5_recording_path(csv_path):
    """Return the final HDF5 path corresponding to a CSV recording."""
    return Path(csv_path).with_suffix(".hdf5")


def _csv_float(row, name, default=0.0):
    value = row.get(name, "")
    if value in (None, ""):
        return float(default)
    return float(value)


def _csv_hand_state(row, name):
    value = str(row.get(name, "0")).strip().lower()
    if value in {"yes", "true", "1", "open"}:
        return 1.0
    if value in {"no", "false", "0", "closed"}:
        return 0.0
    return float(value)


def convert_recording_to_hdf5(csv_path, frame_directory=None, output_path=None,
                              control_hz=DEFAULT_BARE_HAND_RECORD_HZ):
    """Convert one synchronized CSV/JPEG recording into the episode HDF5 schema."""
    import csv

    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("HDF5 conversion requires h5py and numpy") from exc

    csv_path = Path(csv_path)
    frame_directory = (
        Path(frame_directory)
        if frame_directory is not None
        else frame_recording_directory(csv_path)
    )
    output_path = Path(output_path) if output_path is not None else hdf5_recording_path(csv_path)
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"recording has no CSV header: {csv_path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"recording contains no samples: {csv_path}")

    sides = ("right", "left")
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    timestamps = np.asarray([_csv_float(row, "elapsed_s") for row in rows], dtype=np.float64)
    qpos = np.empty((len(rows), 16), dtype=np.float32)
    action = np.empty((len(rows), 16), dtype=np.float32)
    qvel = np.empty((len(rows), 16), dtype=np.float32)
    force = np.empty((len(rows), 12), dtype=np.float32)
    tactile = np.empty((len(rows), 2, 5, 16), dtype=np.float32)

    for index, row in enumerate(rows):
        qpos_values = []
        action_values = []
        velocity_values = []
        force_values = []
        for side in sides:
            qpos_values.extend(_csv_float(row, f"{side}_arm_joint_{joint}_rad") for joint in range(1, 8))
            hand = _csv_hand_state(row, f"{side}_hand_open")
            qpos_values.append(hand)
            action_values.extend(_csv_float(row, f"{side}_arm_joint_{joint}_rad") for joint in range(1, 8))
            action_values.append(hand)
            velocity_values.extend(
                _csv_float(row, f"{side}_arm_joint_{joint}_angular_vel_rad_s")
                for joint in range(1, 8)
            )
            velocity_values.append(0.0)  # filled from hand-state differences below
            force_values.extend(
                _csv_float(row, f"{'' if side == 'right' else 'left_'}{name}")
                for name in (
                    "force_fx_n", "force_fy_n", "force_fz_n",
                    "torque_mx_nm", "torque_my_nm", "torque_mz_nm",
                )
            )
            side_index = 0 if side == "right" else 1
            for finger_index, finger in enumerate(fingers):
                for taxel_index in range(1, 17):
                    tactile[index, side_index, finger_index, taxel_index - 1] = _csv_float(
                        row, f"{side}_tactile_{finger}_taxel_{taxel_index:02d}_raw"
                    )
        qpos[index] = qpos_values
        action[index] = action_values
        qvel[index] = velocity_values
        force[index] = force_values

    if len(rows) > 1:
        dt = np.diff(timestamps, prepend=timestamps[0])
        dt[0] = 1.0 / float(control_hz)
        dt = np.maximum(dt, 1e-6)
        for side_index, qpos_index in enumerate((7, 15)):
            qvel[:, qpos_index] = np.gradient(qpos[:, qpos_index], timestamps, edge_order=1)

    frame_paths = sorted(frame_directory.glob("frame_*.jpg"))
    if len(frame_paths) != len(rows):
        raise ValueError(
            f"CSV/image count mismatch: {len(rows)} samples, {len(frame_paths)} frames"
        )
    import cv2
    images_left = []
    images_right = []
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read image: {frame_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        midpoint = image.shape[1] // 2
        if midpoint <= 0 or image.shape[1] - midpoint != midpoint:
            raise ValueError(f"side-by-side image width must be even: {frame_path}")
        images_left.append(image[:, :midpoint])
        images_right.append(image[:, midpoint:])
    overview_left = np.stack(images_left, axis=0).astype(np.uint8, copy=False)
    overview_right = np.stack(images_right, axis=0).astype(np.uint8, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as episode:
        episode.attrs["sim"] = False
        episode.attrs["control_hz"] = float(control_hz)
        episode.attrs["episode_length"] = int(len(rows))
        episode.attrs["qpos_dim"] = 16
        episode.attrs["image_height"] = int(overview_left.shape[1])
        episode.attrs["image_width"] = int(overview_left.shape[2])
        episode.attrs["image_encoding"] = "rgb"
        observations = episode.create_group("observations")
        observations.create_dataset("qpos", data=qpos)
        observations.create_dataset("qvel", data=qvel)
        observations.create_dataset("force", data=force)
        observations.create_dataset("tactile", data=tactile)
        images = observations.create_group("images")
        images.create_dataset("overview_left", data=overview_left, dtype=np.uint8)
        images.create_dataset("overview_right", data=overview_right, dtype=np.uint8)
        episode.create_dataset("action", data=action)
        episode.create_dataset("timestamp", data=timestamps)
    return output_path


def prompt_recording_decision(input_fn=input):
    while True:
        try:
            choice = input_fn(
                "\n[Record] Save or discard this recording? [s]ave/[d]iscard: "
            )
        except (EOFError, KeyboardInterrupt):
            print("\n[Record] No console input available; saving recording.")
            return "save"
        normalized = choice.strip().lower()
        if normalized in ("s", "save"):
            return "save"
        if normalized in ("d", "discard"):
            return "discard"
        print("[Record] Enter 's' to save or 'd' to discard.")


def discard_recording(paths):
    for path in paths:
        if path is None:
            continue
        try:
            path = Path(path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            print(f"[Record] Discarded {path}")
        except OSError as exc:
            print(f"[Record] Could not discard {path}: {exc}", file=sys.stderr)


def main(argv=None):
    args = parse_args(argv)
    if args.convert_recording:
        try:
            output = convert_recording_to_hdf5(
                args.convert_recording,
                output_path=args.convert_output,
                control_hz=30.0,
            )
        except Exception as exc:
            print(f"[Record] HDF5 conversion failed: {exc}", file=sys.stderr)
            return 1
        print(f"[Record] Converted episode to {output}")
        return 0
    if args.check_dependencies:
        failures = base.check_dependencies()
        try:
            import h5py
            print(f"[OK] HDF5 writer: {h5py.__version__}")
        except Exception as exc:
            failures += 1
            print(f"[MISSING] HDF5 writer (h5py): {exc}")
        return 1 if failures else 0

    sides = selected_sides(args)
    recording_session = 0
    output_path = (
        session_recording_path(args.record, recording_session)
        if args.record is not None
        else None
    )
    if output_path is not None:
        try:
            import h5py  # noqa: F401
        except ImportError as exc:
            print(
                f"[Main] Direct HDF5 recording requires h5py: {exc}",
                file=sys.stderr,
            )
            return 1
    stop_event = threading.Event()
    recording_started_event = (
        threading.Event() if output_path is not None else None
    )
    recording_loss_event = (
        threading.Event() if output_path is not None else None
    )
    recording_loss_settled_event = (
        threading.Event() if output_path is not None else None
    )
    teleop_enabled_event = threading.Event()
    teleop_enabled_event.set()
    frame_synchronizer = (
        RecordingFrameSynchronizer() if output_path is not None else None
    )
    camera = Zed720CameraStreamer(
        args.camera_port,
        stop_event,
        args.jpeg_quality,
        frame_start_event=recording_started_event,
        frame_synchronizer=frame_synchronizer,
        record_image_scale=args.record_image_scale,
        record_image_quality=args.record_image_quality,
    )
    hands = QuestHandSubsystem(
        args,
        sides,
        stop_event,
        record_hand_angles=(output_path is not None),
        recording_loss_event=recording_loss_event,
        teleop_enabled_event=teleop_enabled_event,
        recording_loss_settled_event=recording_loss_settled_event,
    )
    arm_module = None
    arms = {}
    recorder = None
    force_torque = None

    try:
        if output_path is not None:
            force_torque = DualM4313M1AReader(
                args.force_torque_port,
                args.left_force_torque_port,
                args.force_torque_baud,
                stop_event,
                stale_after=args.force_torque_timeout,
            )
            force_torque.start()

        camera.start()
        hands.start()
        arm_module, arms = base.start_arms(args, sides)
        for arm in arms.values():
            arm.relative_rotation_mapping = True
        if output_path is not None:
            recorder = Hdf5EpisodeRecorder(
                output_path,
                args.record_hz,
                hands,
                arms,
                arm_module.DianaApi,
                force_torque,
                recording_started_event=recording_started_event,
                control_hz=30.0,
            )
            camera.start_recording(
                frame_start_event=recording_started_event,
                frame_sink=recorder,
            )

        target_period = 1.0 / arm_module.CONTROL_HZ
        next_tick = time.monotonic()
        last_handshake = next_tick
        print(
            f"[Run] Bare-hand teleoperation active at "
            f"{arm_module.CONTROL_HZ:.0f} Hz. Press Ctrl+C to stop."
        )
        while not stop_event.is_set():
            if recording_loss_event is not None and recording_loss_event.is_set():
                if recording_loss_settled_event is not None:
                    recording_loss_settled_event.wait(timeout=2.0)
                for arm in arms.values():
                    arm.stop_motion("both Quest hand streams lost")

                hdf5_was_created = recorder is not None and recorder.has_data
                camera.finish_recording()
                if recorder is not None:
                    recorder.close()
                recorder = None

                decision = prompt_recording_decision()
                if decision == "discard":
                    discard_recording([output_path if hdf5_was_created else None])
                else:
                    if hdf5_was_created:
                        print(f"[Record] Saved {output_path}")
                    else:
                        print("[Record] No files were created before tracking loss.")

                recording_loss_event.clear()
                if recording_loss_settled_event is not None:
                    recording_loss_settled_event.clear()
                hands.reset_recording_loss_cycle()
                if output_path is not None:
                    recording_session += 1
                    output_path = session_recording_path(
                        args.record, recording_session
                    )
                    recording_started_event.clear()
                    recorder = Hdf5EpisodeRecorder(
                        output_path,
                        args.record_hz,
                        hands,
                        arms,
                        arm_module.DianaApi,
                        force_torque,
                        recording_started_event=recording_started_event,
                        control_hz=30.0,
                    )
                    camera.start_recording(
                        frame_start_event=recording_started_event,
                        frame_sink=recorder,
                    )
                teleop_enabled_event.set()
                next_tick = time.monotonic()
                print(
                    "[Run] Teleoperation re-enabled; "
                    "a new recording session is armed."
                )

            if teleop_enabled_event.is_set():
                for arm in arms.values():
                    arm.update()

            now = time.monotonic()
            if recorder is not None and recorder.sample_if_due(now):
                frame_synchronizer.request_frame(recorder.last_sample_time)
            if now - last_handshake > 2.0:
                for arm in arms.values():
                    arm.send_handshake()
                last_handshake = now

            next_tick += target_period
            sleep_duration = next_tick - time.monotonic()
            if sleep_duration > 0:
                stop_event.wait(sleep_duration)
            elif -sleep_duration > target_period:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received; shutting down.")
    except Exception as exc:
        print(f"[Main] Startup/runtime failure: {exc}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        final_hdf5_was_created = recorder is not None and recorder.has_data
        if output_path is not None:
            try:
                camera.finish_recording()
            except Exception as exc:
                print(f"[Record] Camera recording finalization failed: {exc}", file=sys.stderr)
                return_code = 1
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                print(f"[Record] HDF5 finalization failed: {exc}", file=sys.stderr)
                return_code = 1
            recorder = None

        stop_event.set()
        for arm in arms.values():
            arm.cleanup()
        if force_torque is not None:
            force_torque.stop()
        hands.stop()
        camera.stop()

        if camera.frame_writer_error is not None:
            return_code = 1
        if output_path is not None and final_hdf5_was_created:
            decision = prompt_recording_decision()
            if decision == "discard":
                discard_recording([output_path])
            else:
                print(f"[Record] Saved {output_path}")

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
