"""
dashboard.py
------------
Dashboard demo real-time cho PoC, đúng 3 màn hình đã thiết kế:
  1) Trạng thái kết nối của 3 Tello
  2) Log message ngữ nghĩa (bandwidth) thay vì video stream
  3) Quỹ đạo mục tiêu sau fusion trong không gian 6x6 m

Dùng matplotlib animation cho đơn giản, dễ chạy trên laptop khi demo.
Có thể thay bằng dashboard web (Streamlit/Dash) nếu cần giao diện đẹp hơn
cho báo cáo — đây là bản khung tối thiểu để kiểm chứng luồng dữ liệu.

Cập nhật: Dashboard đã được nâng cấp để dùng MultiTargetTracker thay vì
MultiUAVKalmanFusion đơn lẻ, hỗ trợ theo dõi nhiều mục tiêu đồng thời.
"""
from __future__ import annotations

import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fusion.kalman_fusion import (
    KalmanFusionConfig,
    MultiTargetTracker,
    FusionTrackHistory,
)
from network.ground_receiver import GroundReceiver

import queue
from fusion.coordinate_utils import camera_to_world

ARENA_SIZE_M = 6.0
LOG_MAX_LINES = 12

TRACK_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

UAV_POSES = {
    1: {"pos": (0.5, 0.5, 2.5), "yaw": 45.0,  "pitch": 45.0},
    2: {"pos": (5.5, 0.5, 2.5), "yaw": 135.0, "pitch": 45.0},
    3: {"pos": (3.0, 5.5, 2.8), "yaw": 270.0, "pitch": 50.0},
}

class Dashboard:
    def __init__(self, receiver: GroundReceiver, tracker: MultiTargetTracker):
        self.receiver = receiver
        self.tracker  = tracker
        
        self.log_lines: deque[str] = deque(maxlen=LOG_MAX_LINES)
        self.uav_last_seen: dict[int, float] = {}
        self._t0 = time.time()
        self.last_msg_time: float | None = None

        self.fig, (self.ax_status, self.ax_log, self.ax_traj) = plt.subplots(
            1, 3, figsize=(15, 5)
        )
        self.fig.suptitle("Multi-UAV Task-Oriented Semantic Communication — Demo PoC")

    def _drain_queue_and_update(self):
        
        msgs = []
        while True:
            try:
                msgs.append(self.receiver.message_queue.get_nowait())
            except queue.Empty:
                break

        if not msgs:
            
            self.tracker.predict_only()
            return

        msgs.sort(key=lambda m: m.timestamp_ms)

        observations = []
        for msg in msgs:
            self.uav_last_seen[msg.uav_id] = time.time()

            pose = UAV_POSES.get(
                msg.uav_id,
                {"pos": (3.0, 3.0, 3.0), "yaw": 0.0, "pitch": 90.0},
            )
            cx, cy = float(msg.bbox[0]), float(msg.bbox[1])

            observations.append({
                "bbox_cx_cy": (cx, cy),
                "uav_pose":   (pose["pos"], pose["yaw"], pose["pitch"]),
                "embedding":  msg.embedding,
                "confidence": 0.8,
            })

            world_xy = camera_to_world(
                cx=cx, cy=cy,
                uav_pos=pose["pos"],
                yaw_deg=pose["yaw"],
                pitch_deg=pose["pitch"],
                arena_size_m=ARENA_SIZE_M,
            )
            self.log_lines.append(
                f"[{time.time() - self._t0:6.2f}s] UAV {msg.uav_id} "
                f"seq={msg.seq} world=({world_xy[0]:.2f},{world_xy[1]:.2f})"
            )

        current_t = msgs[-1].timestamp_ms / 1000.0
        dt: float | None = None
        if self.last_msg_time is not None and current_t > self.last_msg_time:
            dt = current_t - self.last_msg_time
        self.last_msg_time = current_t

        self.tracker.step(observations, dt=dt)

    def _draw_status_panel(self):
        self.ax_status.clear()
        self.ax_status.set_title("1. Trạng thái kết nối Tello")
        self.ax_status.axis("off")

        now = time.time()
        for i, uav_id in enumerate([1, 2, 3]):
            last_seen = self.uav_last_seen.get(uav_id)
            connected = last_seen is not None and (now - last_seen) < 2.0
            color = "green" if connected else "red"
            label = "Kết nối" if connected else "Mất kết nối"
            self.ax_status.text(
                0.1, 0.8 - i * 0.25, f"UAV {uav_id}: {label}",
                color=color, fontsize=13, fontweight="bold"
            )

        n_confirmed = sum(
            1 for tid in self.tracker._filters
            if self.tracker._states.get(tid) == "confirmed"
        )
        self.ax_status.text(
            0.1, 0.05,
            f"Track đang bám: {n_confirmed}",
            fontsize=11, color="navy",
        )

        bw = self.receiver.get_bandwidth_kbps()
        y = -0.05
        self.ax_status.text(0.1, y + 0.1, "Băng thông:", fontsize=11)
        for uav_id, kbps in sorted(bw.items()):
            y -= 0.05
            self.ax_status.text(
                0.15, y + 0.1, f"UAV {uav_id}: {kbps:.3f} kbps", fontsize=10
            )

    def _draw_log_panel(self):
        self.ax_log.clear()
        self.ax_log.set_title("2. Log message ngữ nghĩa (thay vì video)")
        self.ax_log.axis("off")
        text = "\n".join(self.log_lines) or "(chưa có message)"
        self.ax_log.text(0.02, 0.98, text, fontsize=8, family="monospace",
                          verticalalignment="top")

    def _draw_trajectory_panel(self):
        self.ax_traj.clear()
        self.ax_traj.set_title("3. Quỹ đạo mục tiêu (fusion, 6x6 m)")
        self.ax_traj.set_xlim(0, ARENA_SIZE_M)
        self.ax_traj.set_ylim(0, ARENA_SIZE_M)
        self.ax_traj.set_aspect("equal")
        self.ax_traj.grid(True, linestyle="--", alpha=0.3)

        for i, (tid, hist) in enumerate(self.tracker.histories.items()):
            color = TRACK_COLORS[i % len(TRACK_COLORS)]
            _, xs, ys = hist.as_arrays()
            if xs.size > 0:
                state = self.tracker._states.get(tid, "tentative")
                alpha = 0.8 if state == "confirmed" else 0.3
                self.ax_traj.plot(
                    xs, ys, "-", color=color, alpha=alpha,
                    label=f"Track {tid} ({state[:4]})"
                )
                self.ax_traj.plot(
                    xs[-1], ys[-1], "o", color=color, markersize=8
                )

        if self.tracker.histories:
            self.ax_traj.legend(loc="upper right", fontsize=7)

    def update(self, _frame):
        self._drain_queue_and_update()
        self._draw_status_panel()
        self._draw_log_panel()
        self._draw_trajectory_panel()

    def run(self, interval_ms: int = 200):
        self.receiver.start()
        ani = animation.FuncAnimation(self.fig, self.update, interval=interval_ms)
        plt.tight_layout()
        plt.show()
        return ani

if __name__ == "__main__":
    receiver = GroundReceiver(listen_port=9000)
    tracker = MultiTargetTracker(
        config=KalmanFusionConfig(dt=0.1),
        association_threshold=0.4,
        max_missed_frames=10,
        min_hits=3,
    )
    dashboard = Dashboard(receiver, tracker)
    dashboard.run()

