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
"""
from __future__ import annotations

import time
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fusion.kalman_fusion import MultiUAVKalmanFusion, FusionTrackHistory
from network.ground_receiver import GroundReceiver

import queue
from fusion.coordinate_utils import camera_to_world

ARENA_SIZE_M = 6.0
LOG_MAX_LINES = 12

# Cấu hình vị trí và góc nhìn thực tế của 3 UAV trong không gian 6x6 m
# (X_uav, Y_uav, Z_uav), yaw_deg, pitch_deg
UAV_POSES = {
    1: {"pos": (0.5, 0.5, 2.5), "yaw": 45.0, "pitch": 45.0},
    2: {"pos": (5.5, 0.5, 2.5), "yaw": 135.0, "pitch": 45.0},
    3: {"pos": (3.0, 5.5, 2.8), "yaw": 270.0, "pitch": 50.0},
}


class Dashboard:
    def __init__(self, receiver: GroundReceiver, fusion: MultiUAVKalmanFusion):
        self.receiver = receiver
        self.fusion = fusion
        self.history = FusionTrackHistory()
        self.log_lines: deque[str] = deque(maxlen=LOG_MAX_LINES)
        self.uav_last_seen: dict[int, float] = {}
        self._t0 = time.time()
        self.last_msg_time: float | None = None

        self.fig, (self.ax_status, self.ax_log, self.ax_traj) = plt.subplots(
            1, 3, figsize=(15, 5)
        )
        self.fig.suptitle("Multi-UAV Task-Oriented Semantic Communication — Demo PoC")

    def _drain_queue_and_update(self):
        # 1. Gom tất cả message có sẵn trong hàng đợi nhận từ receiver
        msgs = []
        while True:
            try:
                msgs.append(self.receiver.message_queue.get_nowait())
            except queue.Empty:
                break

        if not msgs:
            return

        # 2. Sắp xếp các gói tin theo thứ tự thời gian sinh ra (timestamp_ms)
        # để đảm bảo tính tuần tự chính xác, tránh tác động của trễ mạng UDP
        msgs.sort(key=lambda m: m.timestamp_ms)

        for msg in msgs:
            self.uav_last_seen[msg.uav_id] = time.time()

            # 3. Áp dụng chuyển đổi toạ độ từ camera sang toạ độ sân đấu 6x6m
            pose = UAV_POSES.get(msg.uav_id, {"pos": (3.0, 3.0, 3.0), "yaw": 0.0, "pitch": 90.0})
            
            # Sử dụng bbox center (cx, cy)
            cx, cy = msg.bbox[0], msg.bbox[1]
            world_xy = camera_to_world(
                cx=cx,
                cy=cy,
                uav_pos=pose["pos"],
                yaw_deg=pose["yaw"],
                pitch_deg=pose["pitch"],
                arena_size_m=ARENA_SIZE_M
            )

            # 4. Dự đoán trạng thái Kalman Filter với khoảng thời gian dt thực tế
            current_t = msg.timestamp_ms / 1000.0
            if self.last_msg_time is None:
                self.last_msg_time = current_t
                # Bước đầu tiên: chỉ khởi tạo bằng update, chưa predict
                fused_pos = self.fusion.update(world_xy, confidence=0.8)
            else:
                dt = current_t - self.last_msg_time
                if dt > 0:
                    self.fusion.predict(dt=dt)
                    self.last_msg_time = current_t
                # Nếu dt <= 0, ta chỉ cập nhật (update) thêm quan sát mà không dịch chuyển thời gian
                fused_pos = self.fusion.update(world_xy, confidence=0.8)

            self.history.append(time.time() - self._t0, fused_pos)

            self.log_lines.append(
                f"[{time.time() - self._t0:6.2f}s] UAV {msg.uav_id} "
                f"seq={msg.seq} world=({world_xy[0]:.2f},{world_xy[1]:.2f})"
            )

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
            self.ax_status.text(0.1, 0.8 - i * 0.25, f"UAV {uav_id}: {label}",
                                 color=color, fontsize=13, fontweight="bold")

        bw = self.receiver.get_bandwidth_kbps()
        y = 0.05
        self.ax_status.text(0.1, y, "Băng thông:", fontsize=11)
        for uav_id, kbps in sorted(bw.items()):
            y -= 0.05
            self.ax_status.text(0.15, y, f"UAV {uav_id}: {kbps:.3f} kbps", fontsize=10)

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

        _, xs, ys = self.history.as_arrays()
        if xs.size > 0:
            self.ax_traj.plot(xs, ys, "-", color="tab:blue", alpha=0.6, label="Quỹ đạo")
            self.ax_traj.plot(xs[-1], ys[-1], "o", color="tab:red", markersize=8, label="Vị trí hiện tại")
        self.ax_traj.legend(loc="upper right", fontsize=8)

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
    fusion = MultiUAVKalmanFusion()
    dashboard = Dashboard(receiver, fusion)
    dashboard.run()
