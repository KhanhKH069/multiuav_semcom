"""
kalman_fusion.py
-----------------
Phương án fusion CHÍNH của hệ thống: Kalman Filter (constant-velocity model)
hợp nhất các quan sát vị trí (cx, cy) từ nhiều UAV thành một ước lượng
quỹ đạo mục tiêu thống nhất.

Mô hình trạng thái: x = [px, py, vx, vy]^T  (vị trí + vận tốc trong mặt phẳng
không gian thử nghiệm 6x6 m, đã quy đổi từ toạ độ ảnh chuẩn hoá + độ cao/góc
nhìn từng UAV — phần quy đổi 2D ảnh -> toạ độ thế giới 3D/2D nên implement
riêng trong `coordinate_utils.py`, ở đây giả định input đã ở hệ toạ độ chung).

Mỗi UAV đóng vai trò một "sensor" độc lập; độ tin cậy (confidence) của từng
quan sát được dùng làm trọng số nghịch đảo cho ma trận R (measurement noise).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KalmanFusionConfig:
    dt: float = 0.1                    # chu kỳ cập nhật (s), khớp với send_hz phía UAV
    process_noise_std: float = 0.5     # độ "tin" vào mô hình chuyển động (m/s^2 tương đương)
    base_measurement_noise_std: float = 0.3  # độ nhiễu quan sát cơ bản (m), scale theo confidence


class MultiUAVKalmanFusion:
    """Kalman Filter 2D constant-velocity, nhận quan sát tuần tự từ nhiều UAV."""

    def __init__(self, config: KalmanFusionConfig = KalmanFusionConfig()):
        self.cfg = config

        # Trạng thái [px, py, vx, vy]
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0  # uncertainty ban đầu lớn
        self.initialized = False

        dt = self.cfg.dt
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        q = self.cfg.process_noise_std ** 2
        self.Q = q * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2],
        ], dtype=np.float64)

        # Ta chỉ quan sát vị trí (px, py), không quan sát trực tiếp vận tốc
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

    def predict(self, dt: float | None = None):
        if not self.initialized:
            return

        if dt is not None:
            # Recompute transition matrix F and process noise Q dynamically for dt
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float64)

            q = self.cfg.process_noise_std ** 2
            Q = q * np.array([
                [dt**4/4, 0, dt**3/2, 0],
                [0, dt**4/4, 0, dt**3/2],
                [dt**3/2, 0, dt**2, 0],
                [0, dt**3/2, 0, dt**2],
            ], dtype=np.float64)
        else:
            F = self.F
            Q = self.Q

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, position_xy: np.ndarray, confidence: float = 1.0, embedding: np.ndarray | None = None, embedding_threshold: float = 0.7):
        """
        position_xy: array shape (2,) — quan sát (px, py) từ một UAV, đã quy đổi
                     về hệ toạ độ chung của không gian thử nghiệm 6x6 m.
        confidence:  trong (0, 1], độ tin cậy của quan sát (ví dụ dựa trên độ
                     rõ nét bbox, khoảng cách UAV-mục tiêu...). Confidence thấp
                     -> R lớn hơn -> quan sát này ảnh hưởng ít hơn tới ước lượng.
        embedding:   vector đặc trưng ngoại hình (128-dim) của mục tiêu.
        embedding_threshold: ngưỡng Cosine Similarity để chấp nhận liên kết dữ liệu.
        """
        # --- 1. Data Association (Re-identification) ---
        if embedding is not None:
            if not hasattr(self, 'target_embedding') or self.target_embedding is None:
                self.target_embedding = embedding.copy()
            else:
                # Tính Cosine Similarity
                sim = np.dot(self.target_embedding, embedding) / (
                    np.linalg.norm(self.target_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                if sim < embedding_threshold:
                    # Tương đồng quá thấp -> coi là nhiễu hoặc vật thể khác, bỏ qua cập nhật
                    return self.get_position()
                else:
                    # Cập nhật dần embedding của mục tiêu (moving average) để thích ứng với thay đổi ngoại hình
                    self.target_embedding = 0.9 * self.target_embedding + 0.1 * embedding
                    self.target_embedding /= np.linalg.norm(self.target_embedding)

        # --- 2. Kalman Update ---
        z = position_xy.reshape(2, 1)

        if not self.initialized:
            self.x[0, 0], self.x[1, 0] = z[0, 0], z[1, 0]
            self.initialized = True
            return self.get_position()

        noise_std = self.cfg.base_measurement_noise_std / max(confidence, 1e-3)
        R = np.eye(2) * (noise_std ** 2)

        y = z - self.H @ self.x                          # innovation
        S = self.H @ self.P @ self.H.T + R                # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.get_position()

    def get_position(self) -> np.ndarray:
        return self.x[:2, 0].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[2:, 0].copy()


@dataclass
class FusionTrackHistory:
    """Tiện ích lưu lại quỹ đạo để vẽ lên dashboard."""
    positions: list = field(default_factory=list)  # list of (t, px, py)

    def append(self, t: float, position_xy: np.ndarray):
        self.positions.append((t, float(position_xy[0]), float(position_xy[1])))

    def as_arrays(self):
        arr = np.array(self.positions)
        if arr.size == 0:
            return np.array([]), np.array([]), np.array([])
        return arr[:, 0], arr[:, 1], arr[:, 2]


if __name__ == "__main__":
    # Demo nhanh: 1 mục tiêu di chuyển thẳng, 3 UAV quan sát có nhiễu khác nhau
    rng = np.random.default_rng(0)
    fusion = MultiUAVKalmanFusion()
    history = FusionTrackHistory()

    true_pos = np.array([0.5, 0.5])
    velocity = np.array([0.2, 0.1])  # m / step

    for step in range(50):
        fusion.predict()
        true_pos = true_pos + velocity * fusion.cfg.dt

        # 3 UAV quan sát với mức nhiễu và confidence khác nhau
        for uav_conf, noise_scale in [(0.9, 0.05), (0.7, 0.12), (0.5, 0.2)]:
            noisy_obs = true_pos + rng.normal(0, noise_scale, size=2)
            fusion.update(noisy_obs, confidence=uav_conf)

        history.append(step * fusion.cfg.dt, fusion.get_position())

    est_pos = fusion.get_position()
    err = np.linalg.norm(est_pos - true_pos)
    print(f"Vị trí thật (m): {true_pos}")
    print(f"Vị trí ước lượng sau fusion (m): {est_pos}")
    print(f"Sai số cuối cùng: {err:.4f} m")
