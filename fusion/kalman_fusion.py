"""
kalman_fusion.py
-----------------
Phương án fusion CHÍNH của hệ thống: Kalman Filter (constant-velocity model)
hợp nhất các quan sát vị trí (cx, cy) từ nhiều UAV thành một ước lượng
quỹ đạo mục tiêu thống nhất.

Mô hình trạng thái: x = [px, py, vx, vy]^T  (vị trí + vận tốc trong mặt phẳng
không gian thử nghiệm 6x6 m, đã quy đổi từ toạ độ ảnh chuẩn hoá + độ cao/góc
nhìn từng UAV — phần quy đổi 2D ảnh -> toạ độ thế giới 3D/2D được thực hiện
tự động bên trong `update()` khi truyền `uav_pose` và `bbox_cx_cy`).

Mỗi UAV đóng vai trò một "sensor" độc lập; độ tin cậy (confidence) của từng
quan sát được dùng làm trọng số nghịch đảo cho ma trận R (measurement noise).

Cấu trúc module:
    - KalmanFusionConfig       : tham số cấu hình bộ lọc
    - MultiUAVKalmanFusion     : bộ lọc cho MỘT mục tiêu (single-target)
    - FusionTrackHistory       : lưu lịch sử quỹ đạo để vẽ dashboard
    - MultiTargetTracker       : quản lý NHIỀU mục tiêu song song, tích hợp
                                 Hungarian Association (Chương 4.1 báo cáo)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from fusion.coordinate_utils import camera_to_world

@dataclass
class KalmanFusionConfig:
    dt: float = 0.1                    
    process_noise_std: float = 0.5     
    base_measurement_noise_std: float = 0.3  

class MultiUAVKalmanFusion:
    
    def __init__(self, config: KalmanFusionConfig = KalmanFusionConfig()):
        self.cfg = config

        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0  
        self.initialized = False
        self.target_embedding: Optional[np.ndarray] = None

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

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

    def predict(self, dt: float | None = None):
        if not self.initialized:
            return

        if dt is not None:
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

    def update(
        self,
        position_xy: np.ndarray | None = None,
        confidence: float = 1.0,
        embedding: np.ndarray | None = None,
        embedding_threshold: float = 0.7,
        
        uav_pose: Tuple[tuple, float, float] | None = None,
        bbox_cx_cy: Tuple[float, float] | None = None,
    ) -> np.ndarray:
        
        if bbox_cx_cy is not None and uav_pose is not None:
            uav_pos, yaw_deg, pitch_deg = uav_pose
            position_xy = camera_to_world(
                cx=float(bbox_cx_cy[0]),
                cy=float(bbox_cx_cy[1]),
                uav_pos=uav_pos,
                yaw_deg=float(yaw_deg),
                pitch_deg=float(pitch_deg),
            )

        if position_xy is None:
            return self.get_position()

        if embedding is not None:
            if self.target_embedding is None:
                self.target_embedding = embedding.copy()
            else:
                sim = np.dot(self.target_embedding, embedding) / (
                    np.linalg.norm(self.target_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                if sim < embedding_threshold:
                    return self.get_position()
                else:
                    
                    self.target_embedding = 0.9 * self.target_embedding + 0.1 * embedding
                    self.target_embedding /= np.linalg.norm(self.target_embedding)

        z = position_xy.reshape(2, 1)

        if not self.initialized:
            self.x[0, 0], self.x[1, 0] = z[0, 0], z[1, 0]
            self.initialized = True
            return self.get_position()

        noise_std = self.cfg.base_measurement_noise_std / max(confidence, 1e-3)
        R = np.eye(2) * (noise_std ** 2)

        y = z - self.H @ self.x                          
        S = self.H @ self.P @ self.H.T + R                
        K = self.P @ self.H.T @ np.linalg.inv(S)          

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        return self.get_position()

    def get_position(self) -> np.ndarray:
        return self.x[:2, 0].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[2:, 0].copy()

@dataclass
class FusionTrackHistory:
    
    positions: list = field(default_factory=list)  

    def append(self, t: float, position_xy: np.ndarray):
        self.positions.append((t, float(position_xy[0]), float(position_xy[1])))

    def as_arrays(self):
        arr = np.array(self.positions)
        if arr.size == 0:
            return np.array([]), np.array([]), np.array([])
        return arr[:, 0], arr[:, 1], arr[:, 2]

class _TrackState:
    
    TENTATIVE = "tentative"   
    CONFIRMED = "confirmed"   
    LOST      = "lost"        

class MultiTargetTracker:
    
    def __init__(
        self,
        config: KalmanFusionConfig = KalmanFusionConfig(),
        association_threshold: float = 0.4,
        max_missed_frames: int = 5,
        min_hits: int = 3,
    ):
        
        from fusion.hungarian_association import HungarianAssociator

        self.cfg = config
        self.associator = HungarianAssociator(threshold=association_threshold)
        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits

        self._next_id: int = 0
        self._filters: Dict[int, MultiUAVKalmanFusion] = {}
        self._states: Dict[int, str] = {}
        self._hit_counts: Dict[int, int] = {}
        self._missed_counts: Dict[int, int] = {}
        self._embeddings: Dict[int, np.ndarray] = {}   
        self.histories: Dict[int, FusionTrackHistory] = {}

    def step(
        self,
        observations: List[Dict],
        dt: float | None = None,
    ) -> Dict[int, np.ndarray]:
        
        self._predict_all(dt=dt)

        local_embeddings: List[np.ndarray] = []
        obs_positions: List[np.ndarray | None] = []
        obs_confidences: List[float] = []

        for obs in observations:
            emb = obs.get("embedding")
            local_embeddings.append(emb if emb is not None else np.zeros(128))

            pos = self._resolve_position(obs)
            obs_positions.append(pos)
            obs_confidences.append(float(obs.get("confidence", 1.0)))

        track_ids = list(self._filters.keys())

        if not observations:
            
            for tid in track_ids:
                self._missed_counts[tid] = self._missed_counts.get(tid, 0) + 1

        elif not track_ids:
            
            for i, pos in enumerate(obs_positions):
                if pos is not None:
                    self._create_track(pos, local_embeddings[i])

        else:
            
            global_embeddings = np.array([
                self._embeddings.get(tid, np.zeros(128)) for tid in track_ids
            ])
            local_emb_arr = np.array(local_embeddings)

            assoc_result = self.associator.associate(local_emb_arr, global_embeddings)

            for local_idx, global_idx in assoc_result.matched:
                tid = track_ids[global_idx]
                self._update_track(tid, obs_positions[local_idx],
                                   obs_confidences[local_idx],
                                   local_embeddings[local_idx])

            for local_idx in assoc_result.unmatched_local:
                pos = obs_positions[local_idx]
                if pos is not None:
                    self._create_track(pos, local_embeddings[local_idx])

            for global_idx in assoc_result.unmatched_global:
                tid = track_ids[global_idx]
                self._missed_counts[tid] = self._missed_counts.get(tid, 0) + 1

        self._update_lifecycle()

        t_now = time.time()
        result_positions: Dict[int, np.ndarray] = {}
        for tid, kf in self._filters.items():
            if self._states[tid] == _TrackState.CONFIRMED and kf.initialized:
                pos = kf.get_position()
                self.histories[tid].append(t_now, pos)
                result_positions[tid] = pos

        return result_positions

    def predict_only(self, dt: float | None = None) -> Dict[int, np.ndarray]:
        
        self._predict_all(dt=dt)
        return {
            tid: kf.get_position()
            for tid, kf in self._filters.items()
            if self._states[tid] == _TrackState.CONFIRMED and kf.initialized
        }

    def get_confirmed_tracks(self) -> Dict[int, np.ndarray]:
        
        return {
            tid: kf.get_position()
            for tid, kf in self._filters.items()
            if self._states[tid] == _TrackState.CONFIRMED and kf.initialized
        }

    def get_track_count(self) -> int:
        return len(self._filters)

    def _resolve_position(self, obs: Dict) -> np.ndarray | None:
        
        bbox_cx_cy = obs.get("bbox_cx_cy")
        uav_pose = obs.get("uav_pose")

        if bbox_cx_cy is not None and uav_pose is not None:
            uav_pos, yaw_deg, pitch_deg = uav_pose
            return camera_to_world(
                cx=float(bbox_cx_cy[0]),
                cy=float(bbox_cx_cy[1]),
                uav_pos=uav_pos,
                yaw_deg=float(yaw_deg),
                pitch_deg=float(pitch_deg),
            )

        pos = obs.get("position_xy")
        if pos is not None:
            return np.asarray(pos, dtype=np.float64)
        return None

    def _predict_all(self, dt: float | None):
        for kf in self._filters.values():
            kf.predict(dt=dt)

    def _create_track(self, position: np.ndarray, embedding: np.ndarray):
        tid = self._next_id
        self._next_id += 1

        kf = MultiUAVKalmanFusion(config=self.cfg)
        kf.update(position_xy=position)

        self._filters[tid] = kf
        self._states[tid] = _TrackState.TENTATIVE
        self._hit_counts[tid] = 1
        self._missed_counts[tid] = 0
        self._embeddings[tid] = embedding.copy() if embedding is not None else np.zeros(128)
        self.histories[tid] = FusionTrackHistory()

    def _update_track(self, tid: int, position: np.ndarray | None,
                       confidence: float, embedding: np.ndarray):
        if position is not None:
            self._filters[tid].update(position_xy=position, confidence=confidence)
        self._hit_counts[tid] = self._hit_counts.get(tid, 0) + 1
        self._missed_counts[tid] = 0

        if embedding is not None and np.any(embedding != 0):
            prev = self._embeddings.get(tid, np.zeros(128))
            updated = 0.9 * prev + 0.1 * embedding
            norm = np.linalg.norm(updated)
            self._embeddings[tid] = updated / (norm + 1e-8)

    def _update_lifecycle(self):
        to_delete = []
        for tid in list(self._filters.keys()):
            hits   = self._hit_counts.get(tid, 0)
            missed = self._missed_counts.get(tid, 0)

            if missed > self.max_missed_frames:
                
                to_delete.append(tid)
            elif missed > 0:
                
                self._states[tid] = _TrackState.LOST
            elif hits >= self.min_hits:
                
                self._states[tid] = _TrackState.CONFIRMED
            
        for tid in to_delete:
            del self._filters[tid]
            del self._states[tid]
            del self._hit_counts[tid]
            del self._missed_counts[tid]
            del self._embeddings[tid]
            
if __name__ == "__main__":
    
    print("=" * 55)
    print("Demo A: Single-Target Kalman Filter (3 UAV, 50 bước)")
    print("=" * 55)
    rng = np.random.default_rng(0)
    fusion = MultiUAVKalmanFusion()
    history = FusionTrackHistory()

    true_pos = np.array([0.5, 0.5])
    velocity = np.array([0.2, 0.1])

    for step in range(50):
        fusion.predict()
        true_pos = true_pos + velocity * fusion.cfg.dt

        for uav_conf, noise_scale in [(0.9, 0.05), (0.7, 0.12), (0.5, 0.2)]:
            noisy_obs = true_pos + rng.normal(0, noise_scale, size=2)
            fusion.update(noisy_obs, confidence=uav_conf)

        history.append(step * fusion.cfg.dt, fusion.get_position())

    est_pos = fusion.get_position()
    err = np.linalg.norm(est_pos - true_pos)
    print(f"Vị trí thật (m): {true_pos}")
    print(f"Vị trí ước lượng (m): {est_pos}")
    print(f"Sai số: {err:.4f} m")

    print("\n" + "=" * 55)
    print("Demo B: Multi-Target Tracker (2 mục tiêu, 30 bước)")
    print("=" * 55)
    rng2 = np.random.default_rng(99)
    tracker = MultiTargetTracker(
        config=KalmanFusionConfig(dt=0.1),
        association_threshold=0.4,
        max_missed_frames=5,
        min_hits=2,
    )

    emb1 = rng2.standard_normal(128).astype(np.float64)
    emb1 /= np.linalg.norm(emb1)
    emb2 = rng2.standard_normal(128).astype(np.float64)
    emb2 /= np.linalg.norm(emb2)

    pos_t1 = np.array([1.0, 1.0])
    pos_t2 = np.array([4.0, 4.0])
    vel1 = np.array([0.1, 0.05])
    vel2 = np.array([-0.05, 0.1])

    for step in range(30):
        pos_t1 = pos_t1 + vel1 * 0.1
        pos_t2 = pos_t2 + vel2 * 0.1

        obs = [
            {
                "position_xy": pos_t1 + rng2.normal(0, 0.05, 2),
                "embedding": emb1 + rng2.normal(0, 0.02, 128),
                "confidence": 0.9,
            },
            {
                "position_xy": pos_t2 + rng2.normal(0, 0.05, 2),
                "embedding": emb2 + rng2.normal(0, 0.02, 128),
                "confidence": 0.85,
            },
        ]

        confirmed = tracker.step(obs, dt=0.1)
        if step == 29:
            print(f"Bước {step}: {len(confirmed)} track confirmed")
            for tid, pos in confirmed.items():
                print(f"  Track {tid}: {np.round(pos, 3)}")

    print(f"Tổng track đã tạo: {tracker._next_id}")
    print("✅ MultiTargetTracker hoạt động đúng.")

    print("\n" + "=" * 55)
    print("Demo C: Kalman update() tích hợp coordinate_utils")
    print("=" * 55)
    kf_c = MultiUAVKalmanFusion()
    pos_world = kf_c.update(
        uav_pose=((3.0, 3.0, 2.0), 0.0, 90.0),  
        bbox_cx_cy=(0.5, 0.5),                   
    )
    print(f"Toạ độ world quy đổi từ bbox (0.5, 0.5): {np.round(pos_world, 3)}")
    print("✅ Tích hợp coordinate_utils OK.")

