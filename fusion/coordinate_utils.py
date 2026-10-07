"""
coordinate_utils.py
-------------------
Các hàm tiện ích phục vụ quy đổi toạ độ giữa hệ toạ độ camera UAV
(chuẩn hoá trong khoảng [0, 1]) và hệ toạ độ thế giới 2D của sân đấu 6x6 m.

Mô hình chiếu tia từ tâm camera qua điểm ảnh xuống mặt phẳng đất Z = 0.
"""
from __future__ import annotations

import numpy as np

def camera_to_world(
    cx: float,
    cy: float,
    uav_pos: tuple[float, float, float] | np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_h_deg: float = 82.6,
    aspect_ratio: float = 4.0 / 3.0,
    arena_size_m: float = 6.0
) -> np.ndarray:
    
    X_c, Y_c, Z_c = uav_pos

    if Z_c < 0.05:
        return np.array([np.clip(X_c, 0, arena_size_m), np.clip(Y_c, 0, arena_size_m)], dtype=np.float64)

    u = 2.0 * cx - 1.0
    v = 1.0 - 2.0 * cy

    half_fov_h = np.radians(fov_h_deg / 2.0)
    
    tan_half_h = np.tan(half_fov_h)
    tan_half_v = tan_half_h / aspect_ratio

    x_l = u * tan_half_h
    y_l = v * tan_half_v
    z_l = 1.0

    pitch_rad = np.radians(pitch_deg)
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)

    v_rot = np.array([
        y_l * sin_p + z_l * cos_p,
        -x_l,
        y_l * cos_p - z_l * sin_p
    ], dtype=np.float64)

    yaw_rad = np.radians(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)

    v_world = np.array([
        v_rot[0] * cos_y - v_rot[1] * sin_y,
        v_rot[0] * sin_y + v_rot[1] * cos_y,
        v_rot[2]
    ], dtype=np.float64)

    v_z = v_world[2]

    if v_z >= -1e-4:
        
        t = 100.0  
    else:
        t = -Z_c / v_z

    X_ground = X_c + t * v_world[0]
    Y_ground = Y_c + t * v_world[1]

    X_ground = np.clip(X_ground, 0.0, arena_size_m)
    Y_ground = np.clip(Y_ground, 0.0, arena_size_m)

    return np.array([X_ground, Y_ground], dtype=np.float64)

def world_to_camera(
    world_xy: tuple[float, float] | np.ndarray,
    uav_pos: tuple[float, float, float] | np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_h_deg: float = 82.6,
    aspect_ratio: float = 4.0 / 3.0
) -> np.ndarray | None:
    
    X_g, Y_g = world_xy
    X_c, Y_c, Z_c = uav_pos

    dx_w = X_g - X_c
    dy_w = Y_g - Y_c
    dz_w = 0.0 - Z_c  

    yaw_rad = np.radians(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)

    dx_yaw = dx_w * cos_y + dy_w * sin_y
    dy_yaw = -dx_w * sin_y + dy_w * cos_y
    dz_yaw = dz_w

    pitch_rad = np.radians(pitch_deg)
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)

    X_cam = -dy_yaw
    Y_cam = dx_yaw * sin_p + dz_yaw * cos_p
    Z_cam = dx_yaw * cos_p - dz_yaw * sin_p

    if Z_cam <= 1e-3:
        return None

    half_fov_h = np.radians(fov_h_deg / 2.0)
    tan_half_h = np.tan(half_fov_h)
    tan_half_v = tan_half_h / aspect_ratio

    u = X_cam / (Z_cam * tan_half_h)
    v = Y_cam / (Z_cam * tan_half_v)

    if np.abs(u) > 1.0 or np.abs(v) > 1.0:
        return None

    cx = (u + 1.0) / 2.0
    cy = (1.0 - v) / 2.0

    return np.array([cx, cy], dtype=np.float64)

if __name__ == "__main__":
    
    p1 = camera_to_world(0.5, 0.5, (3.0, 3.0, 2.0), yaw_deg=0.0, pitch_deg=90.0)
    print("Test 1 (Nadir, Center): Expected [3.0, 3.0], Got:", p1)
    assert np.allclose(p1, [3.0, 3.0])

    c1 = world_to_camera((3.0, 3.0), (3.0, 3.0, 2.0), yaw_deg=0.0, pitch_deg=90.0)
    print("Reverse Test 1: Expected [0.5, 0.5], Got:", c1)
    assert np.allclose(c1, [0.5, 0.5])

    p2 = camera_to_world(0.5, 0.5, (0.0, 0.0, 2.0), yaw_deg=45.0, pitch_deg=45.0)
    print("Test 2 (Diagonal Look): Got:", p2)
    c2 = world_to_camera(p2, (0.0, 0.0, 2.0), yaw_deg=45.0, pitch_deg=45.0)
    print("Reverse Test 2: Got:", c2)
    assert np.allclose(c2, [0.5, 0.5])

