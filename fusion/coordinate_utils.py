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
    """
    Quy đổi toạ độ tâm bounding box (cx, cy) chuẩn hoá sang toạ độ mặt đất (X, Y).

    Tham số:
      cx, cy      : Toạ độ tâm bbox, chuẩn hoá trong [0, 1]. cx=0: trái, cy=0: trên.
      uav_pos     : Toạ độ 3D của UAV (X_uav, Y_uav, Z_uav) trong hệ thế giới (mét).
      yaw_deg     : Góc xoay yaw của UAV (độ, quay quanh trục Z, 0 là hướng +X, 90 là hướng +Y).
      pitch_deg   : Góc cúi camera (độ, 0 là nhìn ngang, 90 là nhìn thẳng xuống đất).
      fov_h_deg   : Góc mở camera theo phương ngang (Tello mặc định ~82.6 độ).
      aspect_ratio: Tỷ lệ khung hình (mặc định 4:3).
      arena_size_m: Kích thước cạnh sân đấu (m). Dùng để clip toạ độ đầu ra.

    Trả về:
      np.ndarray shape (2,) — Toạ độ mặt đất [X, Y] (mét).
    """
    X_c, Y_c, Z_c = uav_pos

    # Nếu UAV hạ cánh sát đất (Z_c cực nhỏ), trả về vị trí UAV trên mặt đất
    if Z_c < 0.05:
        return np.array([np.clip(X_c, 0, arena_size_m), np.clip(Y_c, 0, arena_size_m)], dtype=np.float64)

    # Chuyển đổi cx, cy sang hệ toạ độ chuẩn hoá đối xứng [-1, 1] trên cảm biến
    # u: trái-phải (-1 -> 1)
    # v: dưới-trên (-1 -> 1) (lưu ý cy=0 ở trên nên v = 1 - 2*cy)
    u = 2.0 * cx - 1.0
    v = 1.0 - 2.0 * cy

    # Tính nửa góc mở camera
    half_fov_h = np.radians(fov_h_deg / 2.0)
    # tan(half_fov_v) = tan(half_fov_h) / aspect_ratio
    tan_half_h = np.tan(half_fov_h)
    tan_half_v = tan_half_h / aspect_ratio

    # Ray vector trong hệ trục của riêng camera (local):
    # Trục Z_cam hướng dọc theo trục quang học.
    # Trục X_cam hướng sang phải.
    # Trục Y_cam hướng lên trên.
    x_l = u * tan_half_h
    y_l = v * tan_half_v
    z_l = 1.0

    # Chuyển đổi từ hệ camera sang hệ trung gian (trước khi xoay Yaw):
    # Khi Pitch = 0, camera hướng theo chiều ngang (+X thế giới),
    # trục phải camera hướng theo -Y thế giới, trục trên camera hướng theo +Z thế giới.
    pitch_rad = np.radians(pitch_deg)
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)

    # Xoay quanh trục phải camera (X_cam) bởi góc Pitch
    # Hướng camera lệch dần xuống đất (-Z thế giới) khi Pitch dương
    # Chiều X_cam_world = [0, -1, 0]
    # Chiều Y_cam_world = [sin_p, 0, cos_p]
    # Chiều Z_cam_world = [cos_p, 0, -sin_p]
    v_rot = np.array([
        y_l * sin_p + z_l * cos_p,
        -x_l,
        y_l * cos_p - z_l * sin_p
    ], dtype=np.float64)

    # Xoay Yaw quanh trục đứng Z thế giới
    yaw_rad = np.radians(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)

    # Ray vector cuối cùng trong hệ toạ độ thế giới
    v_world = np.array([
        v_rot[0] * cos_y - v_rot[1] * sin_y,
        v_rot[0] * sin_y + v_rot[1] * cos_y,
        v_rot[2]
    ], dtype=np.float64)

    # Giao điểm của ray P(t) = UAV + t*v_world với mặt phẳng đất Z = 0
    # Z_c + t * v_world[2] = 0 => t = -Z_c / v_world[2]
    v_z = v_world[2]

    # Nếu ray hướng ngang hoặc hướng lên trên, không có giao điểm đất thực tế hợp lý
    if v_z >= -1e-4:
        # Giả định tia đi xiên ngang hết cỡ, chiếu xa ở mức xa nhất của hướng đó
        t = 100.0  # Một khoảng cách lớn
    else:
        t = -Z_c / v_z

    # Toạ độ thế giới thực tế
    X_ground = X_c + t * v_world[0]
    Y_ground = Y_c + t * v_world[1]

    # Giới hạn toạ độ trong phạm vi sân đấu 6x6 m
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
    """
    Quy đổi toạ độ thế giới thực mặt đất [X, Y] ngược về toạ độ camera chuẩn hoá (cx, cy).

    Trả về:
      np.ndarray shape (2,) — [cx, cy] nếu mục tiêu nằm trong FOV camera và ở phía trước camera.
      None                 — nếu mục tiêu ở sau camera hoặc ngoài tầm quan sát (FOV).
    """
    X_g, Y_g = world_xy
    X_c, Y_c, Z_c = uav_pos

    # Vector từ camera đến mục tiêu trên mặt đất (Z_g = 0)
    dx_w = X_g - X_c
    dy_w = Y_g - Y_c
    dz_w = 0.0 - Z_c  # mục tiêu ở z=0

    # 1. Xoay ngược Yaw (xoay quanh trục Z góc -yaw_deg)
    yaw_rad = np.radians(yaw_deg)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)

    dx_yaw = dx_w * cos_y + dy_w * sin_y
    dy_yaw = -dx_w * sin_y + dy_w * cos_y
    dz_yaw = dz_w

    # 2. Xoay ngược Pitch (xoay quanh trục ngang camera góc -pitch_deg)
    pitch_rad = np.radians(pitch_deg)
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)

    X_cam = -dy_yaw
    Y_cam = dx_yaw * sin_p + dz_yaw * cos_p
    Z_cam = dx_yaw * cos_p - dz_yaw * sin_p

    # Điểm ở sau camera
    if Z_cam <= 1e-3:
        return None

    # Tính nửa góc mở camera
    half_fov_h = np.radians(fov_h_deg / 2.0)
    tan_half_h = np.tan(half_fov_h)
    tan_half_v = tan_half_h / aspect_ratio

    # Tọa độ sensor chuẩn hóa [-1, 1]
    u = X_cam / (Z_cam * tan_half_h)
    v = Y_cam / (Z_cam * tan_half_v)

    # Kiểm tra xem có nằm ngoài FOV không
    if np.abs(u) > 1.0 or np.abs(v) > 1.0:
        return None

    # Quy đổi về range [0, 1]
    cx = (u + 1.0) / 2.0
    cy = (1.0 - v) / 2.0

    return np.array([cx, cy], dtype=np.float64)


if __name__ == "__main__":
    # Smoke test nhanh:
    # 1. UAV ở vị trí trung tâm (3, 3) cao 2m, nhìn thẳng xuống (pitch = 90)
    # Bbox ở chính giữa ảnh (0.5, 0.5) phải chiếu đúng về (3, 3)
    p1 = camera_to_world(0.5, 0.5, (3.0, 3.0, 2.0), yaw_deg=0.0, pitch_deg=90.0)
    print("Test 1 (Nadir, Center): Expected [3.0, 3.0], Got:", p1)
    assert np.allclose(p1, [3.0, 3.0])

    c1 = world_to_camera((3.0, 3.0), (3.0, 3.0, 2.0), yaw_deg=0.0, pitch_deg=90.0)
    print("Reverse Test 1: Expected [0.5, 0.5], Got:", c1)
    assert np.allclose(c1, [0.5, 0.5])

    # 2. UAV ở vị trí (0, 0) cao 2m, quay góc yaw=45 độ, cúi pitch=45 độ,
    p2 = camera_to_world(0.5, 0.5, (0.0, 0.0, 2.0), yaw_deg=45.0, pitch_deg=45.0)
    print("Test 2 (Diagonal Look): Got:", p2)
    c2 = world_to_camera(p2, (0.0, 0.0, 2.0), yaw_deg=45.0, pitch_deg=45.0)
    print("Reverse Test 2: Got:", c2)
    assert np.allclose(c2, [0.5, 0.5])

