"""
benchmark.py
------------
Script chạy đánh giá hiệu năng (benchmark) so sánh định lượng giữa:
  1. Chỉ sử dụng 1 UAV (UAV 1 - Không có fusion)
  2. Hợp nhất Kalman Filter đa UAV (Multi-UAV Kalman Fusion)
  3. Hợp nhất Attention + LSTM đa UAV (Multi-UAV LSTM Fusion)

Quy trình:
  1. Tự động sinh dữ liệu quỹ đạo mục tiêu giả lập (quỹ đạo chữ S/hình tròn).
  2. Tạo quan sát camera chuẩn hoá (cx, cy) cho từng UAV kèm nhiễu Gaussian.
  3. Huấn luyện nhanh mạng LSTMFusionModel bằng dữ liệu quỹ đạo giả lập ngẫu nhiên.
  4. Thực hiện fusion và đánh giá RMSE sai số bám mục tiêu của các phương án.
  5. Xuất biểu đồ so sánh quỹ đạo và sai số ra file ảnh để báo cáo.
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from fusion.coordinate_utils import camera_to_world, world_to_camera
from fusion.kalman_fusion import MultiUAVKalmanFusion, KalmanFusionConfig
from fusion.lstm_fusion import LSTMFusionModel, train_step

# Cấu hình không gian & thời gian
ARENA_SIZE_M = 6.0
DT = 0.1
STEPS = 250
SIGMA_PIXEL = 0.025  # Nhiễu camera (độ lệch chuẩn trên toạ độ chuẩn hoá [0, 1])

# Cấu hình tư thế 3 UAV tương tự dashboard
UAV_POSES = {
    1: {"pos": (0.5, 0.5, 2.5), "yaw": 45.0, "pitch": 45.0},
    2: {"pos": (5.5, 0.5, 2.5), "yaw": 135.0, "pitch": 45.0},
    3: {"pos": (3.0, 5.5, 2.8), "yaw": 270.0, "pitch": 50.0},
}


def generate_target_trajectory(trajectory_type="circle", steps=STEPS, dt=DT):
    """Sinh quỹ đạo thực của mục tiêu (RC car) trong hệ tọa độ 6x6 m."""
    t = np.arange(steps) * dt
    if trajectory_type == "circle":
        # Quỹ đạo tròn tâm (3,3) bán kính 2m
        xs = 3.0 + 1.8 * np.cos(0.15 * t)
        ys = 3.0 + 1.8 * np.sin(0.15 * t)
    elif trajectory_type == "sine":
        # Quỹ đạo hình sin uốn lượn xuyên qua sân
        xs = 1.0 + 4.0 * (t / t[-1])
        ys = 3.0 + 1.5 * np.sin(2.0 * np.pi * (t / t[-1]) * 1.5)
    else:
        # Đường thẳng đổi hướng
        xs = np.zeros(steps)
        ys = np.zeros(steps)
        half = steps // 2
        # Nửa đầu đi thẳng góc
        xs[:half] = 1.0 + 4.0 * np.arange(half) / half
        ys[:half] = 1.5 + 1.5 * np.arange(half) / half
        # Nửa sau bẻ cua
        xs[half:] = xs[half - 1] - 3.0 * np.arange(steps - half) / (steps - half)
        ys[half:] = ys[half - 1] + 1.5 * np.arange(steps - half) / (steps - half)

    # Đảm bảo nằm trong sân
    xs = np.clip(xs, 0.1, ARENA_SIZE_M - 0.1)
    ys = np.clip(ys, 0.1, ARENA_SIZE_M - 0.1)
    return np.stack([xs, ys], axis=1)


def simulate_observations(true_trajectory, uav_poses, sigma=SIGMA_PIXEL, packet_drop_rate=0.0):
    """
    Simulate camera observations for each UAV.
    Returns:
      obs_dict: uav_id -> list of (cx, cy) or None
      world_obs_dict: uav_id -> list of world (x, y) or None
    """
    obs_dict = {uav_id: [] for uav_id in uav_poses}
    world_obs_dict = {uav_id: [] for uav_id in uav_poses}

    for pos in true_trajectory:
        for uav_id, pose in uav_poses.items():
            cam_xy = world_to_camera(pos, pose["pos"], pose["yaw"], pose["pitch"])
            if cam_xy is not None:
                # Mô phỏng rớt gói tin mạng (UDP packet loss)
                if np.random.rand() < packet_drop_rate:
                    obs_dict[uav_id].append(None)
                    world_obs_dict[uav_id].append(None)
                    continue

                # Thêm nhiễu Gaussian trong pixel domain
                noisy_cx = cam_xy[0] + np.random.normal(0, sigma)
                noisy_cy = cam_xy[1] + np.random.normal(0, sigma)
                noisy_cx = np.clip(noisy_cx, 0.0, 1.0)
                noisy_cy = np.clip(noisy_cy, 0.0, 1.0)
                
                # Chiếu ngược lại toạ độ thế giới
                world_est = camera_to_world(noisy_cx, noisy_cy, pose["pos"], pose["yaw"], pose["pitch"])
                obs_dict[uav_id].append(np.array([noisy_cx, noisy_cy]))
                world_obs_dict[uav_id].append(world_est)
            else:
                obs_dict[uav_id].append(None)
                world_obs_dict[uav_id].append(None)

    return obs_dict, world_obs_dict


def train_lstm_fusion(uav_poses, epochs=60, num_trajectories=100):
    """Tự động sinh tập dữ liệu mô phỏng và huấn luyện LSTMFusionModel."""
    print("[LSTM] Đang khởi tạo và huấn luyện mô hình fusion...")
    model = LSTMFusionModel(obs_dim=3, hidden_dim=32, lstm_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Sinh dữ liệu huấn luyện
    obs_seqs = []
    mask_seqs = []
    target_seqs = []

    for _ in range(num_trajectories):
        traj_type = np.random.choice(["circle", "sine", "lines"])
        true_traj = generate_target_trajectory(traj_type, steps=STEPS, dt=DT)
        _, world_obs = simulate_observations(true_traj, uav_poses)

        # Định dạng input cho LSTM: [T, N_uav, 3] trong đó 3 = [px, py, confidence]
        # Nếu UAV không quan sát được, đặt giá trị bằng 0 và mask = False
        seq_data = []
        seq_mask = []
        for t in range(STEPS):
            uav_data = []
            uav_mask = []
            for uav_id in sorted(uav_poses.keys()):
                obs = world_obs[uav_id][t]
                if obs is not None:
                    uav_data.append([obs[0], obs[1], 1.0])
                    uav_mask.append(True)
                else:
                    uav_data.append([0.0, 0.0, 0.0])
                    uav_mask.append(False)
            seq_data.append(uav_data)
            seq_mask.append(uav_mask)

        obs_seqs.append(seq_data)
        mask_seqs.append(seq_mask)
        target_seqs.append(true_traj)

    obs_tensor = torch.tensor(obs_seqs, dtype=torch.float32)      # [B, T, N, 3]
    mask_tensor = torch.tensor(mask_seqs, dtype=torch.bool)        # [B, T, N]
    target_tensor = torch.tensor(target_seqs, dtype=torch.float32)  # [B, T, 2]

    # Vòng lặp training
    model.train()
    for epoch in range(epochs):
        loss_val = train_step(model, optimizer, obs_tensor, target_tensor, mask_tensor)
        if (epoch + 1) % 15 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} — MSE Loss: {loss_val:.5f}")

    model.eval()
    return model


def main():
    # 0. Thiết lập seed ngẫu nhiên để tái lập kết quả
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Sinh quỹ đạo test (hình sin uốn lượn để kiểm thử)
    true_traj = generate_target_trajectory("sine", steps=STEPS, dt=DT)
    
    # ĐẶT TỈ LỆ RỚT GÓI MẠNG LÀ 15% (packet loss)
    packet_drop_rate = 0.15
    print(f"[Network] Mô phỏng UDP packet loss rate = {packet_drop_rate*100}%")
    obs_dict, world_obs = simulate_observations(true_traj, UAV_POSES, packet_drop_rate=packet_drop_rate)

    # 2. Huấn luyện LSTM
    lstm_model = train_lstm_fusion(UAV_POSES, epochs=60)

    # 3. Thực hiện Fusion bằng các phương án

    # Phương án A: Chỉ sử dụng UAV 1 (Không fusion)
    uav1_est = []
    last_known = np.array([3.0, 3.0])  # điểm khởi đầu mặc định
    for obs in world_obs[1]:
        if obs is not None:
            last_known = obs.copy()
        uav1_est.append(last_known)
    uav1_est = np.array(uav1_est)

    # Phương án B: Kalman Filter Fusion (Dynamic DT)
    kf_fusion = MultiUAVKalmanFusion(KalmanFusionConfig(dt=DT, process_noise_std=0.4, base_measurement_noise_std=0.25))
    kf_est = []
    
    # Khởi tạo bước đầu tiên
    initialized = False
    for t in range(STEPS):
        #predict chuyển động
        if initialized:
            kf_fusion.predict(dt=DT)
        
        # Nhận quan sát từ các UAV có thể nhìn thấy tại bước t
        for uav_id in sorted(UAV_POSES.keys()):
            obs = world_obs[uav_id][t]
            if obs is not None:
                kf_fusion.update(obs, confidence=0.8)
                initialized = True
        
        kf_est.append(kf_fusion.get_position())
    kf_est = np.array(kf_est)

    # Phương án C: LSTM + Attention Fusion
    # Tạo tensor đầu vào cho mạng
    test_data = []
    test_mask = []
    for t in range(STEPS):
        uav_data = []
        uav_mask = []
        for uav_id in sorted(UAV_POSES.keys()):
            obs = world_obs[uav_id][t]
            if obs is not None:
                uav_data.append([obs[0], obs[1], 1.0])
                uav_mask.append(True)
            else:
                uav_data.append([0.0, 0.0, 0.0])
                uav_mask.append(False)
        test_data.append(uav_data)
        test_mask.append(uav_mask)
    
    test_obs_tensor = torch.tensor([test_data], dtype=torch.float32)  # [1, T, N, 3]
    test_mask_tensor = torch.tensor([test_mask], dtype=torch.bool)    # [1, T, N]

    with torch.no_grad():
        lstm_est_tensor = lstm_model(test_obs_tensor, test_mask_tensor) # [1, T, 2]
    lstm_est = lstm_est_tensor.squeeze(0).numpy()

    # 4. Tính toán sai số RMSE
    rmse_uav1 = np.sqrt(np.mean(np.sum((uav1_est - true_traj) ** 2, axis=1)))
    rmse_kf = np.sqrt(np.mean(np.sum((kf_est - true_traj) ** 2, axis=1)))
    rmse_lstm = np.sqrt(np.mean(np.sum((lstm_est - true_traj) ** 2, axis=1)))

    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ HIỆN NĂNG BÁM MỤC TIÊU (RMSE)")
    print("=" * 50)
    print(f"1. Chỉ sử dụng UAV 1 (No Fusion):       {rmse_uav1:.4f} mét")
    print(f"2. Kalman Filter Fusion (3 UAVs):        {rmse_kf:.4f} mét")
    print(f"3. Attention + LSTM Fusion (3 UAVs):     {rmse_lstm:.4f} mét")
    
    gain_kf = (rmse_uav1 - rmse_kf) / rmse_uav1 * 100
    gain_lstm = (rmse_uav1 - rmse_lstm) / rmse_uav1 * 100
    print("-" * 50)
    print(f"Mức tăng độ chính xác của Kalman Filter: +{gain_kf:.2f}% (Mục tiêu đề tài: +15%)")
    print(f"Mức tăng độ chính xác của LSTM Fusion:  +{gain_lstm:.2f}%")
    print("=" * 50)

    # 5. Vẽ biểu đồ so sánh
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Đồ thị quỹ đạo
    ax1.plot(true_traj[:, 0], true_traj[:, 1], "k--", label="Quỹ đạo thực", linewidth=2)
    ax1.plot(uav1_est[:, 0], uav1_est[:, 1], "r:", label=f"Chỉ UAV 1 (RMSE: {rmse_uav1:.2f}m)", alpha=0.7)
    ax1.plot(kf_est[:, 0], kf_est[:, 1], "b-", label=f"Kalman Fusion (RMSE: {rmse_kf:.2f}m)", linewidth=1.5)
    ax1.plot(lstm_est[:, 0], lstm_est[:, 1], "g-.", label=f"LSTM Fusion (RMSE: {rmse_lstm:.2f}m)", linewidth=1.5)
    
    # Vẽ vị trí 3 UAV lên đồ thị
    for uav_id, pose in UAV_POSES.items():
        ax1.plot(pose["pos"][0], pose["pos"][1], "m^", markersize=10, label=f"UAV {uav_id} (cao {pose['pos'][2]}m)" if uav_id == 1 else None)
        ax1.text(pose["pos"][0] + 0.1, pose["pos"][1] + 0.1, f"UAV {uav_id}", color="purple", fontweight="bold")

    ax1.set_title("So sánh Quỹ đạo bám mục tiêu trên mặt đất 6x6 m")
    ax1.set_xlabel("X (mét)")
    ax1.set_ylabel("Y (mét)")
    ax1.set_xlim(0, ARENA_SIZE_M)
    ax1.set_ylim(0, ARENA_SIZE_M)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Đồ thị sai số theo thời gian
    err_uav1 = np.linalg.norm(uav1_est - true_traj, axis=1)
    err_kf = np.linalg.norm(kf_est - true_traj, axis=1)
    err_lstm = np.linalg.norm(lstm_est - true_traj, axis=1)
    
    steps_axis = np.arange(STEPS) * DT
    ax2.plot(steps_axis, err_uav1, "r:", label="Chỉ UAV 1", alpha=0.7)
    ax2.plot(steps_axis, err_kf, "b-", label="Kalman Fusion")
    ax2.plot(steps_axis, err_lstm, "g-.", label="LSTM Fusion")
    
    ax2.set_title("Sai số khoảng cách tức thời theo thời gian")
    ax2.set_xlabel("Thời gian (giây)")
    ax2.set_ylabel("Sai số (mét)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    # Tạo thư mục lưu kết quả nếu chưa có
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fusion_benchmark.png")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\n[OK] Đã xuất biểu đồ so sánh thành công tại: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
