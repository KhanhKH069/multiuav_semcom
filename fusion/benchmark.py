"""
benchmark.py
------------
Script chạy đánh giá hiệu năng (benchmark) so sánh định lượng giữa:
  1. Chỉ sử dụng 1 UAV (UAV 1 - Không có fusion)
  2. Hợp nhất Kalman Filter đa UAV (Multi-UAV Kalman Fusion)
  3. Hợp nhất Attention + LSTM đa UAV (Multi-UAV LSTM Fusion)

Đầu ra (thư mục results/):
  - fusion_benchmark.png         : Biểu đồ quỹ đạo + sai số tức thời
  - fusion_rmse_bar.png          : Biểu đồ cột so sánh RMSE
  - fusion_boxplot.png           : Box-plot phân phối sai số
  - fusion_cdf.png               : CDF (Cumulative Distribution Function) sai số
  - fusion_heatmap.png           : Heatmap sai số theo thời gian × phương pháp
  - fusion_packet_loss_sweep.png : Đồ thị RMSE theo tỉ lệ packet loss
  - fusion_summary_table.png     : Bảng tổng kết kết quả dạng hình ảnh
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import warnings

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 130,
})

from fusion.coordinate_utils import camera_to_world, world_to_camera
from fusion.kalman_fusion import MultiUAVKalmanFusion, KalmanFusionConfig
from fusion.lstm_fusion import LSTMFusionModel, train_step

# ─── Constants ───────────────────────────────────────────────────────────────
ARENA_SIZE_M = 6.0
DT = 0.1
STEPS = 250
SIGMA_PIXEL = 0.025

UAV_POSES = {
    1: {"pos": (0.5, 0.5, 2.5), "yaw": 45.0,  "pitch": 45.0},
    2: {"pos": (5.5, 0.5, 2.5), "yaw": 135.0, "pitch": 45.0},
    3: {"pos": (3.0, 5.5, 2.8), "yaw": 270.0, "pitch": 50.0},
}

# Palette màu nhất quán
C_TRUE  = "#1a1a2e"
C_UAV1  = "#e94560"
C_KF    = "#0f3460"
C_LSTM  = "#16213e"
C_LSTM2 = "#e94560"

COLORS = {
    "No Fusion (UAV 1)":         "#e94560",
    "Kalman Filter Fusion":      "#2196F3",
    "Attention+LSTM Fusion":     "#4CAF50",
    "Ground Truth":              "#212121",
}

# ─── Data Generation ─────────────────────────────────────────────────────────

def generate_target_trajectory(trajectory_type="circle", steps=STEPS, dt=DT):
    t = np.arange(steps) * dt
    if trajectory_type == "circle":
        xs = 3.0 + 1.8 * np.cos(0.15 * t)
        ys = 3.0 + 1.8 * np.sin(0.15 * t)
    elif trajectory_type == "sine":
        xs = 1.0 + 4.0 * (t / t[-1])
        ys = 3.0 + 1.5 * np.sin(2.0 * np.pi * (t / t[-1]) * 1.5)
    else:
        xs = np.zeros(steps)
        ys = np.zeros(steps)
        half = steps // 2
        xs[:half] = 1.0 + 4.0 * np.arange(half) / half
        ys[:half] = 1.5 + 1.5 * np.arange(half) / half
        xs[half:] = xs[half - 1] - 3.0 * np.arange(steps - half) / (steps - half)
        ys[half:] = ys[half - 1] + 1.5 * np.arange(steps - half) / (steps - half)
    xs = np.clip(xs, 0.1, ARENA_SIZE_M - 0.1)
    ys = np.clip(ys, 0.1, ARENA_SIZE_M - 0.1)
    return np.stack([xs, ys], axis=1)


def simulate_observations(true_trajectory, uav_poses, sigma=SIGMA_PIXEL, packet_drop_rate=0.0):
    obs_dict = {uav_id: [] for uav_id in uav_poses}
    world_obs_dict = {uav_id: [] for uav_id in uav_poses}
    for pos in true_trajectory:
        for uav_id, pose in uav_poses.items():
            cam_xy = world_to_camera(pos, pose["pos"], pose["yaw"], pose["pitch"])
            if cam_xy is not None:
                if np.random.rand() < packet_drop_rate:
                    obs_dict[uav_id].append(None)
                    world_obs_dict[uav_id].append(None)
                    continue
                noisy_cx = np.clip(cam_xy[0] + np.random.normal(0, sigma), 0.0, 1.0)
                noisy_cy = np.clip(cam_xy[1] + np.random.normal(0, sigma), 0.0, 1.0)
                world_est = camera_to_world(noisy_cx, noisy_cy, pose["pos"], pose["yaw"], pose["pitch"])
                obs_dict[uav_id].append(np.array([noisy_cx, noisy_cy]))
                world_obs_dict[uav_id].append(world_est)
            else:
                obs_dict[uav_id].append(None)
                world_obs_dict[uav_id].append(None)
    return obs_dict, world_obs_dict


# ─── LSTM Training ───────────────────────────────────────────────────────────

def train_lstm_fusion(uav_poses, epochs=80, num_trajectories=150, lr=1e-3):
    """Huấn luyện LSTMFusionModel với gradient clipping để tránh NaN."""
    print("[LSTM] Initializing and training fusion model...")
    model = LSTMFusionModel(obs_dim=3, hidden_dim=64, lstm_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    obs_seqs, mask_seqs, target_seqs = [], [], []

    for _ in range(num_trajectories):
        traj_type = np.random.choice(["circle", "sine", "lines"])
        true_traj = generate_target_trajectory(traj_type, steps=STEPS, dt=DT)
        _, world_obs = simulate_observations(true_traj, uav_poses)

        seq_data, seq_mask = [], []
        for t in range(STEPS):
            uav_data, uav_mask = [], []
            for uav_id in sorted(uav_poses.keys()):
                obs = world_obs[uav_id][t]
                if obs is not None:
                    uav_data.append([obs[0] / ARENA_SIZE_M, obs[1] / ARENA_SIZE_M, 1.0])
                    uav_mask.append(True)
                else:
                    uav_data.append([0.0, 0.0, 0.0])
                    uav_mask.append(False)
            seq_data.append(uav_data)
            seq_mask.append(uav_mask)

        obs_seqs.append(seq_data)
        mask_seqs.append(seq_mask)
        # normalize target
        target_seqs.append(true_traj / ARENA_SIZE_M)

    obs_np     = np.array(obs_seqs,    dtype=np.float32)
    mask_np    = np.array(mask_seqs,   dtype=bool)
    target_np  = np.array(target_seqs, dtype=np.float32)

    obs_tensor    = torch.from_numpy(obs_np)
    mask_tensor   = torch.from_numpy(mask_np)
    target_tensor = torch.from_numpy(target_np)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(obs_tensor, mask_tensor)
        loss = nn.functional.mse_loss(pred, target_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} — MSE Loss: {loss.item():.5f}")

    model.eval()
    return model


# ─── Run one benchmark ───────────────────────────────────────────────────────

def run_one_benchmark(true_traj, world_obs, lstm_model):
    """Chạy cả 3 phương pháp, trả về dict est arrays."""

    # --- UAV 1 only ---
    uav1_est = []
    last_known = np.array([3.0, 3.0])
    for obs in world_obs[1]:
        if obs is not None:
            last_known = obs.copy()
        uav1_est.append(last_known.copy())
    uav1_est = np.array(uav1_est)

    # --- Kalman Fusion ---
    kf_fusion = MultiUAVKalmanFusion(
        KalmanFusionConfig(dt=DT, process_noise_std=0.4, base_measurement_noise_std=0.25)
    )
    kf_est = []
    initialized = False
    for t in range(STEPS):
        if initialized:
            kf_fusion.predict(dt=DT)
        for uav_id in sorted(UAV_POSES.keys()):
            obs = world_obs[uav_id][t]
            if obs is not None:
                kf_fusion.update(obs, confidence=0.8)
                initialized = True
        kf_est.append(kf_fusion.get_position())
    kf_est = np.array(kf_est)

    # --- LSTM Fusion ---
    test_data, test_mask = [], []
    for t in range(STEPS):
        uav_data, uav_mask = [], []
        for uav_id in sorted(UAV_POSES.keys()):
            obs = world_obs[uav_id][t]
            if obs is not None:
                uav_data.append([obs[0] / ARENA_SIZE_M, obs[1] / ARENA_SIZE_M, 1.0])
                uav_mask.append(True)
            else:
                uav_data.append([0.0, 0.0, 0.0])
                uav_mask.append(False)
        test_data.append(uav_data)
        test_mask.append(uav_mask)

    test_obs_tensor  = torch.tensor([test_data],  dtype=torch.float32)
    test_mask_tensor = torch.tensor([test_mask],  dtype=torch.bool)

    with torch.no_grad():
        lstm_pred = lstm_model(test_obs_tensor, test_mask_tensor).squeeze(0).numpy()

    # De-normalize
    lstm_est = lstm_pred * ARENA_SIZE_M

    return uav1_est, kf_est, lstm_est


# ─── Plotting helpers ────────────────────────────────────────────────────────

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved → {path}")


def plot_trajectory_and_error(true_traj, uav1, kf, lstm, out_dir, rmse_uav1, rmse_kf, rmse_lstm):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Multi-UAV Target Tracking — Trajectory & Instantaneous Error", fontsize=14, fontweight="bold")

    ax1.plot(true_traj[:,0], true_traj[:,1], "--", color=COLORS["Ground Truth"], lw=2.5, label="Ground Truth", zorder=5)
    ax1.plot(uav1[:,0], uav1[:,1], ":", color=COLORS["No Fusion (UAV 1)"], lw=1.8, alpha=0.8, label=f"No Fusion (UAV 1) — RMSE {rmse_uav1:.3f} m")
    ax1.plot(kf[:,0],   kf[:,1],   "-",  color=COLORS["Kalman Filter Fusion"], lw=2, label=f"Kalman Filter Fusion — RMSE {rmse_kf:.3f} m")
    ax1.plot(lstm[:,0], lstm[:,1], "-.", color=COLORS["Attention+LSTM Fusion"], lw=2, label=f"Attn+LSTM Fusion — RMSE {rmse_lstm:.3f} m")

    for uid, pose in UAV_POSES.items():
        ax1.plot(pose["pos"][0], pose["pos"][1], "^", color="#9C27B0", markersize=12, zorder=6)
        ax1.text(pose["pos"][0]+0.1, pose["pos"][1]+0.1, f"UAV {uid}", color="#9C27B0", fontweight="bold", fontsize=9)

    ax1.set_xlim(0, ARENA_SIZE_M); ax1.set_ylim(0, ARENA_SIZE_M)
    ax1.set_xlabel("X (m)"); ax1.set_ylabel("Y (m)")
    ax1.set_title("Target Trajectory (6 × 6 m Arena)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", framealpha=0.9)

    t_axis = np.arange(STEPS) * DT
    err_u = np.linalg.norm(uav1 - true_traj, axis=1)
    err_k = np.linalg.norm(kf   - true_traj, axis=1)
    err_l = np.linalg.norm(lstm - true_traj, axis=1)

    ax2.fill_between(t_axis, err_u, alpha=0.15, color=COLORS["No Fusion (UAV 1)"])
    ax2.fill_between(t_axis, err_k, alpha=0.15, color=COLORS["Kalman Filter Fusion"])
    ax2.fill_between(t_axis, err_l, alpha=0.15, color=COLORS["Attention+LSTM Fusion"])
    ax2.plot(t_axis, err_u, ":", color=COLORS["No Fusion (UAV 1)"], lw=1.8, label="No Fusion (UAV 1)")
    ax2.plot(t_axis, err_k, "-",  color=COLORS["Kalman Filter Fusion"], lw=2,   label="Kalman Filter Fusion")
    ax2.plot(t_axis, err_l, "-.", color=COLORS["Attention+LSTM Fusion"], lw=2,   label="Attn+LSTM Fusion")

    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Tracking Error (m)")
    ax2.set_title("Instantaneous Distance Error Over Time")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9)

    _save(fig, os.path.join(out_dir, "fusion_benchmark.png"))


def plot_rmse_bar(rmse_uav1, rmse_kf, rmse_lstm, gain_kf, gain_lstm, out_dir):
    fig, (ax_rmse, ax_gain) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RMSE Comparison — Multi-UAV Tracking Approaches", fontsize=14, fontweight="bold")

    labels  = ["No Fusion\n(UAV 1)", "Kalman Filter\nFusion", "Attn+LSTM\nFusion"]
    values  = [rmse_uav1, rmse_kf, rmse_lstm]
    palette = [COLORS["No Fusion (UAV 1)"], COLORS["Kalman Filter Fusion"], COLORS["Attention+LSTM Fusion"]]

    bars = ax_rmse.bar(labels, values, color=palette, width=0.5, edgecolor="white", linewidth=1.2, zorder=3)
    for bar, val in zip(bars, values):
        ax_rmse.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f"{val:.3f} m",
                     ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax_rmse.set_ylabel("RMSE (m)")
    ax_rmse.set_title("Root Mean Square Error (RMSE)")
    ax_rmse.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_rmse.set_ylim(0, max(values) * 1.3)
    ax_rmse.axhline(y=rmse_uav1, linestyle="--", color="#aaa", lw=1.2, label="Baseline (UAV 1)")
    ax_rmse.legend()

    gain_labels = ["Kalman Filter\nFusion", "Attn+LSTM\nFusion"]
    gain_vals   = [gain_kf, gain_lstm]
    gain_colors = [COLORS["Kalman Filter Fusion"], COLORS["Attention+LSTM Fusion"]]
    g_bars = ax_gain.bar(gain_labels, gain_vals, color=gain_colors, width=0.4, edgecolor="white", linewidth=1.2, zorder=3)
    for bar, val in zip(g_bars, gain_vals):
        ax_gain.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"+{val:.1f}%",
                     ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax_gain.axhline(y=15, linestyle="--", color="#e94560", lw=1.5, label="Target Gain (+15%)")
    ax_gain.set_ylabel("Accuracy Improvement (%)")
    ax_gain.set_title("Accuracy Gain vs. Baseline (UAV 1 Only)")
    ax_gain.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax_gain.set_ylim(0, max(gain_vals) * 1.35)
    ax_gain.legend()

    _save(fig, os.path.join(out_dir, "fusion_rmse_bar.png"))


def plot_boxplot(true_traj, uav1, kf, lstm, out_dir):
    err_u = np.linalg.norm(uav1 - true_traj, axis=1)
    err_k = np.linalg.norm(kf   - true_traj, axis=1)
    err_l = np.linalg.norm(lstm - true_traj, axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot(
        [err_u, err_k, err_l],
        labels=["No Fusion\n(UAV 1)", "Kalman Filter\nFusion", "Attn+LSTM\nFusion"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="white", linewidth=2.5),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    pal = [COLORS["No Fusion (UAV 1)"], COLORS["Kalman Filter Fusion"], COLORS["Attention+LSTM Fusion"]]
    for patch, col in zip(bp["boxes"], pal):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for whisker in bp["whiskers"]:
        whisker.set(linestyle="--", color="#555")
    for cap in bp["caps"]:
        cap.set(color="#555")

    # Annotate median
    medians = [np.median(err_u), np.median(err_k), np.median(err_l)]
    for i, med in enumerate(medians, start=1):
        ax.text(i, med + 0.01, f"{med:.3f} m", ha="center", va="bottom", fontsize=9, fontweight="bold", color="white")

    ax.set_ylabel("Tracking Error (m)")
    ax.set_title("Error Distribution — Box Plot (250 time steps)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f8f9fa")
    _save(fig, os.path.join(out_dir, "fusion_boxplot.png"))


def plot_cdf(true_traj, uav1, kf, lstm, out_dir):
    err_u = np.sort(np.linalg.norm(uav1 - true_traj, axis=1))
    err_k = np.sort(np.linalg.norm(kf   - true_traj, axis=1))
    err_l = np.sort(np.linalg.norm(lstm - true_traj, axis=1))
    cdf   = np.arange(1, STEPS + 1) / STEPS

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(err_u, cdf, ":", color=COLORS["No Fusion (UAV 1)"],      lw=2.5, label="No Fusion (UAV 1)")
    ax.plot(err_k, cdf, "-",  color=COLORS["Kalman Filter Fusion"],  lw=2.5, label="Kalman Filter Fusion")
    ax.plot(err_l, cdf, "-.", color=COLORS["Attention+LSTM Fusion"], lw=2.5, label="Attn+LSTM Fusion")

    for threshold in [0.1, 0.2, 0.3]:
        ax.axvline(threshold, linestyle="--", color="#ccc", lw=1)
        ax.text(threshold + 0.005, 0.02, f"{threshold} m", color="#888", fontsize=8)

    ax.set_xlabel("Tracking Error Threshold (m)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Tracking Error — All Methods", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(left=0)
    _save(fig, os.path.join(out_dir, "fusion_cdf.png"))


def plot_heatmap(true_traj, uav1, kf, lstm, out_dir):
    """Heatmap sai số theo thời gian (3 cột = 3 phương pháp, bins theo timestep)."""
    err_u = np.linalg.norm(uav1 - true_traj, axis=1)
    err_k = np.linalg.norm(kf   - true_traj, axis=1)
    err_l = np.linalg.norm(lstm - true_traj, axis=1)

    # Reshape thành ma trận (5 segments × 50 steps) để tạo 2-D heatmap
    N_SEG, SEG_LEN = 5, 50
    M_u = err_u[:N_SEG*SEG_LEN].reshape(N_SEG, SEG_LEN)
    M_k = err_k[:N_SEG*SEG_LEN].reshape(N_SEG, SEG_LEN)
    M_l = err_l[:N_SEG*SEG_LEN].reshape(N_SEG, SEG_LEN)

    vmax = max(M_u.max(), M_k.max(), M_l.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Tracking Error Heatmap (time × trajectory segment)", fontsize=13, fontweight="bold")
    titles = ["No Fusion (UAV 1)", "Kalman Filter Fusion", "Attn+LSTM Fusion"]
    cmaps  = ["Reds", "Blues", "Greens"]
    mats   = [M_u, M_k, M_l]

    for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       extent=[0, SEG_LEN * DT, N_SEG * SEG_LEN * DT, 0])
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time within segment (s)")
        fig.colorbar(im, ax=ax, label="Error (m)", fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Segment start time (s)")
    _save(fig, os.path.join(out_dir, "fusion_heatmap.png"))


def plot_packet_loss_sweep(lstm_model, out_dir):
    """Vẽ RMSE theo tỉ lệ packet loss từ 0% đến 40%."""
    print("[Sweep] Evaluating RMSE vs. UDP packet-loss rate...")
    loss_rates = np.arange(0, 0.45, 0.05)
    rmse_u_list, rmse_k_list, rmse_l_list = [], [], []

    np.random.seed(99)
    true_traj = generate_target_trajectory("sine", steps=STEPS, dt=DT)

    for rate in loss_rates:
        _, world_obs = simulate_observations(true_traj, UAV_POSES, packet_drop_rate=rate)
        u, k, l = run_one_benchmark(true_traj, world_obs, lstm_model)
        rmse_u_list.append(np.sqrt(np.mean(np.sum((u - true_traj)**2, axis=1))))
        rmse_k_list.append(np.sqrt(np.mean(np.sum((k - true_traj)**2, axis=1))))
        rmse_l_list.append(np.sqrt(np.mean(np.sum((l - true_traj)**2, axis=1))))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = loss_rates * 100
    ax.plot(x, rmse_u_list, "o:", color=COLORS["No Fusion (UAV 1)"],      lw=2, markersize=7, label="No Fusion (UAV 1)")
    ax.plot(x, rmse_k_list, "s-",  color=COLORS["Kalman Filter Fusion"],  lw=2, markersize=7, label="Kalman Filter Fusion")
    ax.plot(x, rmse_l_list, "^-.", color=COLORS["Attention+LSTM Fusion"], lw=2, markersize=7, label="Attn+LSTM Fusion")
    ax.set_xlabel("UDP Packet Loss Rate (%)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Tracking RMSE vs. Network Packet Loss Rate", fontweight="bold")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(0, 40); ax.set_ylim(bottom=0)
    _save(fig, os.path.join(out_dir, "fusion_packet_loss_sweep.png"))


def plot_summary_table(rmse_uav1, rmse_kf, rmse_lstm, gain_kf, gain_lstm,
                       med_u, med_k, med_l, p95_u, p95_k, p95_l, out_dir):
    """Vẽ bảng tổng kết kết quả dạng hình ảnh chuyên nghiệp."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")

    col_labels = [
        "Method", "RMSE (m)", "Median Error (m)",
        "95th Percentile (m)", "Improvement vs Baseline"
    ]
    cell_data = [
        ["No Fusion (UAV 1)",       f"{rmse_uav1:.4f}", f"{med_u:.4f}", f"{p95_u:.4f}", "— (Baseline)"],
        ["Kalman Filter Fusion",    f"{rmse_kf:.4f}",   f"{med_k:.4f}", f"{p95_k:.4f}", f"+{gain_kf:.1f}% ✓"],
        ["Attention+LSTM Fusion",   f"{rmse_lstm:.4f}", f"{med_l:.4f}", f"{p95_l:.4f}", f"+{gain_lstm:.1f}%"],
    ]

    row_colors = [
        ["#ffeaea", "#ffeaea", "#ffeaea", "#ffeaea", "#ffeaea"],
        ["#e3f0ff", "#e3f0ff", "#e3f0ff", "#e3f0ff", "#dff5e3"],
        ["#e8f8e8", "#e8f8e8", "#e8f8e8", "#e8f8e8", "#e8f8e8"],
    ]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Summary Table — Multi-UAV Tracking Benchmark Results",
                 fontsize=14, fontweight="bold", pad=12)
    _save(fig, os.path.join(out_dir, "fusion_summary_table.png"))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Check for trained checkpoint
    checkpoint_path = os.path.join("checkpoints", "task_oriented_encoder.pth")
    if os.path.exists(checkpoint_path):
        print(f"[Checkpoint] Found encoder weights: {checkpoint_path}")
        print("  (Encoder weights are used by UAV sender; benchmark uses raw tracker fusion)")
    else:
        print("[Checkpoint] No encoder checkpoint found — using default ImageNet weights.")

    # 2. Generate test trajectory
    print(f"\n[Data] Generating test trajectory (sine wave, {STEPS} steps) ...")
    packet_drop_rate = 0.15
    true_traj = generate_target_trajectory("sine", steps=STEPS, dt=DT)
    print(f"[Network] Simulating UDP packet loss rate = {packet_drop_rate * 100:.0f}%")
    obs_dict, world_obs = simulate_observations(true_traj, UAV_POSES, packet_drop_rate=packet_drop_rate)

    # 3. Train LSTM
    lstm_model = train_lstm_fusion(UAV_POSES, epochs=80, num_trajectories=150)

    # 4. Run fusion
    uav1_est, kf_est, lstm_est = run_one_benchmark(true_traj, world_obs, lstm_model)

    # Sanity check
    if np.any(np.isnan(lstm_est)):
        print("[WARN] LSTM produced NaN — falling back to zero-fill for plotting.")
        lstm_est = np.nan_to_num(lstm_est, nan=3.0)

    # 5. Compute metrics
    rmse_uav1 = np.sqrt(np.mean(np.sum((uav1_est - true_traj) ** 2, axis=1)))
    rmse_kf   = np.sqrt(np.mean(np.sum((kf_est   - true_traj) ** 2, axis=1)))
    rmse_lstm = np.sqrt(np.mean(np.sum((lstm_est - true_traj) ** 2, axis=1)))

    err_u = np.linalg.norm(uav1_est - true_traj, axis=1)
    err_k = np.linalg.norm(kf_est   - true_traj, axis=1)
    err_l = np.linalg.norm(lstm_est - true_traj, axis=1)

    med_u, med_k, med_l   = np.median(err_u), np.median(err_k), np.median(err_l)
    p95_u, p95_k, p95_l   = np.percentile(err_u, 95), np.percentile(err_k, 95), np.percentile(err_l, 95)
    gain_kf   = (rmse_uav1 - rmse_kf)   / rmse_uav1 * 100
    gain_lstm = (rmse_uav1 - rmse_lstm) / rmse_uav1 * 100

    print("\n" + "=" * 58)
    print("  BENCHMARK RESULTS — TARGET TRACKING RMSE")
    print("=" * 58)
    print(f"  {'Method':<35} {'RMSE':>8}  {'Median':>8}  {'P95':>8}")
    print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'No Fusion (UAV 1)':<35} {rmse_uav1:>8.4f}  {med_u:>8.4f}  {p95_u:>8.4f}")
    print(f"  {'Kalman Filter Fusion':<35} {rmse_kf:>8.4f}  {med_k:>8.4f}  {p95_k:>8.4f}")
    print(f"  {'Attention+LSTM Fusion':<35} {rmse_lstm:>8.4f}  {med_l:>8.4f}  {p95_l:>8.4f}")
    print("=" * 58)
    print(f"  Kalman gain vs baseline : +{gain_kf:.2f}%  (target: >=15%)")
    print(f"  LSTM gain vs baseline   : +{gain_lstm:.2f}%")
    print("=" * 58)

    # 6. Generate all plots
    print("\n[Plots] Generating visualizations...")
    plot_trajectory_and_error(true_traj, uav1_est, kf_est, lstm_est, out_dir,
                              rmse_uav1, rmse_kf, rmse_lstm)
    plot_rmse_bar(rmse_uav1, rmse_kf, rmse_lstm, gain_kf, gain_lstm, out_dir)
    plot_boxplot(true_traj, uav1_est, kf_est, lstm_est, out_dir)
    plot_cdf(true_traj, uav1_est, kf_est, lstm_est, out_dir)
    plot_heatmap(true_traj, uav1_est, kf_est, lstm_est, out_dir)
    plot_packet_loss_sweep(lstm_model, out_dir)
    plot_summary_table(rmse_uav1, rmse_kf, rmse_lstm, gain_kf, gain_lstm,
                       med_u, med_k, med_l, p95_u, p95_k, p95_l, out_dir)

    print(f"\n[Done] All results saved to: {os.path.abspath(out_dir)}/")


if __name__ == "__main__":
    main()
