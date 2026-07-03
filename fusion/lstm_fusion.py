"""
lstm_fusion.py
--------------
Phương án fusion SO SÁNH (không nằm trên đường găng của PoC): một mạng nhỏ
Attention + LSTM học cách hợp nhất quan sát từ nhiều UAV thành vị trí mục
tiêu, để đối chiếu hiệu năng với Kalman Filter (kalman_fusion.py).

Input mỗi bước thời gian: quan sát từ N_UAV, mỗi quan sát là vector
[px, py, confidence] (có thể mở rộng thêm appearance embedding nếu cần
matching giữa các UAV).
Kiến trúc:
    1) Attention pooling giữa các UAV tại mỗi bước thời gian
       -> vector hợp nhất theo không gian (spatial fusion)
    2) LSTM chạy theo trục thời gian trên chuỗi vector đã hợp nhất
       -> vector hợp nhất theo thời gian (temporal smoothing)
    3) Head tuyến tính -> ước lượng vị trí (px, py)

Quy mô nhỏ (hidden_dim mặc định 32) để phù hợp lượng dữ liệu quỹ đạo
hạn chế từ mô phỏng — theo đúng khuyến nghị giữ phương án này làm thử
nghiệm bổ sung, không phải hạ tầng chính của hệ thống.
"""
from __future__ import annotations

import torch
import torch.nn as nn

OBS_DIM = 3  # px, py, confidence


class UAVAttentionPool(nn.Module):
    """Hợp nhất quan sát từ N UAV tại một bước thời gian bằng attention."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(obs_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        obs:  [B, N_uav, obs_dim] — quan sát từ N UAV tại 1 bước thời gian.
              Với UAV không có quan sát hợp lệ ở bước này (mất gói...), đặt
              hàng tương ứng bằng 0 và mask=False.
        mask: [B, N_uav] bool, True nếu UAV đó có quan sát hợp lệ.
        return: [B, hidden_dim]
        """
        h = torch.tanh(self.proj(obs))            # [B, N, H]
        scores = self.attn_score(h).squeeze(-1)    # [B, N]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)    # [B, N]
        fused = torch.einsum("bn,bnh->bh", weights, h)  # [B, H]
        return fused


class LSTMFusionModel(nn.Module):
    """Attention pooling theo UAV -> LSTM theo thời gian -> vị trí (px, py)."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 32, lstm_layers: int = 1):
        super().__init__()
        self.spatial_pool = UAVAttentionPool(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)  # -> (px, py)

    def forward(self, obs_seq: torch.Tensor, mask_seq: torch.Tensor | None = None) -> torch.Tensor:
        """
        obs_seq:  [B, T, N_uav, obs_dim]
        mask_seq: [B, T, N_uav] bool hoặc None
        return:   [B, T, 2] — quỹ đạo ước lượng theo thời gian
        """
        B, T, N, D = obs_seq.shape
        fused_steps = []
        for t in range(T):
            m = mask_seq[:, t, :] if mask_seq is not None else None
            fused_steps.append(self.spatial_pool(obs_seq[:, t, :, :], m))
        fused_seq = torch.stack(fused_steps, dim=1)  # [B, T, H]

        lstm_out, _ = self.lstm(fused_seq)  # [B, T, H]
        positions = self.head(lstm_out)     # [B, T, 2]
        return positions


def train_step(model: LSTMFusionModel, optimizer: torch.optim.Optimizer,
                obs_seq: torch.Tensor, target_positions: torch.Tensor,
                mask_seq: torch.Tensor | None = None) -> float:
    """Một bước huấn luyện đơn giản (MSE loss). Dùng cho vòng lặp train tự viết."""
    model.train()
    optimizer.zero_grad()
    pred = model(obs_seq, mask_seq)
    loss = nn.functional.mse_loss(pred, target_positions)
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    # Smoke test với dữ liệu giả lập: batch=4, T=20 bước, 3 UAV
    torch.manual_seed(0)
    B, T, N_UAV = 4, 20, 3

    model = LSTMFusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs_seq = torch.rand(B, T, N_UAV, OBS_DIM)
    mask_seq = torch.ones(B, T, N_UAV, dtype=torch.bool)
    target = torch.rand(B, T, 2)

    for epoch in range(5):
        loss = train_step(model, optimizer, obs_seq, target, mask_seq)
        print(f"Epoch {epoch}: loss = {loss:.4f}")

    with torch.no_grad():
        pred = model(obs_seq, mask_seq)
    print(f"Output shape: {pred.shape} (kỳ vọng [B, T, 2] = {(B, T, 2)})")
