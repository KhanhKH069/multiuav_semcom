"""
encoder.py
----------
Khối task-oriented encoder cho mỗi UAV.

Input:  1 khung hình (frame) từ camera Tello (BGR, HxWx3)
Output: vector 132 float32
        [0:4]   -> bounding box chuẩn hoá (cx, cy, w, h), giá trị trong [0, 1]
        [4:132] -> appearance embedding (128 chiều) dùng để đối chiếu
                   mục tiêu giữa các UAV ở bước fusion (re-identification)

Backbone: ResNet18 (pretrained ImageNet) -> bỏ lớp FC cuối,
gắn 2 head riêng: bbox head (regression) và embedding head.

Đây là code khung (scaffold) — cần huấn luyện lại (fine-tune) trên UAV123
trước khi dùng thực tế.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

BBOX_DIM = 4
EMBED_DIM = 128
OUTPUT_DIM = BBOX_DIM + EMBED_DIM  # = 132


class TaskOrientedEncoder(nn.Module):
    """ResNet18 backbone + 2 head: bbox regression & appearance embedding."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Bỏ avgpool + fc gốc, giữ lại phần feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # -> [B, 512, H/32, W/32]
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = 512
        self.bbox_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, BBOX_DIM),
            nn.Sigmoid(),  # ép về [0, 1] vì bbox đã chuẩn hoá theo kích thước ảnh
        )
        self.embed_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, EMBED_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W] đã chuẩn hoá (xem `PREPROCESS` bên dưới)
        return: [B, 132]
        """
        feat = self.backbone(x)          # [B, 512, h, w]
        feat = self.pool(feat).flatten(1)  # [B, 512]

        bbox = self.bbox_head(feat)          # [B, 4]
        embed = self.embed_head(feat)        # [B, 128]
        # L2-normalize embedding để appearance matching ổn định hơn ở bước fusion
        embed = nn.functional.normalize(embed, p=2, dim=1)

        return torch.cat([bbox, embed], dim=1)  # [B, 132]


PREPROCESS = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class RealtimeEncoderRunner:
    """Wrapper tiện dùng trực tiếp trên luồng video Tello (BGR numpy frame)."""

    def __init__(self, weights_path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TaskOrientedEncoder(pretrained=(weights_path is None)).to(self.device)
        if weights_path:
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def encode_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: HxWx3 uint8 (OpenCV). Trả về vector 132 float32."""
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        tensor = PREPROCESS(frame_rgb.copy()).unsqueeze(0).to(self.device)
        out = self.model(tensor)  # [1, 132]
        return out.squeeze(0).cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    # Smoke test nhanh với ảnh giả lập
    runner = RealtimeEncoderRunner()
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    vec = runner.encode_frame(dummy_frame)
    print(f"Output shape: {vec.shape}, dtype: {vec.dtype}")
    assert vec.shape == (OUTPUT_DIM,)
    print("Encoder OK — bbox:", vec[:4], "| embedding[:5]:", vec[4:9])
