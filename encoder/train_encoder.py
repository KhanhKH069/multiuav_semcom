"""
train_encoder.py
----------------
Script huấn luyện (fine-tuning) khối TaskOrientedEncoder.
Khối này sử dụng hàm Loss tổng hợp:
1. MSELoss cho Bounding Box Regression.
2. TripletMarginLoss (hoặc CosineEmbeddingLoss) cho Appearance Embedding.

Đây là script cấu trúc khung (scaffold) để chứng minh tính toàn vẹn của dự án.
Cần thay thế `DummyDataset` bằng `UAV123Dataset` thực tế.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from encoder.encoder import TaskOrientedEncoder
from encoder.uav123_dataset import UAV123Dataset

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Duong dan data thuc te
SEQ_ROOT  = "data_seq/UAV123"
ANNO_ROOT = "anno/UAV123"

class DummyUAV123Dataset(Dataset):
    """Giả lập Dataset UAV123 để chạy thử nghiệm training pipeline."""
    def __init__(self, size=100):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Ảnh BGR giả lập đã chuẩn hoá: [3, 224, 224]
        img = torch.randn(3, 224, 224)
        # Bounding box chuẩn hoá (cx, cy, w, h) trong [0, 1]
        bbox = torch.rand(4)
        # Label ID giả lập để dùng cho Triplet Loss (Re-identification)
        target_id = torch.randint(0, 5, (1,)).item()
        return img, bbox, target_id


def get_triplet_loss(embeddings, labels, margin=1.0):
    """
    Tinh toan Triplet Loss don gian tren batch.
    Luan luon tra ve tensor de backward() va float() luon hoat dong.
    """
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    losses = []  # Gom tat ca triplet loss vao list, tranh in-place op

    batch_size = embeddings.size(0)
    for i in range(batch_size):
        anchor = embeddings[i]
        label  = labels[i]

        pos_idx = (labels == label).nonzero().view(-1)
        pos_idx = pos_idx[pos_idx != i]
        if len(pos_idx) == 0:
            continue
        positive = embeddings[pos_idx[0]]

        neg_idx = (labels != label).nonzero().view(-1)
        if len(neg_idx) == 0:
            continue
        negative = embeddings[neg_idx[0]]

        losses.append(triplet_loss_fn(anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)))

    if not losses:
        # Tra ve tensor 0 van co grad_fn de .backward() khong loi
        return (embeddings * 0).sum()

    return torch.stack(losses).mean()



def train():
    print(f"Bắt đầu huấn luyện TaskOrientedEncoder trên thiết bị: {DEVICE}")
    
    # 1. Chuẩn bị DataLoader
    try:
        dataset = UAV123Dataset(seq_root=SEQ_ROOT, anno_root=ANNO_ROOT)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[HINT] Kiem tra lai duong dan SEQ_ROOT va ANNO_ROOT trong train_encoder.py")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Khởi tạo Mô hình và Optimizer
    model = TaskOrientedEncoder(pretrained=True, freeze_backbone=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Hàm Loss cho phần Bounding Box (Regression)
    bbox_loss_fn = nn.MSELoss()
    
    # 3. Vòng lặp huấn luyện (Training Loop)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_bbox_loss = 0.0
        total_embed_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", unit="batch")
        for batch_idx, (imgs, bboxes, labels) in enumerate(pbar):
            imgs, bboxes, labels = imgs.to(DEVICE), bboxes.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(imgs) # Kích thước [B, 132]
            
            # Tách Bbox (4) và Embedding (128)
            pred_bboxes = outputs[:, :4]
            pred_embeddings = outputs[:, 4:]
            
            # Tính hàm Loss
            bbox_loss = bbox_loss_fn(pred_bboxes, bboxes)
            embed_loss = get_triplet_loss(pred_embeddings, labels)
            
            # Trọng số kết hợp hai loss (tuỳ chỉnh theo thực nghiệm)
            loss = bbox_loss + 0.1 * embed_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bbox_loss += bbox_loss.item()
            total_embed_loss += float(embed_loss.detach())
            
            # Cập nhật thông tin loss trên progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'bbox': f"{bbox_loss.item():.4f}",
                'embed': f"{embed_loss.item():.4f}"
            })
            
        # In tổng kết epoch bằng tqdm.write để tránh làm vỡ giao diện progress bar
        tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}] - "
                   f"Loss: {total_loss/len(dataloader):.4f} "
                   f"(Bbox: {total_bbox_loss/len(dataloader):.4f}, "
                   f"Embed: {total_embed_loss/len(dataloader):.4f})")
              
    print("Huan luyen hoan tat. Dang luu mo hinh...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/task_oriented_encoder.pth")
    print("Da luu trong so tai checkpoints/task_oriented_encoder.pth")

if __name__ == "__main__":
    train()
