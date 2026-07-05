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



class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    BatchSampler để đảm bảo mỗi batch chứa P classes (sequences) và K samples mỗi class.
    Batch size = P * K (ví dụ: 4 * 4 = 16).
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        
        # Nhóm các index theo sequence ID (label) để tránh việc load ảnh chậm
        self.label_to_indices = {}
        for idx in range(len(dataset)):
            label = dataset.samples[idx][2]
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        
    def __iter__(self):
        label_to_indices_copy = {
            label: list(indices) for label, indices in self.label_to_indices.items()
        }
        for label in label_to_indices_copy:
            np.random.shuffle(label_to_indices_copy[label])
            
        # Chỉ giữ lại các label có đủ n_samples
        active_labels = [l for l in self.labels if len(label_to_indices_copy[l]) >= self.n_samples]
        
        while len(active_labels) >= self.n_classes:
            selected_labels = np.random.choice(active_labels, self.n_classes, replace=False)
            batch = []
            for label in selected_labels:
                for _ in range(self.n_samples):
                    batch.append(label_to_indices_copy[label].pop(0))
            
            # Loại bỏ các label không còn đủ n_samples
            for label in selected_labels:
                if len(label_to_indices_copy[label]) < self.n_samples:
                    active_labels.remove(label)
                    
            np.random.shuffle(batch)
            yield batch
            
    def __len__(self):
        num_batches = 0
        lengths = [len(indices) for indices in self.label_to_indices.values() if len(indices) >= self.n_samples]
        while len(lengths) >= self.n_classes:
            lengths.sort(reverse=True)
            for i in range(self.n_classes):
                lengths[i] -= self.n_samples
            lengths = [l for l in lengths if l >= self.n_samples]
            num_batches += 1
        return num_batches


def train():
    print(f"Bắt đầu huấn luyện TaskOrientedEncoder trên thiết bị: {DEVICE}")
    
    # 0. Phân chia Train/Val ở mức sequence-level
    try:
        all_seqs = sorted(
            d for d in os.listdir(SEQ_ROOT)
            if os.path.isdir(os.path.join(SEQ_ROOT, d))
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[HINT] Kiem tra lai duong dan SEQ_ROOT va ANNO_ROOT trong train_encoder.py")
        return

    if not all_seqs:
        print("[ERROR] Khong tim thay sequence nao trong SEQ_ROOT")
        return

    # Chia 80% train, 20% val
    np.random.seed(42)
    shuffled_seqs = list(all_seqs)
    np.random.shuffle(shuffled_seqs)
    split_idx = int(len(shuffled_seqs) * 0.8)
    train_seqs = shuffled_seqs[:split_idx]
    val_seqs = shuffled_seqs[split_idx:]
    
    print(f"[INFO] Tổng số sequences trong dataset: {len(all_seqs)}")
    print(f"[INFO] Train sequences ({len(train_seqs)}): {train_seqs[:5]} ...")
    print(f"[INFO] Val sequences ({len(val_seqs)}): {val_seqs[:5]} ...")
    
    # 1. Chuẩn bị DataLoader
    train_dataset = UAV123Dataset(seq_root=SEQ_ROOT, anno_root=ANNO_ROOT, allowed_seq_names=train_seqs)
    val_dataset = UAV123Dataset(seq_root=SEQ_ROOT, anno_root=ANNO_ROOT, allowed_seq_names=val_seqs)
    
    # Dùng BalancedBatchSampler: P=4 classes, K=4 samples per class -> Batch Size = 16
    train_sampler = BalancedBatchSampler(train_dataset, n_classes=4, n_samples=4)
    val_sampler = BalancedBatchSampler(val_dataset, n_classes=4, n_samples=4)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    
    # 2. Khởi tạo Mô hình và Optimizer
    model = TaskOrientedEncoder(pretrained=True, freeze_backbone=False).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Hàm Loss cho phần Bounding Box (Regression)
    bbox_loss_fn = nn.MSELoss()
    
    # 3. Vòng lặp huấn luyện (Training Loop)
    for epoch in range(EPOCHS):
        # 3.1 Training Phase
        model.train()
        total_loss = 0.0
        total_bbox_loss = 0.0
        total_embed_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train", unit="batch")
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
            
            # Trọng số kết hợp hai loss
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
            
        tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {total_loss/len(train_loader):.4f} "
                   f"(Bbox: {total_bbox_loss/len(train_loader):.4f}, "
                   f"Embed: {total_embed_loss/len(train_loader):.4f})")
        
        # 3.2 Validation Phase
        model.eval()
        val_loss = 0.0
        val_bbox_loss = 0.0
        val_embed_loss = 0.0
        
        with torch.no_grad():
            for imgs, bboxes, labels in val_loader:
                imgs, bboxes, labels = imgs.to(DEVICE), bboxes.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass
                outputs = model(imgs)
                pred_bboxes = outputs[:, :4]
                pred_embeddings = outputs[:, 4:]
                
                # Tính hàm Loss
                bbox_loss = bbox_loss_fn(pred_bboxes, bboxes)
                embed_loss = get_triplet_loss(pred_embeddings, labels)
                loss = bbox_loss + 0.1 * embed_loss
                
                val_loss += loss.item()
                val_bbox_loss += bbox_loss.item()
                val_embed_loss += float(embed_loss.detach())
                
        tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}] - Val Loss:   {val_loss/len(val_loader):.4f} "
                   f"(Bbox: {val_bbox_loss/len(val_loader):.4f}, "
                   f"Embed: {val_embed_loss/len(val_loader):.4f})")
        tqdm.write("-" * 60)
              
    print("Huan luyen hoan tat. Dang luu mo hinh...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/task_oriented_encoder.pth")
    print("Da luu trong so tai checkpoints/task_oriented_encoder.pth")

if __name__ == "__main__":
    train()
