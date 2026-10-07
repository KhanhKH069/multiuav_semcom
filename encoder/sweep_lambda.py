"""
sweep_lambda.py
---------------
Chạy thực nghiệm sweep siêu tham số λ (lambda) kiểm soát tỷ trọng
của Triplet Embedding Loss trong hàm mục tiêu huấn luyện:

    Loss = bbox_loss + λ * embed_loss

Các giá trị λ được thử: 0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0

Kết quả mỗi run (train/val loss theo epoch) được lưu tại:
    results/lambda_sweep/lambda_<value>/metrics.csv
    results/lambda_sweep/lambda_<value>/best_encoder.pth
    results/lambda_sweep/comparison.png

Cách chạy:
    python -m encoder.sweep_lambda
    python -m encoder.sweep_lambda --epochs 3 --lambdas 0 0.1 1.0
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from encoder.encoder import TaskOrientedEncoder
from encoder.uav123_dataset import UAV123Dataset

# --- Cau hinh mac dinh ---
DEFAULT_LAMBDAS = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_EPOCHS  = 5          # nho de tham do nhanh
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-4
MAX_GRAD_NORM   = 1.0
N_CLASSES       = 4
N_SAMPLES       = 4

SEQ_ROOT  = "data_seq/UAV123"
ANNO_ROOT = "anno/UAV123"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Triplet Loss ---
def get_triplet_loss(embeddings, labels, margin=1.0):
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    losses = []
    batch_size = embeddings.size(0)
    for i in range(batch_size):
        anchor  = embeddings[i]
        label   = labels[i]
        pos_idx = (labels == label).nonzero().view(-1)
        pos_idx = pos_idx[pos_idx != i]
        if len(pos_idx) == 0:
            continue
        positive = embeddings[pos_idx[0]]
        neg_idx  = (labels != label).nonzero().view(-1)
        if len(neg_idx) == 0:
            continue
        negative = embeddings[neg_idx[0]]
        losses.append(triplet_loss_fn(
            anchor.unsqueeze(0), positive.unsqueeze(0), negative.unsqueeze(0)
        ))
    if not losses:
        return (embeddings * 0).sum()
    return torch.stack(losses).mean()


# --- BalancedBatchSampler ---
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset   = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.label_to_indices = {}
        for idx in range(len(dataset)):
            label = dataset.samples[idx][2]
            self.label_to_indices.setdefault(label, []).append(idx)
        self.labels = list(self.label_to_indices.keys())

    def __iter__(self):
        lti = {l: list(idxs) for l, idxs in self.label_to_indices.items()}
        for l in lti:
            np.random.shuffle(lti[l])
        active = [l for l in self.labels if len(lti[l]) >= self.n_samples]
        while len(active) >= self.n_classes:
            chosen = np.random.choice(active, self.n_classes, replace=False)
            batch  = []
            for l in chosen:
                for _ in range(self.n_samples):
                    batch.append(lti[l].pop(0))
            active = [l for l in active if len(lti[l]) >= self.n_samples]
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        num_batches = 0
        lengths = [len(v) for v in self.label_to_indices.values()
                   if len(v) >= self.n_samples]
        while len(lengths) >= self.n_classes:
            lengths.sort(reverse=True)
            for i in range(self.n_classes):
                lengths[i] -= self.n_samples
            lengths = [l for l in lengths if l >= self.n_samples]
            num_batches += 1
        return num_batches


# --- Mot run cho 1 gia tri lambda ---
def run_one_lambda(lam, epochs, train_loader, val_loader, out_dir):
    """Huan luyen encoder voi loss = bbox_loss + lam * embed_loss."""
    model       = TaskOrientedEncoder(pretrained=True, freeze_backbone=False).to(DEVICE)
    optimizer   = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler   = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    bbox_loss_fn = nn.MSELoss()

    os.makedirs(out_dir, exist_ok=True)
    metrics_rows = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        model.train()
        t_loss = t_bbox = t_embed = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"  lam={lam} | Epoch [{epoch+1}/{epochs}] Train",
            unit="batch", leave=False,
        )
        for imgs, bboxes, labels in pbar:
            imgs, bboxes, labels = imgs.to(DEVICE), bboxes.to(DEVICE), labels.to(DEVICE)
            out       = model(imgs)
            bl        = bbox_loss_fn(out[:, :4], bboxes)
            el        = get_triplet_loss(out[:, 4:], labels)
            loss      = bl + lam * el
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            t_loss  += loss.item()
            t_bbox  += bl.item()
            t_embed += float(el.detach())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n_train   = max(len(train_loader), 1)
        avg_train = t_loss  / n_train
        avg_tbbox = t_bbox  / n_train
        avg_temb  = t_embed / n_train

        # Val
        model.eval()
        v_loss = v_bbox = v_embed = 0.0
        with torch.no_grad():
            for imgs, bboxes, labels in val_loader:
                imgs, bboxes, labels = imgs.to(DEVICE), bboxes.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                bl   = bbox_loss_fn(out[:, :4], bboxes)
                el   = get_triplet_loss(out[:, 4:], labels)
                loss = bl + lam * el
                v_loss  += loss.item()
                v_bbox  += bl.item()
                v_embed += float(el.detach())

        n_val    = max(len(val_loader), 1)
        avg_val  = v_loss  / n_val
        avg_vbbox= v_bbox  / n_val
        avg_vemb = v_embed / n_val

        row = dict(
            epoch=epoch+1,
            train_loss=round(avg_train, 6), train_bbox=round(avg_tbbox, 6), train_embed=round(avg_temb, 6),
            val_loss=round(avg_val, 6),     val_bbox=round(avg_vbbox, 6),   val_embed=round(avg_vemb, 6),
        )
        metrics_rows.append(row)

        tqdm.write(
            f"  lam={lam} | E[{epoch+1}/{epochs}] "
            f"train={avg_train:.4f}(bbox={avg_tbbox:.4f},emb={avg_temb:.4f}) | "
            f"val={avg_val:.4f}(bbox={avg_vbbox:.4f},emb={avg_vemb:.4f})"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(out_dir, "best_encoder.pth"))
            tqdm.write(f"    [OK] Checkpoint cap nhat (val={best_val_loss:.4f})")

        scheduler.step()

    # Luu CSV
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"  -> Da luu metrics: {csv_path}")
    return metrics_rows


# --- Ve bieu do so sanh ---
def plot_comparison(all_results, out_dir):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[WARN] matplotlib chua cai -- bo qua ve bieu do.")
        return

    lam_list = list(all_results.keys())
    colors   = [cm.tab10(i / max(len(lam_list), 1)) for i in range(len(lam_list))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Lambda Sweep -- loss = bbox_loss + lambda*embed_loss", fontsize=13)

    for ax, title, key in zip(
        axes,
        ["Total Val Loss", "BBox Val Loss", "Embed Val Loss"],
        ["val_loss",       "val_bbox",      "val_embed"],
    ):
        for color, lam in zip(colors, lam_list):
            rows   = all_results[lam]
            epochs = [r["epoch"] for r in rows]
            vals   = [r[key]     for r in rows]
            ax.plot(epochs, vals, marker="o", label=f"lam={lam}", color=color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(out_dir, "comparison.png")
    plt.savefig(png_path, dpi=120)
    plt.close()
    print(f"  -> Da luu bieu do: {png_path}")


# --- Tong ket ---
def print_summary(all_results):
    print("\n" + "=" * 70)
    print(f"{'lambda':>8} | {'Best Val Loss':>14} | {'BBox':>12} | {'Embed':>12}")
    print("-" * 70)
    for lam, rows in all_results.items():
        best = min(rows, key=lambda r: r["val_loss"])
        print(f"{lam:>8} | {best['val_loss']:>14.6f} | {best['val_bbox']:>12.6f} | {best['val_embed']:>12.6f}")
    print("=" * 70)


# --- Main ---
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep lambda cho ham muc tieu: loss = bbox_loss + lambda*embed_loss"
    )
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS, metavar="LAM")
    parser.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--out-dir", default="results/lambda_sweep")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Lambda Sweep -- Encoder Training")
    print(f"  Device  : {DEVICE}")
    print(f"  Lambdas : {args.lambdas}")
    print(f"  Epochs  : {args.epochs} moi run")
    print(f"  Out dir : {args.out_dir}")
    print(f"{'='*60}\n")

    try:
        all_seqs = sorted(d for d in os.listdir(SEQ_ROOT)
                          if os.path.isdir(os.path.join(SEQ_ROOT, d)))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    if not all_seqs:
        print("[ERROR] Khong tim thay sequence nao trong SEQ_ROOT")
        return

    np.random.seed(42)
    shuffled   = list(all_seqs)
    np.random.shuffle(shuffled)
    split      = int(len(shuffled) * 0.8)
    train_seqs = shuffled[:split]
    val_seqs   = shuffled[split:]

    print(f"[INFO] Dataset: {len(all_seqs)} sequences (train={len(train_seqs)}, val={len(val_seqs)})")

    train_ds  = UAV123Dataset(seq_root=SEQ_ROOT, anno_root=ANNO_ROOT, allowed_seq_names=train_seqs)
    val_ds    = UAV123Dataset(seq_root=SEQ_ROOT, anno_root=ANNO_ROOT, allowed_seq_names=val_seqs)

    train_loader = DataLoader(train_ds, batch_sampler=BalancedBatchSampler(train_ds, N_CLASSES, N_SAMPLES))
    val_loader   = DataLoader(val_ds,   batch_sampler=BalancedBatchSampler(val_ds,   N_CLASSES, N_SAMPLES))

    all_results = {}
    for lam in args.lambdas:
        lam_str = str(lam).replace(".", "_")
        lam_dir = os.path.join(args.out_dir, f"lambda_{lam_str}")
        print(f"\n{'─'*60}")
        print(f">> Bat dau run lambda = {lam}")
        rows = run_one_lambda(lam, args.epochs, train_loader, val_loader, lam_dir)
        all_results[lam] = rows

    print_summary(all_results)
    plot_comparison(all_results, args.out_dir)
    print(f"\nHoan tat! Ket qua tai: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
