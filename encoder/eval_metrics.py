"""
eval_metrics.py
---------------
Danh gia encoder sau khi huan luyen, tinh 5 chi so:
  1. BBox IoU (higher is better)
  2. Center Error [pixel] (lower is better)
  3. Re-ID Recall@1 (higher is better)
  4. Association Accuracy - Hungarian (higher is better)
  5. Tracking RMSE [pixel] (lower is better)

Cach chay (chay tu thu muc goc cua du an):

  # Danh gia 1 checkpoint cu the:
  python -m encoder.eval_metrics --checkpoint checkpoints/best_encoder.pth

  # Danh gia nhieu checkpoint cung luc (sau sweep lambda):
  python -m encoder.eval_metrics --sweep-dir results/lambda_sweep

  # Danh gia voi epoch = 0 (khong huan luyen, random weights):
  python -m encoder.eval_metrics --no-checkpoint
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder.encoder import TaskOrientedEncoder
from encoder.uav123_dataset import UAV123Dataset

try:
    from fusion.hungarian_association import HungarianAssociator
    _ASSOC_OK = True
except ImportError:
    _ASSOC_OK = False

# ─── Config ──────────────────────────────────────────────────────────────────
SEQ_ROOT  = "data_seq/UAV123"
ANNO_ROOT = "anno/UAV123"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dung tap val co dinh (seed=42, 20% sequences) - khop voi sweep_lambda
EVAL_SEED     = 42
EVAL_VAL_FRAC = 0.2
MAX_FRAMES    = 200   # gioi han frame/seq de eval nhanh (None = het)
BATCH_SIZE    = 32    # batch lon hon khi eval (khong tinh grad)
IMG_SIZE      = 224   # PREPROCESS resize 224x224


# ─── Metric helpers ───────────────────────────────────────────────────────────

def xywh_to_xyxy(boxes_cxcywh: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """[cx,cy,w,h] (normalized) -> [x1,y1,x2,y2] (pixel)."""
    b = boxes_cxcywh * img_size
    x1 = b[:, 0] - b[:, 2] / 2
    y1 = b[:, 1] - b[:, 3] / 2
    x2 = b[:, 0] + b[:, 2] / 2
    y2 = b[:, 1] + b[:, 3] / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def compute_iou_batch(pred_cxcywh: np.ndarray, gt_cxcywh: np.ndarray) -> np.ndarray:
    """IoU theo tung cap (N,). Input la cx,cy,w,h da normalize."""
    p = xywh_to_xyxy(pred_cxcywh)
    g = xywh_to_xyxy(gt_cxcywh)

    ix1 = np.maximum(p[:, 0], g[:, 0])
    iy1 = np.maximum(p[:, 1], g[:, 1])
    ix2 = np.minimum(p[:, 2], g[:, 2])
    iy2 = np.minimum(p[:, 3], g[:, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)

    area_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    area_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    union  = area_p + area_g - inter + 1e-8

    return inter / union


def compute_center_error(pred_cxcywh: np.ndarray, gt_cxcywh: np.ndarray) -> np.ndarray:
    """Sai so tam (pixel). Input normalize [0,1]."""
    pred_px = pred_cxcywh[:, :2] * IMG_SIZE
    gt_px   = gt_cxcywh[:, :2]   * IMG_SIZE
    return np.sqrt(np.sum((pred_px - gt_px) ** 2, axis=1))


def compute_reid_recall_at_1(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Re-ID Recall@1: moi sample la query, tim 1 neighbor gan nhat (tru ban than),
    kiem tra xem co cung label khong.
    """
    N = len(embeddings)
    if N < 2:
        return float("nan")

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    emb   = embeddings / norms

    sim   = emb @ emb.T                    # [N, N]
    np.fill_diagonal(sim, -1e9)            # loai ban than
    nn_idx = np.argmax(sim, axis=1)        # [N]

    correct = (labels[nn_idx] == labels).sum()
    return float(correct) / N


def compute_association_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """
    Association Accuracy dung Hungarian Associator:
    Chia labels thanh 2 nhom (local / global), chay Hungarian, dem chinh xac.
    """
    if not _ASSOC_OK:
        return float("nan")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    # Chia: 50% sample dau tien cua moi label la 'global', con lai la 'local'
    global_embs, global_labels = [], []
    local_embs,  local_labels  = [], []
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            global_embs.append(embeddings[idx[0]])
            global_labels.append(lab)
            continue
        mid = len(idx) // 2
        global_embs.extend(embeddings[idx[:mid]])
        global_labels.extend([lab] * mid)
        local_embs.extend(embeddings[idx[mid:]])
        local_labels.extend([lab] * (len(idx) - mid))

    if len(local_embs) == 0 or len(global_embs) == 0:
        return float("nan")

    assoc  = HungarianAssociator(threshold=0.5)
    result = assoc.associate(np.array(global_embs), np.array(local_embs))

    correct = 0
    for gi, li in result.matched:
        if global_labels[gi] == local_labels[li]:
            correct += 1
    total = len(local_embs)
    return float(correct) / total if total > 0 else float("nan")


def compute_tracking_rmse(pred_cxcywh: np.ndarray, gt_cxcywh: np.ndarray) -> float:
    """
    Tracking RMSE [pixel] cua tam bounding box theo toan bo sequence.
    """
    pred_px = pred_cxcywh[:, :2] * IMG_SIZE
    gt_px   = gt_cxcywh[:, :2]   * IMG_SIZE
    mse     = np.mean(np.sum((pred_px - gt_px) ** 2, axis=1))
    return float(np.sqrt(mse))


# ─── Eval chính ──────────────────────────────────────────────────────────────

def build_val_loader():
    """Tao val loader cung cap voi sweep_lambda."""
    try:
        all_seqs = sorted(d for d in os.listdir(SEQ_ROOT)
                          if os.path.isdir(os.path.join(SEQ_ROOT, d)))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None

    rng = np.random.default_rng(EVAL_SEED)
    shuffled = list(all_seqs)
    rng.shuffle(shuffled)
    split    = int(len(shuffled) * (1 - EVAL_VAL_FRAC))
    val_seqs = shuffled[split:]

    print(f"[INFO] Val set: {len(val_seqs)}/{len(all_seqs)} sequences")

    val_ds = UAV123Dataset(
        seq_root=SEQ_ROOT,
        anno_root=ANNO_ROOT,
        allowed_seq_names=val_seqs,
        max_frames=MAX_FRAMES,
    )
    loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    return loader


def evaluate_one(model: nn.Module, loader: DataLoader) -> dict:
    """
    Chay inference tren loader, tra ve dict chua 5 metrics.
    """
    model.eval()

    all_pred_bbox  = []
    all_gt_bbox    = []
    all_embeddings = []
    all_labels     = []

    with torch.no_grad():
        for imgs, bboxes, labels in tqdm(loader, desc="  Eval", leave=False):
            imgs   = imgs.to(DEVICE)
            out    = model(imgs)          # [B, 132]
            pred_b = out[:, :4].cpu().numpy()
            pred_e = out[:, 4:].cpu().numpy()

            all_pred_bbox.append(pred_b)
            all_gt_bbox.append(bboxes.numpy())
            all_embeddings.append(pred_e)
            all_labels.append(labels.numpy())

    pred_bbox  = np.concatenate(all_pred_bbox,  axis=0)   # [N, 4]
    gt_bbox    = np.concatenate(all_gt_bbox,    axis=0)   # [N, 4]
    embeddings = np.concatenate(all_embeddings, axis=0)   # [N, 128]
    labels_arr = np.concatenate(all_labels,     axis=0)   # [N]

    ious    = compute_iou_batch(pred_bbox, gt_bbox)
    cerr    = compute_center_error(pred_bbox, gt_bbox)
    recall1 = compute_reid_recall_at_1(embeddings, labels_arr)
    assoc   = compute_association_accuracy(embeddings, labels_arr)
    t_rmse  = compute_tracking_rmse(pred_bbox, gt_bbox)

    return {
        "BBox IoU (↑)":              round(float(np.mean(ious)),   4),
        "Center Error px (↓)":       round(float(np.mean(cerr)),   4),
        "Re-ID Recall@1 (↑)":        round(recall1,                4),
        "Association Acc (↑)":       round(assoc,                  4) if not np.isnan(assoc) else float("nan"),
        "Tracking RMSE px (↓)":      round(t_rmse,                 4),
    }


def load_model(checkpoint_path: str | None) -> nn.Module:
    model = TaskOrientedEncoder(pretrained=(checkpoint_path is None), freeze_backbone=False).to(DEVICE)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"  [OK] Loaded checkpoint: {checkpoint_path}")
    elif checkpoint_path:
        print(f"  [WARN] Checkpoint khong tim thay: {checkpoint_path}")
    else:
        print(f"  [INFO] Dung ImageNet pretrained weights (chua fine-tune)")
    return model


def print_table(rows: list[dict]):
    """In bang ket qua ra terminal."""
    if not rows:
        return

    col_names = ["Label"] + list(rows[0]["metrics"].keys())
    col_widths = [max(len(c), 20) for c in col_names]

    # Tinh do rong cho moi cot
    for row in rows:
        col_widths[0] = max(col_widths[0], len(row["label"]))
        for i, v in enumerate(row["metrics"].values()):
            col_widths[i+1] = max(col_widths[i+1], len(f"{v:.4f}"))

    sep  = "+" + "+".join("-" * (w+2) for w in col_widths) + "+"
    header = "|" + "|".join(f" {c:<{col_widths[i]}} " for i, c in enumerate(col_names)) + "|"

    print("\n" + sep)
    print(header)
    print(sep)
    for row in rows:
        vals = [row["label"]] + [f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
                                  for v in row["metrics"].values()]
        line = "|" + "|".join(f" {v:<{col_widths[i]}} " for i, v in enumerate(vals)) + "|"
        print(line)
    print(sep + "\n")


def save_csv(rows: list[dict], path: str):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["label"] + list(rows[0]["metrics"].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            r = {"label": row["label"]}
            r.update(row["metrics"])
            writer.writerow(r)
    print(f"[OK] CSV saved: {path}")


def save_png_table(rows: list[dict], path: str):
    """Xuat bang ket qua thanh file PNG dep."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({"font.family": "DejaVu Sans"})
    except ImportError:
        print("[WARN] matplotlib chua cai, bo qua xuat PNG.")
        return

    col_names = ["Lambda / Label"] + list(rows[0]["metrics"].keys())
    cell_data = []
    for row in rows:
        cells = [row["label"]]
        for v in row["metrics"].values():
            cells.append(f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A")
        cell_data.append(cells)

    fig, ax = plt.subplots(figsize=(14, 1 + 0.5 * len(rows)))
    ax.axis("off")

    # Mau header
    header_colors = [["#1a1a2e"] * len(col_names)]
    row_colors    = [["#f0f4f8" if i % 2 == 0 else "#ffffff"] * len(col_names)
                     for i in range(len(rows))]

    table = ax.table(
        cellText=cell_data,
        colLabels=col_names,
        cellLoc="center",
        loc="center",
        colColours=["#1a1a2e"] * len(col_names),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # To mau header text
    for j in range(len(col_names)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    # To mau xen ke
    for i in range(len(rows)):
        bg = "#eef2f7" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_names)):
            table[i+1, j].set_facecolor(bg)

    # To mau cot dau (label)
    for i in range(1, len(rows)+1):
        table[i, 0].set_facecolor("#0f3460")
        table[i, 0].set_text_props(color="white", fontweight="bold")

    plt.title(
        "Lambda Sweep — Evaluation Metrics\n"
        "loss = bbox_loss + λ·embed_loss  |  val set (UAV123)",
        fontsize=11, pad=12, color="#1a1a2e"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] PNG table saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Danh gia encoder: BBox IoU, Center Error, Re-ID Recall@1, Association Acc, Tracking RMSE"
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--checkpoint", metavar="PATH",
        help="Duong dan toi file .pth de danh gia (vi du: checkpoints/best_encoder.pth)"
    )
    grp.add_argument(
        "--sweep-dir", metavar="DIR", default=None,
        help="Thu muc ket qua cua sweep_lambda (se tu dong tim tat ca best_encoder.pth)"
    )
    grp.add_argument(
        "--no-checkpoint", action="store_true",
        help="Danh gia voi ImageNet pretrained weights (baseline, khong fine-tune)"
    )
    p.add_argument(
        "--out-dir", default="results/eval",
        help="Thu muc luu ket qua CSV va PNG (mac dinh: results/eval)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*65}")
    print("Encoder Evaluation — 5 Metrics")
    print(f"  Device : {DEVICE}")
    print(f"{'='*65}\n")

    # Build loader
    loader = build_val_loader()
    if loader is None or len(loader) == 0:
        print("[ERROR] Val loader rong, kiem tra lai dataset.")
        return

    # Xac dinh danh sach checkpoints can chay
    eval_jobs = []  # list of (label, checkpoint_path_or_None)

    if args.sweep_dir:
        # Tim tat ca lambda_*/best_encoder.pth
        pattern = os.path.join(args.sweep_dir, "lambda_*", "best_encoder.pth")
        ckpts   = sorted(glob.glob(pattern))
        if not ckpts:
            print(f"[ERROR] Khong tim thay best_encoder.pth trong: {args.sweep_dir}")
            return
        for ckpt in ckpts:
            lam_folder = os.path.basename(os.path.dirname(ckpt))
            # lambda_0_1 -> 0.1
            label = lam_folder.replace("lambda_", "lambda=").replace("_", ".", 1)
            eval_jobs.append((label, ckpt))
        # Them baseline (chua fine-tune)
        eval_jobs.insert(0, ("Baseline (no FT)", None))

    elif args.checkpoint:
        eval_jobs = [(os.path.basename(args.checkpoint), args.checkpoint)]
    else:
        eval_jobs = [("Baseline (no FT)", None)]

    # Chay danh gia
    result_rows = []
    for label, ckpt in eval_jobs:
        print(f"\n>> {label}")
        model   = load_model(ckpt)
        metrics = evaluate_one(model, loader)
        result_rows.append({"label": label, "metrics": metrics})
        for k, v in metrics.items():
            vstr = f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
            print(f"     {k}: {vstr}")

    # In bang tong hop
    print_table(result_rows)

    # Luu file
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "eval_metrics.csv")
    png_path = os.path.join(args.out_dir, "eval_metrics_table.png")
    save_csv(result_rows, csv_path)
    save_png_table(result_rows, png_path)


if __name__ == "__main__":
    main()
