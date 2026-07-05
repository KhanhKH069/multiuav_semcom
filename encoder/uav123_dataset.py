"""
uav123_dataset.py  (revised)
----------------------------
Dataset class cho UAV123 voi cau truc thu muc thuc te cua du an:

  data_seq/UAV123/<seq_name>/000001.jpg  ...
  anno/UAV123/<seq_name>[_<part>].txt

Vi mot so sequence bi tach thanh nhieu phan (vd car1_1, car1_2, car1_3),
class nay tu dong quet tat ca file annotation khop voi ten thu muc.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from encoder.encoder import PREPROCESS


class UAV123Dataset(Dataset):
    """
    Dataset cho UAV123 khop voi cau truc thu muc thuc te:
      seq_root  = data_seq/UAV123/<seq_name>/
      anno_root = anno/UAV123/<seq_name>[_suffix].txt

    Tham so:
      seq_root  : Thu muc chua cac sequence (mac dinh: data_seq/UAV123)
      anno_root : Thu muc chua cac file annotation (mac dinh: anno/UAV123)
      max_frames: Gioi han so frame moi sequence (None = lay het).
                  Huu ich khi thu nghiem nhanh.
    """

    def __init__(
        self,
        seq_root: str = "data_seq/UAV123",
        anno_root: str = "anno/UAV123",
        max_frames: int | None = None,
        allowed_seq_names: list[str] | None = None,
    ):
        self.seq_root = seq_root
        self.anno_root = anno_root
        self.samples = []   # list of (img_path, bbox_xywh, target_id)

        if not os.path.exists(seq_root):
            raise FileNotFoundError(
                f"Khong tim thay thu muc sequence tai: {seq_root}"
            )
        if not os.path.exists(anno_root):
            raise FileNotFoundError(
                f"Khong tim thay thu muc annotation tai: {anno_root}"
            )

        # Lay danh sach sequence (cac thu muc con trong seq_root)
        seq_names = sorted(
            d for d in os.listdir(seq_root)
            if os.path.isdir(os.path.join(seq_root, d))
        )
        self.seq_to_id = {name: idx for idx, name in enumerate(seq_names)}

        if allowed_seq_names is not None:
            seq_names = [s for s in seq_names if s in allowed_seq_names]

        print(f"[INFO] Tim thay {len(seq_names)} sequence: {seq_names[:5]} ...")

        total_skipped = 0
        for seq_name in seq_names:
            seq_dir = os.path.join(seq_root, seq_name)
            target_id = self.seq_to_id[seq_name]

            # Tim tat ca file annotation tuong ung: <seq_name>.txt hoac <seq_name>_*.txt
            anno_pattern = os.path.join(anno_root, f"{seq_name}*.txt")
            anno_files = sorted(glob.glob(anno_pattern))

            if not anno_files:
                print(f"[WARN] Khong tim thay annotation cho '{seq_name}', bo qua.")
                total_skipped += 1
                continue

            # Doc danh sach anh trong sequence, sap xep theo ten file
            img_files = sorted(
                f for f in os.listdir(seq_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            if not img_files:
                print(f"[WARN] Khong co anh trong '{seq_dir}', bo qua.")
                total_skipped += 1
                continue

            # Doc va ghep tat ca phan annotation thanh mot danh sach lien tuc
            all_anno_lines = []
            for anno_file in anno_files:
                with open(anno_file, "r") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                all_anno_lines.extend(lines)

            # Khop anh <-> annotation theo thu tu
            num_samples = min(len(img_files), len(all_anno_lines))
            if max_frames is not None:
                num_samples = min(num_samples, max_frames)

            added = 0
            for idx in range(num_samples):
                img_path = os.path.join(seq_dir, img_files[idx])
                line = all_anno_lines[idx]

                # Ho tro ca comma va space separator
                parts = line.split(',') if ',' in line else line.split()
                if len(parts) != 4:
                    continue
                try:
                    bbox = [float(p) for p in parts]  # [x, y, w, h]
                except ValueError:
                    continue

                # Bo qua bbox co w hoac h = 0 (annotation khong hop le)
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                self.samples.append((img_path, bbox, target_id))
                added += 1

        print(
            f"[INFO] Nap thanh cong {len(self.samples)} mau "
            f"tu {len(seq_names) - total_skipped}/{len(seq_names)} sequence."
        )

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, target_id = self.samples[idx]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Khong doc duoc anh: {img_path}")

        height, width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = PREPROCESS(img_rgb)

        # Chuyen [x, y, w, h] -> [cx, cy, w_norm, h_norm] trong [0, 1]
        x, y, w, h = bbox
        cx = max(0.0, min(1.0, (x + w / 2.0) / width))
        cy = max(0.0, min(1.0, (y + h / 2.0) / height))
        w_norm = max(0.0, min(1.0, w / width))
        h_norm = max(0.0, min(1.0, h / height))

        bbox_tensor = torch.tensor([cx, cy, w_norm, h_norm], dtype=torch.float32)
        return img_tensor, bbox_tensor, target_id


# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        ds = UAV123Dataset(max_frames=50)   # thu nhanh 50 frame moi sequence
        print(f"Dataset size: {len(ds)}")
        img, bbox, tid = ds[0]
        print(f"  Image: {img.shape}, BBox: {bbox}, TargetID: {tid}")
    except Exception as e:
        print(f"[ERROR] {e}")
