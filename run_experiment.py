"""
run_experiment.py
-----------------
Chay toan bo pipeline trong 1 lenh:
  1. Sweep lambda (train encoder voi nhieu gia tri lambda)
  2. Danh gia tat ca checkpoint -> tinh 5 metrics
  3. Xuat bang ket qua (CSV + PNG)

Cach chay:
  python run_experiment.py                   # mac dinh: 2 epoch, 7 gia tri lambda
  python run_experiment.py --epochs 5        # nhieu epoch hon
  python run_experiment.py --epochs 2 --lambdas 0 0.1 1.0   # thu nhanh 3 gia tri
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import io
# Reconfigure stdout sang UTF-8 de in Unicode tren Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import time

# ─── Parse args truoc khi import bat ky thu gi nang ──────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Full pipeline: Lambda Sweep -> Eval -> Table"
    )
    p.add_argument(
        "--epochs", type=int, default=2,
        help="So epoch moi run (mac dinh: 2)"
    )
    p.add_argument(
        "--lambdas", nargs="+", type=float,
        default=[0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        metavar="LAM",
        help="Danh sach gia tri lambda (mac dinh: 0 0.01 0.05 0.1 0.2 0.5 1.0)"
    )
    p.add_argument(
        "--sweep-dir", default="results/lambda_sweep",
        help="Thu muc luu ket qua sweep (mac dinh: results/lambda_sweep)"
    )
    p.add_argument(
        "--eval-dir", default="results/eval",
        help="Thu muc luu ket qua eval (mac dinh: results/eval)"
    )
    p.add_argument(
        "--max-frames", type=int, default=300, metavar="N",
        help="Gioi han so frame/sequence de chay nhanh hon (mac dinh: 300; dung None=het)"
    )
    return p.parse_args()


def banner(text: str):
    w = 65
    print("\n" + "=" * w)
    print(f"  {text}")
    print("=" * w)


def main():
    args = parse_args()

    t_start = time.time()

    w = 65
    sep = '+' + '-' * (w - 2) + '+'
    print(sep)
    print('| FULL PIPELINE: Lambda Sweep + Evaluation' + ' ' * (w - 44) + '|')
    print(sep)
    print(f'|  Epochs per run : {args.epochs:<{w-22}}|')
    print(f'|  Lambda values  : {str(args.lambdas):<{w-22}}|')
    print(f'|  Max frames/seq : {args.max_frames:<{w-22}}|')
    print(f'|  Sweep output   : {args.sweep_dir:<{w-22}}|')
    print(f'|  Eval output    : {args.eval_dir:<{w-22}}|')
    print(sep)

    # ─── Phan 1: Import nang ─────────────────────────────────────────────────
    import numpy as np
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else ''
    print(f"[Device] {DEVICE}" + (f" - {gpu_name}" if gpu_name else ""))

    # Import cac module noi bo
    import encoder.sweep_lambda  as _sweep_mod
    import encoder.eval_metrics  as _eval_mod
    from encoder.sweep_lambda import (
        BalancedBatchSampler, run_one_lambda,
        plot_comparison, print_summary,
        SEQ_ROOT, ANNO_ROOT, N_CLASSES, N_SAMPLES,
    )
    from encoder.uav123_dataset import UAV123Dataset
    from encoder.eval_metrics import (
        build_val_loader, evaluate_one, load_model,
        print_table, save_csv, save_png_table,
    )
    from torch.utils.data import DataLoader

    # Ep buoc cac module su dung cung DEVICE (tranh bug module-level DEVICE = cpu)
    _sweep_mod.DEVICE = DEVICE
    _eval_mod.DEVICE  = DEVICE
    print(f"[Device] sweep_lambda.DEVICE = {_sweep_mod.DEVICE}")
    print(f"[Device] eval_metrics.DEVICE = {_eval_mod.DEVICE}")

    # ═════════════════════════════════════════════════════════════════════════
    # BUOC 1: LAMBDA SWEEP
    # ═════════════════════════════════════════════════════════════════════════
    banner("BUOC 1/2 — Lambda Sweep (Huan luyen)")

    t1 = time.time()

    try:
        all_seqs = sorted(d for d in os.listdir(SEQ_ROOT)
                          if os.path.isdir(os.path.join(SEQ_ROOT, d)))
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not all_seqs:
        print("[ERROR] Khong tim thay sequence nao trong SEQ_ROOT")
        sys.exit(1)

    np.random.seed(42)
    shuffled   = list(all_seqs)
    np.random.shuffle(shuffled)
    split      = int(len(shuffled) * 0.8)
    train_seqs = shuffled[:split]
    val_seqs   = shuffled[split:]

    print(f"[INFO] Dataset: {len(all_seqs)} sequences "
          f"(train={len(train_seqs)}, val={len(val_seqs)})")

    train_ds = UAV123Dataset(seq_root=SEQ_ROOT, anno_root=ANNO_ROOT,
                             allowed_seq_names=train_seqs, max_frames=args.max_frames)
    val_ds   = UAV123Dataset(seq_root=SEQ_ROOT, anno_root=ANNO_ROOT,
                             allowed_seq_names=val_seqs,   max_frames=args.max_frames)

    # num_workers=2 (Windows an toan), pin_memory giup transfer GPU nhanh hon
    N_WORKERS = 2
    PIN_MEM   = (DEVICE.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler(train_ds, N_CLASSES, N_SAMPLES),
        num_workers=N_WORKERS,
        pin_memory=PIN_MEM,
    )
    val_loader_s = DataLoader(
        val_ds,
        batch_sampler=BalancedBatchSampler(val_ds, N_CLASSES, N_SAMPLES),
        num_workers=N_WORKERS,
        pin_memory=PIN_MEM,
    )

    all_sweep_results = {}
    for lam in args.lambdas:
        lam_str = str(lam).replace(".", "_")
        lam_dir = os.path.join(args.sweep_dir, f"lambda_{lam_str}")
        print(f"\n{'─'*60}")
        print(f">> Lambda = {lam}")
        rows = run_one_lambda(lam, args.epochs, train_loader, val_loader_s, lam_dir)
        all_sweep_results[lam] = rows

    print_summary(all_sweep_results)
    plot_comparison(all_sweep_results, args.sweep_dir)

    t1_elapsed = time.time() - t1
    print(f"\n[OK] Sweep hoan tat trong {t1_elapsed/60:.1f} phut")

    # ═════════════════════════════════════════════════════════════════════════
    # BUOC 2: DANH GIA -> BANG KET QUA
    # ═════════════════════════════════════════════════════════════════════════
    banner("BUOC 2/2 — Danh gia Metrics (5 chi so)")

    t2 = time.time()

    eval_loader = build_val_loader()
    if eval_loader is None or len(eval_loader) == 0:
        print("[ERROR] Val loader rong, kiem tra lai dataset.")
        sys.exit(1)

    result_rows = []

    # Them baseline (ImageNet pretrained, chua fine-tune)
    print("\n>> Baseline (ImageNet, chua fine-tune)")
    model   = load_model(None)
    metrics = evaluate_one(model, eval_loader)
    result_rows.append({"label": "Baseline (no FT)", "metrics": metrics})
    for k, v in metrics.items():
        print(f"     {k}: {v:.4f}")

    # Danh gia tung checkpoint lambda
    for lam in args.lambdas:
        lam_str  = str(lam).replace(".", "_")
        ckpt     = os.path.join(args.sweep_dir, f"lambda_{lam_str}", "best_encoder.pth")
        label    = f"lambda={lam}"
        print(f"\n>> {label}")
        model   = load_model(ckpt)
        metrics = evaluate_one(model, eval_loader)
        result_rows.append({"label": label, "metrics": metrics})
        for k, v in metrics.items():
            vstr = f"{v:.4f}" if not (isinstance(v, float) and __import__('math').isnan(v)) else "N/A"
            print(f"     {k}: {vstr}")

    # Xuat ket qua
    print_table(result_rows)
    os.makedirs(args.eval_dir, exist_ok=True)
    save_csv(result_rows,     os.path.join(args.eval_dir, "eval_metrics.csv"))
    save_png_table(result_rows, os.path.join(args.eval_dir, "eval_metrics_table.png"))

    t2_elapsed = time.time() - t2
    print(f"\n[OK] Eval hoan tat trong {t2_elapsed/60:.1f} phut")

    # ═════════════════════════════════════════════════════════════════════════
    # TONG KET
    # ═════════════════════════════════════════════════════════════════════════
    total = time.time() - t_start
    w = 65
    sep = '+' + '-' * (w - 2) + '+'
    print()
    print(sep)
    print('| PIPELINE HOAN TAT' + ' ' * (w - 20) + '|')
    print(sep)
    print(f'|  Tong thoi gian  : {f"{total/60:.1f} phut":<{w-22}}|')
    print(f'|  Sweep results   : {args.sweep_dir:<{w-22}}|')
    csv_out = os.path.join(args.eval_dir, "eval_metrics.csv")
    png_out = os.path.join(args.eval_dir, "eval_metrics_table.png")
    cmp_out = os.path.join(args.sweep_dir, "comparison.png")
    print(f'|  Eval CSV        : {csv_out:<{w-22}}|')
    print(f'|  Eval PNG table  : {png_out:<{w-22}}|')
    print(f'|  Comparison plot : {cmp_out:<{w-22}}|')
    print(sep)
    print()


if __name__ == "__main__":
    main()
