"""
main.py
-------
Entry point chính để khởi động toàn bộ hệ thống Multi-UAV Semantic Communication.

Chế độ hoạt động (--mode):
    sim       : Chạy mô phỏng — khởi động dashboard + 3 UAV sender giả lập (mặc định)
    benchmark : Chạy đánh giá hiệu năng fusion và xuất biểu đồ kết quả

Ví dụ sử dụng:
    # Chạy mô phỏng (không cần Tello):
    python main.py

    # Chạy mô phỏng với encoder weights đã huấn luyện:
    python main.py --checkpoint checkpoints/best_encoder.pth

    # Chạy benchmark so sánh Kalman vs LSTM Fusion:
    python main.py --mode benchmark
"""
import argparse
import logging
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multiuav")

def run_simulation(checkpoint: str | None) -> None:
    
    log.info("🚀 Khởi động hệ thống Multi-UAV Semantic Communication (Mô phỏng)...")

    if checkpoint:
        log.info(f"  Encoder weights: {checkpoint}")
    else:
        log.warning("  Không có checkpoint — encoder dùng trọng số ImageNet chưa fine-tune.")

    processes = []

    try:
        log.info("[1/2] Khởi động Ground Station Dashboard...")
        dashboard_cmd = [sys.executable, "-m", "dashboard.dashboard"]
        p_dash = subprocess.Popen(dashboard_cmd)
        processes.append(p_dash)

        time.sleep(2)

        log.info("[2/2] Khởi động 3 UAV Senders giả lập...")
        for uav_id in range(1, 4):
            uav_cmd = [
                sys.executable, "-m", "network.uav_sender",
                "--uav-id", str(uav_id),
                "--ground-ip", "127.0.0.1",
                "--ground-port", "9000",
                "--dummy",
            ]
            if checkpoint:
                uav_cmd += ["--weights", checkpoint]  

            p_uav = subprocess.Popen(uav_cmd)
            processes.append(p_uav)
            log.info(f"  → Đã bắt đầu UAV {uav_id} (PID: {p_uav.pid})")
            time.sleep(0.5)

        log.info("✅ Hệ thống đã chạy. Nhấn Ctrl+C ở terminal này để đóng tất cả.")

        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        log.info("🛑 Đang đóng tất cả các tiến trình...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
        log.info("Đã thoát hoàn toàn.")

def run_benchmark() -> None:
    
    log.info("📊 Khởi động chế độ Benchmark...")
    log.info("  Sẽ sinh quỹ đạo mô phỏng, huấn luyện LSTM và xuất biểu đồ.")

    try:
        import runpy
        runpy.run_module("fusion.benchmark", run_name="__main__")
    except ImportError as e:
        log.error(f"Không import được module benchmark: {e}")
        log.error("Hãy đảm bảo đã cài đủ dependencies: uv sync hoặc pip install scipy torch matplotlib")
        sys.exit(1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-UAV Task-Oriented Semantic Communication System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py                                      # Mô phỏng (không Tello)
  python main.py --checkpoint checkpoints/best_encoder.pth
  python main.py --mode benchmark                     # So sánh Kalman vs LSTM
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["sim", "benchmark"],
        default="sim",
        help="Chế độ chạy: 'sim' = mô phỏng (mặc định), 'benchmark' = đánh giá hiệu năng",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Đường dẫn tới file weights encoder (.pth) đã fine-tune trên UAV123",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    log.info("=" * 55)
    log.info("Multi-UAV Task-Oriented Semantic Communication")
    log.info(f"  Mode: {args.mode}")
    if args.checkpoint:
        log.info(f"  Checkpoint: {args.checkpoint}")
    log.info("=" * 55)

    if args.mode == "benchmark":
        run_benchmark()
    else:
        run_simulation(checkpoint=args.checkpoint)

if __name__ == "__main__":
    main()

