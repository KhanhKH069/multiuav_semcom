import subprocess
import time
import sys

def main():
    print("🚀 Đang khởi động hệ thống Multi-UAV Semantic Communication (Mô phỏng)...")
    
    processes = []
    
    try:
        print("[1/2] Khởi động Ground Station Dashboard...")
        dashboard_cmd = [sys.executable, "-m", "dashboard.dashboard"]
        p_dash = subprocess.Popen(dashboard_cmd)
        processes.append(p_dash)
        
        time.sleep(2)
        
        print("[2/2] Khởi động 3 UAV Senders giả lập...")
        for uav_id in range(1, 4):
            uav_cmd = [
                sys.executable, "-m", "network.uav_sender",
                "--uav-id", str(uav_id),
                "--ground-ip", "127.0.0.1",
                "--ground-port", "9000",
                "--dummy"
            ]
            p_uav = subprocess.Popen(uav_cmd)
            processes.append(p_uav)
            print(f"  -> Đã bắt đầu UAV {uav_id}")
            time.sleep(0.5)

        print("\n✅ Hệ thống đã chạy. Nhấn Ctrl+C ở terminal này để đóng tất cả.")
        
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n🛑 Đang đóng tất cả các tiến trình...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                p.wait()
        print("Đã thoát hoàn toàn.")

if __name__ == "__main__":
    main()
