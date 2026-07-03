"""
uav_sender.py
-------------
Chạy trên (hoặc bên cạnh, nếu xử lý trên laptop nối Tello) mỗi UAV.
Lấy khung hình -> encoder -> đóng gói message -> gửi UDP về ground station.

Cách dùng:
    python uav_sender.py --uav-id 1 --ground-ip 192.168.10.100 --ground-port 9000
"""
from __future__ import annotations

import argparse
import socket
import time

from encoder.encoder import RealtimeEncoderRunner
from network.message import make_message, encode_message




class DummyFrameSource:
    """Placeholder cho luồng video Tello — thay bằng djitellopy trong triển khai thực."""

    def __init__(self):
        import numpy as np
        self._np = np

    def read(self):
        return self._np.random.randint(0, 255, (480, 640, 3), dtype=self._np.uint8)


class TelloFrameSource:
    """Nguồn khung hình từ camera Tello thực tế."""

    def __init__(self):
        try:
            from djitellopy import Tello
        except ImportError as e:
            raise ImportError(
                "Chưa cài đặt thư viện djitellopy! Vui lòng chạy lệnh: "
                "pip install djitellopy opencv-python"
            ) from e

        self.tello = Tello()
        print("[Tello] Đang kết nối tới UAV...")
        self.tello.connect()
        print("[Tello] Đã kết nối. Đang bật video stream...")
        self.tello.streamon()
        self.frame_reader = self.tello.get_frame_read()

    def read(self):
        return self.frame_reader.frame

    def close(self):
        print("[Tello] Tắt video stream...")
        try:
            self.tello.streamoff()
        except Exception:
            pass


def run_sender(uav_id: int, ground_ip: str, ground_port: int,
               weights_path: str | None = None, send_hz: float = 10.0, dummy: bool = True):
    encoder = RealtimeEncoderRunner(weights_path=weights_path)
    
    if dummy:
        frame_source = DummyFrameSource()
        print(f"[UAV {uav_id}] Sử dụng nguồn ảnh giả lập (Dummy).")
    else:
        frame_source = TelloFrameSource()
        print(f"[UAV {uav_id}] Khởi tạo camera Tello thực tế thành công.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    seq = 0
    period = 1.0 / send_hz

    print(f"[UAV {uav_id}] Bắt đầu gửi message tới {ground_ip}:{ground_port} ở {send_hz} Hz")

    try:
        while True:
            t0 = time.time()

            frame = frame_source.read()
            if frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            feature_vec = encoder.encode_frame(frame)
            msg = make_message(uav_id=uav_id, seq=seq, feature_vec=feature_vec)
            packed = encode_message(msg)

            sock.sendto(packed, (ground_ip, ground_port))
            seq += 1

            elapsed = time.time() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print(f"[UAV {uav_id}] Dừng gửi message.")
    finally:
        sock.close()
        if hasattr(frame_source, "close"):
            frame_source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uav-id", type=int, required=True)
    parser.add_argument("--ground-ip", type=str, required=True)
    parser.add_argument("--ground-port", type=int, default=9000)
    parser.add_argument("--weights", type=str, default=None, help="Checkpoint encoder đã fine-tune")
    parser.add_argument("--hz", type=float, default=10.0, help="Tần suất gửi message (Hz)")
    parser.add_argument("--dummy", action="store_true", default=True, help="Chạy chế độ giả lập nguồn video (mặc định bật, dùng --no-dummy để kết nối Tello thật)")
    args = parser.parse_args()

    run_sender(args.uav_id, args.ground_ip, args.ground_port, args.weights, args.hz, dummy=args.dummy)
