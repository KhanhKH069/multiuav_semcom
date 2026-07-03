"""
ground_receiver.py
-------------------
Chạy trên trạm điều khiển mặt đất. Lắng nghe UDP, giải mã message từ nhiều
UAV, và đẩy vào một queue để khối fusion (Kalman Filter) tiêu thụ.

Cũng chịu trách nhiệm log băng thông thực tế (byte/s mỗi UAV) để phục vụ
biểu đồ so sánh trong báo cáo.
"""
from __future__ import annotations

import queue
import socket
import threading
import time
from collections import defaultdict

from network.message import decode_message, MESSAGE_SIZE, SemanticMessage


class GroundReceiver:
    def __init__(self, listen_ip: str = "0.0.0.0", listen_port: int = 9000):
        self.listen_ip = listen_ip
        self.listen_port = listen_port
        self.message_queue: "queue.Queue[SemanticMessage]" = queue.Queue()

        # Thống kê băng thông: uav_id -> (tổng byte, thời điểm bắt đầu đo)
        self._bytes_received: dict[int, int] = defaultdict(int)
        self._start_time: dict[int, float] = {}
        self._lock = threading.Lock()

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = False

    def start(self):
        self._sock.bind((self.listen_ip, self.listen_port))
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print(f"[Ground] Đang lắng nghe UDP tại {self.listen_ip}:{self.listen_port}")

    def stop(self):
        self._running = False
        self._sock.close()

    def _listen_loop(self):
        # Buffer lớn hơn MESSAGE_SIZE một chút để dự phòng
        buf_size = MESSAGE_SIZE + 32
        while self._running:
            try:
                data, addr = self._sock.recvfrom(buf_size)
            except OSError:
                break  # socket đã đóng

            try:
                msg = decode_message(data)
            except ValueError as e:
                print(f"[Ground] Bỏ qua message lỗi từ {addr}: {e}")
                continue

            with self._lock:
                self._bytes_received[msg.uav_id] += len(data)
                self._start_time.setdefault(msg.uav_id, time.time())

            self.message_queue.put(msg)

    def get_bandwidth_kbps(self) -> dict[int, float]:
        """Băng thông trung bình (kbps) mỗi UAV kể từ khi nhận message đầu tiên."""
        now = time.time()
        result = {}
        with self._lock:
            for uav_id, total_bytes in self._bytes_received.items():
                elapsed = max(now - self._start_time[uav_id], 1e-6)
                result[uav_id] = (total_bytes * 8 / 1000) / elapsed  # kbps
        return result


if __name__ == "__main__":
    receiver = GroundReceiver(listen_port=9000)
    receiver.start()

    try:
        while True:
            time.sleep(2.0)
            bw = receiver.get_bandwidth_kbps()
            if bw:
                summary = ", ".join(f"UAV {k}: {v:.3f} kbps" for k, v in sorted(bw.items()))
                print(f"[Ground] Băng thông hiện tại — {summary}")
            print(f"[Ground] Số message đang chờ xử lý trong queue: {receiver.message_queue.qsize()}")
    except KeyboardInterrupt:
        receiver.stop()
        print("[Ground] Đã dừng.")
