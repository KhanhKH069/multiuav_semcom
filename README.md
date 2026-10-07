# Multi-UAV Task-Oriented Semantic Communication

Hệ thống **Task-Oriented Semantic Communication** cho bài toán giám sát và bám đuổi mục tiêu bằng cụm đa UAV. Thay vì truyền raw video (~50 Mbps/UAV), mỗi UAV chạy một encoder nhẹ ngay tại biên để nén khung hình thành vector đặc trưng ~155 byte, sau đó Ground Station thực hiện liên kết và dung hợp (Hungarian + Kalman) từ nhiều UAV để tái tạo quỹ đạo mục tiêu.

## Cấu trúc thư mục

```
multiuav_semcom/
├── encoder/
│   ├── encoder.py            # TaskOrientedEncoder: ResNet-18 → 132 floats (bbox + embedding)
│   ├── train_encoder.py      # Fine-tune encoder: MSE + Triplet Loss, LR Scheduler, Best Checkpoint
│   └── uav123_dataset.py     # Dataset class cho UAV123 (seq_root / anno_root)
├── network/
│   ├── message.py            # Định dạng message ~155 byte (bbox float16 + embedding int8 lượng tử hoá)
│   ├── uav_sender.py         # Gửi message từ UAV về Ground Station (UDP)
│   └── ground_receiver.py    # UDP server phía Ground Station + đo băng thông
├── fusion/
│   ├── hungarian_association.py  # ⭐ Hungarian Algorithm cho Target Association (Chương 4.1)
│   ├── kalman_fusion.py          # ⭐ Kalman Filter single/multi-target + coordinate_utils tích hợp
│   ├── coordinate_utils.py       # Quy đổi Camera ↔ World (ray casting, FOV Tello 82.6°)
│   ├── lstm_fusion.py            # Phương án so sánh: Attention + LSTM Fusion
│   └── benchmark.py              # Đánh giá RMSE: UAV đơn vs Kalman Fusion vs LSTM Fusion
└── dashboard/
    └── dashboard.py          # Hiển thị real-time: 3 luồng video + quỹ đạo fusion + băng thông
```

## Cài đặt

```bash
# Dùng uv (khuyến nghị — dự án dùng pyproject.toml):
uv sync

# Hoặc pip thủ công:
pip install torch torchvision numpy matplotlib scipy tqdm
# Khi tích hợp Tello thật:
pip install djitellopy opencv-python
```

> **Lưu ý**: `scipy` cần thiết cho Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`).

## Chạy thử (mô phỏng, không cần Tello)

```bash
# Chế độ mô phỏng mặc định — khởi động dashboard + 3 UAV sender giả lập:
python main.py

# Mô phỏng với encoder đã fine-tune (thay thế trọng số ImageNet mặc định):
python main.py --checkpoint checkpoints/best_encoder.pth
```

## Chạy Benchmark (đánh giá hiệu năng fusion)

```bash
# So sánh định lượng: Kalman Fusion vs LSTM Fusion vs chỉ UAV đơn (RMSE, Coordination Gain)
python main.py --mode benchmark

# Hoặc trực tiếp:
python -m fusion.benchmark
```

Kết quả xuất ra `results/fusion_benchmark.png` — biểu đồ quỹ đạo 2D và sai số theo thời gian.

## Huấn luyện Encoder trên UAV123

```bash
# Chuẩn bị dữ liệu:
#   data_seq/UAV123/<seq_name>/*.jpg
#   anno/UAV123/<seq_name>.txt

# Chạy training (EPOCHS=20, batch=16, LR Cosine Annealing):
python -m encoder.train_encoder
```

Sau khi huấn luyện, hai file checkpoint được lưu:
- `checkpoints/best_encoder.pth` — weights có val loss thấp nhất
- `checkpoints/task_oriented_encoder.pth` — weights cuối epoch

## Kiến trúc hệ thống

```
UAV 1 ─── [Encoder] ──┐
UAV 2 ─── [Encoder] ──┼──[UDP/TCP]──► Ground Station
UAV 3 ─── [Encoder] ──┘               │
   ↑                                   ├── Hungarian Association (fusion/hungarian_association.py)
ResNet-18 backbone                     ├── Multi-Target Kalman Filter (fusion/kalman_fusion.py)
BBox Head: 512→128→4                   └── Dashboard (dashboard/dashboard.py)
Embed Head: 512→256→128
```

## Pipeline thực tế (Hardware PoC)

```bash
# Terminal 1 — Ground Station (nhận + hiển thị):
python -m dashboard.dashboard

# Terminal 2..4 — UAV Senders (mỗi Tello một terminal):
python -m network.uav_sender --uav-id 1 --ground-ip <IP> --ground-port 9000
python -m network.uav_sender --uav-id 2 --ground-ip <IP> --ground-port 9000
python -m network.uav_sender --uav-id 3 --ground-ip <IP> --ground-port 9000

# Thêm --checkpoint để dùng encoder đã fine-tune (thay trọng số ImageNet):
python -m network.uav_sender --uav-id 1 --checkpoint checkpoints/best_encoder.pth ...
```

## Các bước còn cần hoàn thiện trước khi chạy phần cứng thực

1. **Huấn luyện Encoder**: Chạy `python -m encoder.train_encoder` với dữ liệu UAV123 thực tế.
2. **Triển khai Tello SDK**: `network/uav_sender.py` đã hỗ trợ `TelloFrameSource` dùng `djitellopy`.
3. **Hiệu chuẩn Tọa độ (Calibration)**: Đo chính xác `(uav_pos, yaw_deg, pitch_deg)` cho từng UAV trên bãi thử rồi truyền vào `update(uav_pose=...)` của Kalman Filter.
4. **Tinh chỉnh Kalman params**: Điều chỉnh `process_noise_std` / `base_measurement_noise_std` bằng dữ liệu thực từ PoC.
5. **Đo hiệu năng thực tế**: Dùng `GroundReceiver.get_bandwidth_kbps()` và so sánh với baseline video 50 Mbps.
