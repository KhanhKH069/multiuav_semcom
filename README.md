# Multi-UAV Task-Oriented Semantic Communication — Code khung

Code khung (scaffold) cho đề tài "Task-Oriented Semantic Communication cho
Multi-UAV". Chia theo đúng 4 nhóm việc đã phân công.

## Cấu trúc

```
multiuav_semcom/
├── encoder/
│   └── encoder.py          # Nhóm AI & Edge — ResNet18 → 132 floats
├── network/
│   ├── message.py          # Định dạng message ~155-200 byte (bbox + embedding lượng tử hoá)
│   ├── uav_sender.py        # Nhóm Control & Network — chạy trên/cạnh mỗi Tello
│   └── ground_receiver.py   # UDP server phía ground station + đo băng thông
├── fusion/
│   ├── kalman_fusion.py    # Nhóm Data & Fusion — phương án CHÍNH
│   └── lstm_fusion.py       # Nhóm Data & Fusion — phương án so sánh
└── dashboard/
    └── dashboard.py         # Nhóm Dashboard & Báo cáo — demo 3 màn hình
```

## Cài đặt

```bash
pip install torch torchvision numpy matplotlib
# Khi tích hợp Tello thật:
pip install djitellopy opencv-python
```

## Chạy thử (mô phỏng, không cần Tello)

```bash
# Terminal 1 — ground station (nhận + hiển thị)
python -m dashboard.dashboard

# Terminal 2..4 — giả lập 3 UAV (dùng DummyFrameSource)
python -m network.uav_sender --uav-id 1 --ground-ip 127.0.0.1 --ground-port 9000 --dummy
python -m network.uav_sender --uav-id 2 --ground-ip 127.0.0.1 --ground-port 9000 --dummy
python -m network.uav_sender --uav-id 3 --ground-ip 127.0.0.1 --ground-port 9000 --dummy
```

## Các tính năng đã hoàn thiện trước khi chạy trên phần cứng thực

1. **Huấn luyện Encoder**: `encoder/train_encoder.py` đã cung cấp mã nguồn chuẩn để fine-tune `TaskOrientedEncoder` trên UAV123 (kết hợp Bounding Box Regression và Appearance Embedding bằng Triplet Loss).
2. **Triển khai Tello SDK**: `network/uav_sender.py` đã hỗ trợ `TelloFrameSource` dùng `djitellopy` để đọc trực tiếp video từ UAV thật.
3. **Hiệu chuẩn Tọa độ (Calibration)**: `fusion/coordinate_utils.py` cung cấp phép chiếu Camera-to-World 3D sang 2D với đầy đủ tham số Pitch/Yaw theo vị trí camera từng UAV.
4. `fusion/kalman_fusion.py`: tinh chỉnh `process_noise_std` /
   `base_measurement_noise_std` bằng dữ liệu thật từ sim/PoC.
5. Đo bandwidth/accuracy/gain thực tế bằng `GroundReceiver.get_bandwidth_kbps()`
   và so sánh với baseline video 50 Mbps để đưa vào báo cáo.
