"""
message.py
----------
Định dạng message ngữ nghĩa gửi từ mỗi UAV về trạm fusion.

Layout (little-endian, dùng struct):
  uav_id      : uint8   (1 byte)   - ID của UAV (1, 2, 3, ...)
  seq         : uint32  (4 byte)   - số thứ tự message, dùng để phát hiện mất gói
  timestamp_ms: uint64  (8 byte)   - epoch time (ms) lúc trích xuất đặc trưng
  bbox[4]     : float16 (8 byte)   - (cx, cy, w, h) chuẩn hoá [0,1]
  embed[128]  : float16 (256 byte) -> QUÁ LỚN nếu dùng float16 nguyên bản (256B)
                nên lượng tử hoá về int8 (xem QuantizedEmbedding bên dưới)

Vì 128 float (kể cả float16) đã vượt quá ngân sách ~200 byte, ta lượng tử hoá
appearance embedding về int8 (mỗi phần tử 1 byte, kèm 1 scale float32 dùng
chung) thay vì float16/float32. Khi đó:

  header (uav_id + seq + timestamp) = 1 + 4 + 8            = 13 byte
  bbox (4 x float16)                                        = 8 byte
  embed_scale (1 x float32)                                 = 4 byte
  embed (128 x int8)                                        = 128 byte
  checksum (uint16, CRC đơn giản)                            = 2 byte
  -----------------------------------------------------------------
  TỔNG                                                      = 155 byte  (< 200 byte mục tiêu)

Sai số lượng tử hoá int8 trên embedding đã L2-normalize là chấp nhận được
cho bài toán appearance matching (không cần độ chính xác float đầy đủ).
"""
from __future__ import annotations

import struct
import time
import zlib
from dataclasses import dataclass

import numpy as np

BBOX_DIM = 4
EMBED_DIM = 128

# '<' = little-endian, không padding
_HEADER_FMT = "<BIQ"                 # uav_id, seq, timestamp_ms
_BBOX_FMT = f"<{BBOX_DIM}e"          # 4 x float16 ('e' = IEEE 754 half)
_SCALE_FMT = "<f"                    # 1 x float32
_EMBED_FMT = f"<{EMBED_DIM}b"        # 128 x int8
_CHECKSUM_FMT = "<H"                 # 1 x uint16

HEADER_SIZE = struct.calcsize(_HEADER_FMT)
BBOX_SIZE = struct.calcsize(_BBOX_FMT)
SCALE_SIZE = struct.calcsize(_SCALE_FMT)
EMBED_SIZE = struct.calcsize(_EMBED_FMT)
CHECKSUM_SIZE = struct.calcsize(_CHECKSUM_FMT)
MESSAGE_SIZE = HEADER_SIZE + BBOX_SIZE + SCALE_SIZE + EMBED_SIZE + CHECKSUM_SIZE


@dataclass
class SemanticMessage:
    uav_id: int
    seq: int
    timestamp_ms: int
    bbox: np.ndarray      # shape (4,), float32, giá trị [0,1]
    embedding: np.ndarray  # shape (128,), float32, đã L2-normalize


def _quantize_embedding(embedding: np.ndarray) -> tuple[np.ndarray, float]:
    """L2-normalized float32 -> int8 + scale, sao cho embedding ~= int8 * scale."""
    max_abs = float(np.max(np.abs(embedding))) or 1e-8
    scale = max_abs / 127.0
    quantized = np.clip(np.round(embedding / scale), -127, 127).astype(np.int8)
    return quantized, scale


def _dequantize_embedding(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * scale


def encode_message(msg: SemanticMessage) -> bytes:
    """Đóng gói SemanticMessage -> bytes (~155 byte)."""
    header = struct.pack(_HEADER_FMT, msg.uav_id, msg.seq, msg.timestamp_ms)
    bbox_bytes = struct.pack(_BBOX_FMT, *msg.bbox.astype(np.float16).tolist())

    quantized, scale = _quantize_embedding(msg.embedding)
    scale_bytes = struct.pack(_SCALE_FMT, scale)
    embed_bytes = struct.pack(_EMBED_FMT, *quantized.tolist())

    payload = header + bbox_bytes + scale_bytes + embed_bytes
    checksum = zlib.crc32(payload) & 0xFFFF
    return payload + struct.pack(_CHECKSUM_FMT, checksum)


def decode_message(data: bytes) -> SemanticMessage:
    """Giải mã bytes -> SemanticMessage. Raise ValueError nếu checksum sai."""
    if len(data) != MESSAGE_SIZE:
        raise ValueError(f"Kích thước message không hợp lệ: {len(data)} (expect {MESSAGE_SIZE})")

    payload, checksum_bytes = data[:-CHECKSUM_SIZE], data[-CHECKSUM_SIZE:]
    (checksum_recv,) = struct.unpack(_CHECKSUM_FMT, checksum_bytes)
    checksum_calc = zlib.crc32(payload) & 0xFFFF
    if checksum_recv != checksum_calc:
        raise ValueError("Checksum không khớp — message có thể bị lỗi trên đường truyền")

    offset = 0
    uav_id, seq, timestamp_ms = struct.unpack(_HEADER_FMT, payload[offset:offset + HEADER_SIZE])
    offset += HEADER_SIZE

    bbox = np.array(struct.unpack(_BBOX_FMT, payload[offset:offset + BBOX_SIZE]), dtype=np.float32)
    offset += BBOX_SIZE

    (scale,) = struct.unpack(_SCALE_FMT, payload[offset:offset + SCALE_SIZE])
    offset += SCALE_SIZE

    quantized = np.array(struct.unpack(_EMBED_FMT, payload[offset:offset + EMBED_SIZE]), dtype=np.int8)
    embedding = _dequantize_embedding(quantized, scale)

    return SemanticMessage(
        uav_id=uav_id, seq=seq, timestamp_ms=timestamp_ms,
        bbox=bbox, embedding=embedding,
    )


def make_message(uav_id: int, seq: int, feature_vec: np.ndarray) -> SemanticMessage:
    """Tiện ích: từ vector 132 floats (output của encoder) -> SemanticMessage."""
    assert feature_vec.shape == (BBOX_DIM + EMBED_DIM,), "Encoder phải xuất đúng 132 floats"
    return SemanticMessage(
        uav_id=uav_id,
        seq=seq,
        timestamp_ms=int(time.time() * 1000),
        bbox=feature_vec[:BBOX_DIM],
        embedding=feature_vec[BBOX_DIM:],
    )


if __name__ == "__main__":
    fake_feature = np.random.rand(132).astype(np.float32)
    fake_feature[4:] /= np.linalg.norm(fake_feature[4:])  # giả lập L2-normalize

    msg = make_message(uav_id=1, seq=42, feature_vec=fake_feature)
    packed = encode_message(msg)
    print(f"Kích thước message: {len(packed)} byte (mục tiêu ~200 byte)")

    decoded = decode_message(packed)
    bbox_err = np.abs(decoded.bbox - msg.bbox).max()
    embed_err = np.abs(decoded.embedding - msg.embedding).max()
    print(f"Sai số bbox tối đa (do float16): {bbox_err:.6f}")
    print(f"Sai số embedding tối đa (do int8 quantization): {embed_err:.6f}")
