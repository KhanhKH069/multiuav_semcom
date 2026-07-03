import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import cv2
import numpy as np

def generate_sequence(seq_name, bg_color, box_size, traj_fn, num_frames=100):
    """
    Tạo một sequence ảnh giả lập với mục tiêu di động và file annotation tương ứng.
    
    bg_color: (B, G, R)
    box_size: (w, h)
    traj_fn: hàm nhận t (1->num_frames) và trả về (x, y) góc trên bên trái của bbox
    """
    data_root = os.path.join("data", "UAV123")
    img_dir = os.path.join(data_root, "UAV123", seq_name)
    anno_dir = os.path.join(data_root, "anno", "UAV123")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)
    
    anno_path = os.path.join(anno_dir, f"{seq_name}.txt")
    
    width, height = 640, 480
    w_box, h_box = box_size
    
    with open(anno_path, "w") as f_anno:
        for t in range(1, num_frames + 1):
            # 1. Tạo ảnh nền
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :] = bg_color
            
            # 2. Lấy tọa độ bbox từ hàm quỹ đạo
            x, y = traj_fn(t)
            
            # Đảm bảo bbox nằm trong ảnh
            x = max(0, min(width - w_box, x))
            y = max(0, min(height - h_box, y))
            
            # 3. Vẽ vật thể (hộp chữ nhật đặc)
            # car1 màu đỏ, person1 màu xanh dương
            color = (0, 0, 255) if "car" in seq_name else (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w_box, y + h_box), color, -1)
            
            # 4. Ghi ảnh xuống đĩa
            img_filename = f"{t:06d}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            cv2.imwrite(img_path, img)
            
            # 5. Ghi annotation: x, y, w, h
            f_anno.write(f"{x},{y},{w_box},{h_box}\n")
            
    print(f"[OK] Da tao xong sequence '{seq_name}': {num_frames} anh va tep annotation.")

def main():
    print("[INFO] Dang sinh tap du lieu UAV123 mau...")
    
    # Quỹ đạo cho car1 (di chuyển ngang)
    def traj_car(t):
        x = int(50 + (t - 1) * 5)
        y = int(220 + np.sin((t - 1) / 5.0) * 30)
        return x, y

    # Quỹ đạo cho person1 (di chuyển dọc)
    def traj_person(t):
        x = int(300 + np.cos((t - 1) / 5.0) * 30)
        y = int(50 + (t - 1) * 3.5)
        return x, y

    generate_sequence(
        seq_name="car1",
        bg_color=(34, 139, 34), # Màu xanh cỏ (Green)
        box_size=(60, 30),
        traj_fn=traj_car,
        num_frames=100
    )

    generate_sequence(
        seq_name="person1",
        bg_color=(128, 128, 128), # Màu xám bê tông (Gray)
        box_size=(30, 60),
        traj_fn=traj_person,
        num_frames=100
    )
    
    print("[DONE] Sinh tap du lieu mau thanh cong tai thu muc 'data/UAV123'!")

if __name__ == "__main__":
    main()
