#!/usr/bin/env python3
"""
download_mimic.py

Script để tải dataset MIMIC-CXR từ Hugging Face.
Hỗ trợ tải ảnh từ trường 'image' embed và lưu thành file ảnh.
"""

import os
import argparse
import requests
from datasets import load_dataset
from PIL import Image
import io
import numpy as np
from shutil import copyfile

def save_image_from_example(example, out_path, fname_base):
    """
    Cố gắng lưu ảnh từ `example`. Trả về True nếu thành công, False nếu không.
    Ưu tiên:
      1. `image` field nếu có (bytes / numpy / PIL)
      2. `image_path` nếu là URL
      3. `image_path` nếu là đường dẫn local tồn tại
    """
    # 1. thử image embed
    img = example.get("image", None)
    if img is not None:
        try:
            # Nếu ảnh là bytes
            if isinstance(img, (bytes, bytearray)):
                im = Image.open(io.BytesIO(img)).convert("L")
            else:
                # Nếu là PIL Image
                if isinstance(img, Image.Image):
                    im = img.convert("L")
                else:
                    # Nếu là numpy array hoặc tương tự
                    arr = None
                    if isinstance(img, np.ndarray):
                        arr = img
                    else:
                        # Huggingface có thể wrap thành dict hoặc khác
                        try:
                            arr = np.array(img)
                        except Exception:
                            arr = None
                    if arr is not None:
                        # Nếu arr có dạng H×W×C hoặc H×W
                        if arr.ndim == 3:
                            # giả sử màu chéo cuối là channels, lấy kênh đầu hoặc convert to grayscale
                            im = Image.fromarray(arr).convert("L")
                        else:
                            im = Image.fromarray(arr).convert("L")
                    else:
                        raise ValueError("Không chuyển được img trả về thành mảng")
            
            # Tạo tên file
            fname = example.get("image_id", fname_base + ".png")
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                fname = fname_base + ".png"
            
            fullpath = os.path.join(out_path, fname)
            im.save(fullpath)
            return True
            
        except Exception as e:
            print(f"[WARN] Lỗi khi lưu từ trường 'image': {e}")
            return False

    # 2. thử image_path
    img_path = example.get("image_path", None)
    if img_path:
        try:
            # URL
            if img_path.startswith("http://") or img_path.startswith("https://"):
                resp = requests.get(img_path, stream=True, timeout=10)
                if resp.status_code == 200:
                    fname = example.get("image_id", os.path.basename(img_path))
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        fname = os.path.basename(img_path)
                    fullpath = os.path.join(out_path, fname)
                    with open(fullpath, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return True
                else:
                    print(f"[WARN] HTTP lỗi {resp.status_code} khi lấy ảnh từ URL: {img_path}")
            else:
                # local path case
                if os.path.exists(img_path):
                    fname = example.get("image_id", os.path.basename(img_path))
                    fullpath = os.path.join(out_path, fname)
                    copyfile(img_path, fullpath)
                    return True
                else:
                    print(f"[WARN] Đường dẫn local không tồn tại: {img_path}")
        except Exception as e:
            print(f"[WARN] Lỗi khi xử lý 'image_path': {e}")

    return False

def download_mimic_cxr(save_dir, limit=None):
    """
    Tải ảnh từ MIMIC-CXR.
    Dataset hỗ trợ trường 'image' embed, nên script sẽ ưu tiên dùng trường này.
    """
    try:
        # Tải dataset
        print("Đang tải MIMIC-CXR dataset từ Hugging Face...")
        ds = load_dataset("MLforHealthcare/mimic-cxr", split=["train","validation","test"], streaming=False)
        
        # Tạo thư mục đầu ra
        base_out = os.path.join(save_dir)
        os.makedirs(base_out, exist_ok=True)
        
        # Xử lý từng split
        splits = {"train": ds[0], "validation": ds[1], "test": ds[2]}
        
        for split_name, ds_split in splits.items():
            print(f"\nĐang xử lý split: {split_name} (tổng: {len(ds_split)} mẫu)")
            out_split_dir = os.path.join(base_out, split_name)
            os.makedirs(out_split_dir, exist_ok=True)
            
            count_saved = 0
            count_seen = 0
            
            for idx, example in enumerate(ds_split):
                count_seen += 1
                fname_base = f"{split_name}_{idx+1}"
                
                # Lưu ảnh
                success = save_image_from_example(example, out_split_dir, fname_base)
                if success:
                    count_saved += 1
                
                # In tiến độ
                if (count_seen % 100 == 0) or (count_seen == len(ds_split)):
                    print(f"  Đã xử lý {count_seen}/{len(ds_split)} mẫu, lưu được {count_saved} ảnh")
                
                # Dừng nếu đạt giới hạn
                if limit is not None and count_saved >= limit:
                    print(f"  Đã đạt giới hạn {limit} ảnh cho split {split_name}")
                    break
            
            print(f"  → Hoàn thành split {split_name}: Đã lưu {count_saved} ảnh")
        
        print("\nTải dữ liệu hoàn tất!")
        
    except Exception as e:
        print(f"\nCó lỗi xảy ra: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Tải dataset MIMIC-CXR từ Hugging Face")
    parser.add_argument("--out_dir", type=str, required=True, 
                       help="Thư mục gốc để lưu dataset")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Giới hạn số ảnh mỗi split để lưu (None để tải hết)")
    args = parser.parse_args()

    # Gọi hàm tải MIMIC-CXR
    download_mimic_cxr(args.out_dir, limit=args.limit)

if __name__ == "__main__":
    main()
