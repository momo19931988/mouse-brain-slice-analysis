
import os
from pathlib import Path
import cv2
import numpy as np
from tifffile import imread, imwrite
from tkinter import filedialog, Tk
import csv
from collections import defaultdict
import re

def pad_to_canvas_centered(image: np.ndarray, target_size=(6810, 5495)) -> tuple:
    h, w = image.shape
    target_h, target_w = target_size
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode='edge')
    return padded, pad_left, pad_top, pad_right, pad_bottom

def extract_group_id(filename):
    # 去掉最后的 _DAPI_rotated.tif / _Iba1_rotated.tif / _TH_rotated.tif
    name = filename.replace('_DAPI_rotated.tif', '')                    .replace('_Iba1_rotated.tif', '')                    .replace('_TH_rotated.tif', '')
    return name

def main():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="选择包含旋转图像的文件夹")
    if not folder:
        print("❌ 未选择文件夹")
        return

    folder = Path(folder)
    output_dir = folder / "canvas_padded_grouped"
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_images = defaultdict(list)
    sizes = []

    for file in sorted(folder.glob("*_rotated.tif")):
        try:
            img = imread(str(file))
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            elif img.ndim == 3:
                if img.shape[0] not in [1, 3, 4]:
                    img = np.transpose(img, (2, 0, 1))
            else:
                raise ValueError("Unsupported image shape")

            sizes.append((img.shape[2], img.shape[1]))
            group_id = extract_group_id(file.name)
            grouped_images[group_id].append((file, img))
        except Exception as e:
            print(f"❌ 加载失败: {file.name}, 错误: {e}")

    if not grouped_images:
        print("⚠️ 没有找到任何 _rotated.tif 图像")
        return

    max_w = max(w for w, h in sizes)
    max_h = max(h for w, h in sizes)
    print(f"📏 自动统一画布尺寸: {max_w} x {max_h}")

    log_path = output_dir / "padding_info.csv"
    with open(log_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["original_name", "new_name", "group_id", "channel_index",
                         "original_width", "original_height",
                         "pad_left", "pad_top", "pad_right", "pad_bottom",
                         "final_width", "final_height"])

        for idx, (group_id, items) in enumerate(grouped_images.items(), start=1):
            ch_count = 1
            for file, img_stack in items:
                for ch_index, ch_img in enumerate(img_stack):
                    padded, pad_l, pad_t, pad_r, pad_b = pad_to_canvas_centered(ch_img, target_size=(max_h, max_w))
                    new_name = f"image{idx}_channel{ch_count}.tif"
                    out_path = output_dir / new_name
                    imwrite(str(out_path), padded)
                    print(f"✅ {file.name} → {new_name}")
                    writer.writerow([
                        file.name, new_name, group_id, ch_count,
                        ch_img.shape[1], ch_img.shape[0],
                        pad_l, pad_t, pad_r, pad_b,
                        padded.shape[1], padded.shape[0]
                    ])
                    ch_count += 1

if __name__ == '__main__':
    main()
