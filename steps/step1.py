
import os
import numpy as np
from pathlib import Path
import tifffile
import tkinter as tk
from tkinter import filedialog
from skimage import io
import cv2
import re
from imageio.v2 import imwrite

def rotate_image_stack_clockwise(image_stack):
    return np.stack([np.rot90(img, k=3) for img in image_stack], axis=0)

def save_channels(image_stack, base_name, output_dir):
    if image_stack.ndim != 3 or image_stack.shape[0] < 3:
        print(f"❌ 图像格式不正确或通道数不足: {base_name}")
        return
    names = ["DAPI", "TH", "Iba1"]
    for i in range(3):
        ch_img = image_stack[i]
        out_path = output_dir / f"{base_name}_{names[i]}.tif"
        imwrite(str(out_path), ch_img)
        print(f"✅ 拆分保存: {out_path.name}")

def rotate_to_horizontal(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def compute_pca_angle(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts -= np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_direction = eigvecs[:, 1]
    angle_rad = np.arctan2(main_direction[1], main_direction[0])
    angle_deg = np.degrees(angle_rad)
    if angle_deg < -90:
        angle_deg += 180
    elif angle_deg > 90:
        angle_deg -= 180
    return angle_deg

def main():
    root = tk.Tk()
    root.withdraw()
    input_path = filedialog.askdirectory(title="请选择包含 .ome.tif 图像的文件夹")
    if not input_path:
        print("❌ 没有选择任何文件夹")
        return

    input_dir = Path(input_path)
    split_dir = input_dir / "split_output"
    rotated_dir = split_dir / "rotated_output"
    split_dir.mkdir(exist_ok=True)
    rotated_dir.mkdir(exist_ok=True)

    ome_files = list(input_dir.glob("*.ome.tif"))
    for file in ome_files:
        print(f"🧠 拆分处理: {file.name}")
        image = tifffile.imread(str(file))
        if image.ndim == 4:
            image = image[0]
        rotated = rotate_image_stack_clockwise(image)
        base_name = file.stem.replace(".ome", "")
        save_channels(rotated, base_name, split_dir)

    tif_files = list(split_dir.glob("*.tif"))
    dapi_files = [f for f in tif_files if "DAPI" in f.name.upper()]

    for dapi_path in dapi_files:
        print(f"📐 PCA角度校准: {dapi_path.name}")
        match = re.match(r"(.+)_DAPI", dapi_path.stem, re.IGNORECASE)
        if not match:
            continue
        group_prefix = match.group(1)
        ref_img = io.imread(str(dapi_path))
        gray = ref_img if ref_img.ndim == 2 else cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ 没有找到轮廓: {dapi_path.name}")
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        angle = compute_pca_angle(largest_contour)

        # 限制旋转角度在 -45 到 45 度
        if angle < -45:
            angle = -45
        elif angle > 45:
            angle = 45

        group_files = [f for f in tif_files if f.stem.startswith(group_prefix)]
        for f in group_files:
            img = io.imread(str(f))
            rotated_img = rotate_to_horizontal(img, angle)
            rotated_img = rotated_img.astype(img.dtype)
            save_path = rotated_dir / f"{f.stem}_rotated.tif"
            imwrite(str(save_path), rotated_img)
            print(f"✅ 已旋转保存: {save_path.name}")

    print("🎉 全部完成！旋转图像保存在:", rotated_dir)

if __name__ == "__main__":
    main()
