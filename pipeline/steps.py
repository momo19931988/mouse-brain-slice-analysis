import os
import re
import cv2
import numpy as np
import tifffile
from pathlib import Path
from skimage import io
from skimage.transform import resize
from imageio.v2 import imwrite
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import NMF
import SimpleITK as sitk
import napari

import tkinter as tk
from tkinter import filedialog


# ---------------- Step1 Adjust brain slice orientation ----------------
def rotate_image_stack_clockwise(image_stack):
    return np.stack([np.rot90(img, k=3) for img in image_stack], axis=0)

def save_channels(image_stack, base_name, output_dir):
    names = ["DAPI", "TH", "Iba1"]
    for i in range(3):
        ch_img = image_stack[i]
        out_path = output_dir / f"{base_name}_{names[i]}.tif"
        imwrite(str(out_path), ch_img)
        print(f"✅ Saved channel: {out_path.name}")

def rotate_to_horizontal(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0]); sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

def compute_pca_angle(contour):
    pts = contour.reshape(-1, 2).astype(np.float32)
    pts -= np.mean(pts, axis=0)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_direction = eigvecs[:, 1]
    angle_rad = np.arctan2(main_direction[1], main_direction[0])
    angle_deg = np.degrees(angle_rad)
    if angle_deg < -90: angle_deg += 180
    elif angle_deg > 90: angle_deg -= 180
    return angle_deg

def step1_main():
    root = tk.Tk(); root.withdraw()
    input_path = filedialog.askdirectory(title="Select folder containing .ome.tif images")
    if not input_path: return

    input_dir = Path(input_path)
    split_dir = input_dir / "split_output"
    rotated_dir = split_dir / "rotated_output"
    split_dir.mkdir(exist_ok=True); rotated_dir.mkdir(exist_ok=True)

    ome_files = list(input_dir.glob("*.ome.tif"))
    for file in ome_files:
        image = tifffile.imread(str(file))
        if image.ndim == 4: image = image[0]
        rotated = rotate_image_stack_clockwise(image)
        base_name = file.stem.replace(".ome", "")
        save_channels(rotated, base_name, split_dir)

    tif_files = list(split_dir.glob("*.tif"))
    dapi_files = [f for f in tif_files if "DAPI" in f.name.upper()]
    for dapi_path in dapi_files:
        match = re.match(r"(.+)_DAPI", dapi_path.stem, re.IGNORECASE)
        if not match: continue
        group_prefix = match.group(1)
        ref_img = io.imread(str(dapi_path))
        gray = ref_img if ref_img.ndim == 2 else cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        largest_contour = max(contours, key=cv2.contourArea)
        angle = compute_pca_angle(largest_contour)
        angle = max(-45, min(45, angle))
        group_files = [f for f in tif_files if f.stem.startswith(group_prefix)]
        for f in group_files:
            img = io.imread(str(f))
            rotated_img = rotate_to_horizontal(img, angle).astype(img.dtype)
            save_path = rotated_dir / f"{f.stem}_rotated.tif"
            imwrite(str(save_path), rotated_img)
            print(f"✅ Rotated and saved: {save_path.name}")


# ---------------- Step2 Standardize image size ----------------
def pad_to_canvas_centered(image: np.ndarray, target_size=(6810, 5495)) -> tuple:
    h, w = image.shape
    target_h, target_w = target_size
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    pad_top = (target_h - h) // 2
    pad_bottom = target_h - h - pad_top
    padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    return padded, pad_left, pad_top, pad_right, pad_bottom

def extract_group_id(filename):
    return filename.replace('_DAPI_rotated.tif','').replace('_Iba1_rotated.tif','').replace('_TH_rotated.tif','')

def step2_main():
    root = tk.Tk(); root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing rotated images")
    if not folder: return
    folder = Path(folder)
    output_dir = folder / "canvas_padded_grouped"; output_dir.mkdir(parents=True, exist_ok=True)
    grouped_images = defaultdict(list); sizes = []
    for file in sorted(folder.glob("*_rotated.tif")):
        img = tifffile.imread(str(file))
        if img.ndim == 2: img = np.expand_dims(img, axis=0)
        elif img.ndim == 3 and img.shape[0] not in [1, 3, 4]:
            img = np.transpose(img, (2, 0, 1))
        sizes.append((img.shape[2], img.shape[1]))
        grouped_images[extract_group_id(file.name)].append((file, img))
    max_w = max(w for w, h in sizes); max_h = max(h for w, h in sizes)
    for idx, (group_id, items) in enumerate(grouped_images.items(), start=1):
        ch_count = 1
        for file, img_stack in items:
            for ch_img in img_stack:
                padded, *_ = pad_to_canvas_centered(ch_img, target_size=(max_h, max_w))
                out_path = output_dir / f"image{idx}_channel{ch_count}.tif"
                tifffile.imwrite(str(out_path), padded)
                ch_count += 1
                print(f"✅ {file.name} → {out_path.name}")


# ---------------- Step3 Extract fluorescence channels (NMF) ----------------
def nmf_remove_background(img, n_components=2, keep_component=1):
    h, w = img.shape
    flat = img.reshape(-1, 1)
    flat = np.clip(flat, 0, None)
    X = np.hstack([flat, flat.copy()])
    model = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
    W = model.fit_transform(X)
    H = model.components_
    reconstructed = np.outer(W[:, keep_component], H[keep_component, :]).mean(axis=1)
    return np.clip(reconstructed.reshape(h, w), 0, None)

def step3_main():
    root = tk.Tk(); root.withdraw()
    input_dir = filedialog.askdirectory(title="Select folder containing channel images")
    if not input_dir: return
    input_path = Path(input_dir)
    output_dir = input_path / "nmf_output"; output_dir.mkdir(exist_ok=True)
    ch_files = {f"channel{i}": sorted(input_path.glob(f"*channel{i}.tif")) for i in [1,2,3]}
    results = []
    for ch_name, files in ch_files.items():
        for f in files:
            base = re.sub(rf"_{ch_name}\.tif$", "", f.name)
            img = tifffile.imread(str(f)).astype(np.float32)
            img_sep = nmf_remove_background(img)
            out_name = f"{base}_{ch_name}_nmfsep.tif"
            imwrite(str(output_dir / out_name),
                    np.clip(img_sep*65535/np.max(img_sep),0,65535).astype(np.uint16))
            results.append({"Image": base, "Channel": ch_name})
            print(f"✅ Processed {ch_name}: {base}")
    pd.DataFrame(results).to_csv(output_dir/"summary.csv", index=False)


# ---------------- Step4 Align images (SimpleITK) ----------------
def step4_main():
    root = tk.Tk(); root.withdraw()
    template_path = filedialog.askopenfilename(title="Select DAPI template image")
    if not template_path: return
    template_np = tifffile.imread(template_path).astype(np.float32)
    template_sitk = sitk.Cast(sitk.GetImageFromArray(template_np), sitk.sitkFloat32)
    input_dir = filedialog.askdirectory(title="Select folder containing *_nmfsep.tif files")
    if not input_dir: return
    input_path = Path(input_dir)
    output_dir = input_path / "aligned"; output_dir.mkdir(exist_ok=True)
    base_names = sorted(set(f.name.replace('_channel1_nmfsep.tif', '') for f in input_path.glob("*_channel1_nmfsep.tif")))
    for base in base_names:
        ch_imgs = [tifffile.imread(str(input_path / f"{base}_channel{i}_nmfsep.tif")).astype(np.float32) for i in [1,2,3]]
        ch_sitk = [sitk.Cast(sitk.GetImageFromArray(x), sitk.sitkFloat32) for x in ch_imgs]
        elastix = sitk.ImageRegistrationMethod()
        elastix.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        elastix.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-6, numberOfIterations=200)
        elastix.SetInitialTransform(sitk.CenteredTransformInitializer(template_sitk, ch_sitk[0], sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY))
        elastix.SetInterpolator(sitk.sitkLinear)
        transform = elastix.Execute(template_sitk, ch_sitk[0])
        aligned = [sitk.GetArrayFromImage(sitk.Resample(x, template_sitk, transform, sitk.sitkLinear)) for x in ch_sitk]
        for i, arr in enumerate(aligned,1):
            tifffile.imwrite(str(output_dir/f"{base}_channel{i}_aligned.tif"), arr.astype(np.float32))
        print(f"✅ {base} aligned successfully")


# ---------------- Step5 Downsample images ----------------
def downsample_image(image, factor=0.75):
    h, w = image.shape
    new_h, new_w = int(h*factor), int(w*factor)
    return resize(image, (new_h, new_w), preserve_range=True, anti_aliasing=True)

def step5_main():
    root = tk.Tk(); root.withdraw()
    input_dir = filedialog.askdirectory(title="Select input folder")
    if not input_dir: return
    input_path = Path(input_dir); output_dir = input_path/"downsampled"; output_dir.mkdir(exist_ok=True)
    for file in input_path.glob("*.tif"):
        img = tifffile.imread(str(file))
        if img.ndim == 3:
            downsampled = np.stack([downsample_image(ch) for ch in img])
        else:
            downsampled = downsample_image(img)
        tifffile.imwrite(str(output_dir/file.name), downsampled.astype(np.float32))
        print(f"✅ Downsampled: {file.name}")


# ---------------- Step6 Normalize fluorescence intensity ----------------
def rescale_image(img, global_min, global_max):
    img = np.clip(img, global_min, global_max)
    scaled = (img - global_min)/(global_max-global_min)
    return (scaled*65535).astype(np.uint16)

def step6_main():
    root = tk.Tk(); root.withdraw()
    input_path = filedialog.askdirectory(title="Select folder containing images")
    if not input_path: return
    input_dir = Path(input_path); output_dir = input_dir/"rescaled"; output_dir.mkdir(exist_ok=True)
    channels = {"channel1": [], "channel2": [], "channel3": []}
    for file in input_dir.glob("*.tif"):
        for ch in channels:
            if ch in file.name: channels[ch].append(file)
    gmin, gmax = {}, {}
    for ch, files in channels.items():
        vals = [tifffile.imread(str(f)) for f in files]
        gmin[ch] = min(x.min() for x in vals); gmax[ch] = max(x.max() for x in vals)
        for f in files:
            img = tifffile.imread(str(f))
            tifffile.imwrite(str(output_dir/f.name), rescale_image(img,gmin[ch],gmax[ch]))
            print(f"✅ Rescaled: {f.name}")


# ---------------- Step7 Extract brain region coordinates ----------------
def step7_main():
    image_path = Path("C:/2026/imagingpro/snpc_ana/WT/image9_channel1_aligned.tif")
    if not image_path.exists():
        print("❌ Example image not found"); return
    image = io.imread(str(image_path))
    viewer = napari.Viewer()
    viewer.add_image(image, name="DAPI", colormap='gray')
    viewer.add_shapes(name="Brain_Regions", shape_type='polygon', edge_color='cyan', face_color='transparent')
    print("✅ Image loaded. Please draw ROIs in Napari and save shapes from File → Save Shapes")
    napari.run()
