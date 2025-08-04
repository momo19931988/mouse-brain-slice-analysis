
import numpy as np
import tifffile
from pathlib import Path
from skimage.transform import resize
from tkinter import filedialog, Tk

def downsample_image(image, factor=0.75):
    h, w = image.shape
    new_h, new_w = int(h * factor), int(w * factor)
    return resize(image, (new_h, new_w), preserve_range=True, anti_aliasing=True)

def main():
    root = Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="选择输入文件夹 (降采样到 75%)")
    if not input_dir:
        print("❌ 未选择文件夹")
        return

    input_path = Path(input_dir)
    output_dir = input_path / "downsampled_75pct"
    output_dir.mkdir(exist_ok=True)

    files = list(input_path.glob("*.tif"))
    if not files:
        print("⚠️ 文件夹内没有找到 .tif 文件")
        return

    for file in files:
        try:
            img = tifffile.imread(str(file))
            if img.ndim == 3:
                downsampled = np.stack([downsample_image(ch) for ch in img])
            else:
                downsampled = downsample_image(img)
            out_path = output_dir / file.name
            tifffile.imwrite(str(out_path), downsampled.astype(np.float32))
            print(f"✅ 已处理: {file.name}")
        except Exception as e:
            print(f"❌ 处理失败: {file.name}, 错误: {e}")

    print("🎉 所有图片已降采样到 75% 分辨率")

if __name__ == "__main__":
    main()
