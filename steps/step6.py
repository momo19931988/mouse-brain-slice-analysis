
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from tkinter import Tk, filedialog

def rescale_image(img, global_min, global_max):
    img = np.clip(img, global_min, global_max)
    scaled = (img - global_min) / (global_max - global_min)
    return (scaled * 65535).astype(np.uint16)  # 输出16位

def main():
    # 用 Tkinter 选择文件夹
    root = Tk()
    root.withdraw()
    input_path = filedialog.askdirectory(title="请选择包含图像的文件夹")
    if not input_path:
        print("❌ 没有选择文件夹，程序退出。")
        return

    input_dir = Path(input_path)
    output_dir = input_dir / "rescaled_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.tif"))

    # 按通道分类
    channels = {"channel1": [], "channel2": [], "channel3": []}
    for file in files:
        for ch in channels:
            if ch in file.name:
                channels[ch].append(file)
                break

    # 计算每个通道的 global min/max
    global_mins = {}
    global_maxs = {}
    for ch, ch_files in channels.items():
        if not ch_files:
            print(f"⚠️ {ch} 没有找到文件，跳过。")
            continue
        min_val = np.inf
        max_val = -np.inf
        print(f"🔍 Calculating global min/max for {ch}...")
        for file in tqdm(ch_files):
            img = tifffile.imread(str(file))
            min_val = min(min_val, img.min())
            max_val = max(max_val, img.max())
        global_mins[ch] = min_val
        global_maxs[ch] = max_val
        print(f"🌟 {ch}: min={min_val}, max={max_val}")

    # 逐个文件 rescale
    print("⚙️ Rescaling images...")
    for ch, ch_files in channels.items():
        if not ch_files:
            continue
        gmin = global_mins[ch]
        gmax = global_maxs[ch]
        for file in tqdm(ch_files):
            img = tifffile.imread(str(file))
            rescaled_img = rescale_image(img, gmin, gmax)
            out_path = output_dir / file.name
            tifffile.imwrite(str(out_path), rescaled_img)
            print(f"✅ Saved: {out_path.name}")

    print("🎉 All images rescaled and saved to:", output_dir)

if __name__ == "__main__":
    main()
