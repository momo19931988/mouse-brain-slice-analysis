
import numpy as np
import tifffile
from pathlib import Path
from sklearn.decomposition import NMF
from skimage.filters import threshold_otsu
from imageio.v2 import imwrite
import pandas as pd
import re
from tkinter import filedialog, Tk

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

def compute_statistics(img, mask):
    masked_values = img[mask]
    return {
        'mean': np.mean(masked_values),
        'median': np.median(masked_values),
        'sum': np.sum(masked_values),
        'std': np.std(masked_values)
    }

def nmf_pipeline(input_dir):
    input_path = Path(input_dir)
    output_dir = input_path / "nmf_multichannel_output_with_channel1"
    output_dir.mkdir(exist_ok=True)

    ch1_files = sorted(input_path.glob("*_channel1.tif"))
    ch2_files = sorted(input_path.glob("*_channel2.tif"))
    ch3_files = sorted(input_path.glob("*_channel3.tif"))
    results = []

    for files, channel_name in [(ch1_files, 'channel1'), (ch2_files, 'channel2'), (ch3_files, 'channel3')]:
        for f in files:
            base = re.sub(f"_{channel_name}\.tif$", "", f.name)
            try:
                img = tifffile.imread(str(f)).astype(np.float32)
                img_sep = nmf_remove_background(img, n_components=2, keep_component=1)

                try:
                    otsu_thresh = threshold_otsu(img_sep)
                    mask = img_sep > otsu_thresh
                except:
                    mask = np.ones_like(img_sep, dtype=bool)

                out_name = f"{base}_{channel_name}_nmfsep.tif"
                imwrite(str(output_dir / out_name), np.clip(img_sep * 65535 / np.max(img_sep), 0, 65535).astype(np.uint16))

                stats = compute_statistics(img_sep, mask)
                stats.update({'Image': base, 'Channel': channel_name, 'Method': 'NMF_sep_multichannel'})
                results.append(stats)

                print(f"✅ Processed {channel_name} NMF separation: {base}")

            except Exception as e:
                print(f"❌ Failed to process {base} {channel_name}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "nmf_multichannel_summary.csv", index=False)
    print("📊 Multichannel NMF separation summary saved.")

def main():
    root = Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="选择包含通道图像的文件夹 (NMF multichannel with channel1)")
    if not input_dir:
        print("❌ 未选择文件夹")
        return
    nmf_pipeline(input_dir)
    print("🎉 Multichannel NMF separation pipeline (with channel1) completed.")

if __name__ == "__main__":
    main()
