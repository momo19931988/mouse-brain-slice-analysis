import napari 
from skimage.io import imread
from pathlib import Path
import numpy as np

def main():
    # 修改为你的图像路径（也可以后面在 napari 中手动加载）
    image_path = Path("C:/2026/imagingpro/snpc_ana/WT/image9_channel1_aligned.tif")  # 替换为你的实际路径

    if not image_path.exists():
        print(f"❌ 图像文件未找到: {image_path}")
        return

    image = imread(str(image_path))

    # 创建 napari viewer
    viewer = napari.Viewer()
    viewer.add_image(image, name="DAPI", colormap='gray')

    # 添加一个 shapes 图层用于手动绘图
    shapes = viewer.add_shapes(name="Brain_Regions", shape_type='polygon', edge_color='cyan', face_color='transparent')

    print("✅ 图像加载成功。请用 GUI 绘制脑区 ROI（多边形），完成后点击 'File -> Save Shapes' 保存为 .csv/.npy")

    napari.run()

if __name__ == "__main__":
    main()
