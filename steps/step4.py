
import SimpleITK as sitk
import numpy as np
import tifffile
from pathlib import Path
from tkinter import filedialog, Tk

def register_rigid(fixed, moving):
    elastix = sitk.ImageRegistrationMethod()
    elastix.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elastix.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                     minStep=1e-6,
                                                     numberOfIterations=200,
                                                     gradientMagnitudeTolerance=1e-6)
    elastix.SetInitialTransform(sitk.CenteredTransformInitializer(fixed, moving,
                                                                  sitk.Euler2DTransform(),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY))
    elastix.SetInterpolator(sitk.sitkLinear)
    elastix.SetShrinkFactorsPerLevel([4, 2, 1])
    elastix.SetSmoothingSigmasPerLevel([2, 1, 0])
    elastix.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = elastix.Execute(fixed, moving)
    return transform

def apply_transform(moving, transform, reference):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    out = resampler.Execute(moving)
    return out

def main():
    root = Tk()
    root.withdraw()
    template_path = filedialog.askopenfilename(title="选择 DAPI template 图", filetypes=[("TIFF files", "*.tif")])
    if not template_path:
        print("❌ 未选择 template 图")
        return
    template_np = tifffile.imread(template_path).astype(np.float32)
    template_sitk = sitk.GetImageFromArray(template_np)
    template_sitk = sitk.Cast(template_sitk, sitk.sitkFloat32)

    input_dir = filedialog.askdirectory(title="选择包含 *_channelX_nmfsep.tif 文件夹")
    if not input_dir:
        print("❌ 未选择文件夹")
        return

    input_path = Path(input_dir)
    output_dir = input_path / "aligned_to_template_sitk"
    output_dir.mkdir(exist_ok=True)

    base_names = sorted(set(f.name.replace('_channel1_nmfsep.tif', '') for f in input_path.glob("*_channel1_nmfsep.tif")))
    for base in base_names:
        try:
            ch1_np = tifffile.imread(str(input_path / f"{base}_channel1_nmfsep.tif")).astype(np.float32)
            ch2_np = tifffile.imread(str(input_path / f"{base}_channel2_nmfsep.tif")).astype(np.float32)
            ch3_np = tifffile.imread(str(input_path / f"{base}_channel3_nmfsep.tif")).astype(np.float32)

            ch1_sitk = sitk.GetImageFromArray(ch1_np)
            ch2_sitk = sitk.GetImageFromArray(ch2_np)
            ch3_sitk = sitk.GetImageFromArray(ch3_np)
            ch1_sitk = sitk.Cast(ch1_sitk, sitk.sitkFloat32)
            ch2_sitk = sitk.Cast(ch2_sitk, sitk.sitkFloat32)
            ch3_sitk = sitk.Cast(ch3_sitk, sitk.sitkFloat32)

            transform = register_rigid(template_sitk, ch1_sitk)
            angle_deg = np.degrees(transform.GetAngle())
            if abs(angle_deg) > 60:
                print(f"⚠️ {base} 旋转角 {angle_deg:.2f}° 超出限制，强制设为0°")
                transform.SetAngle(0.0)
            ch1_aligned = apply_transform(ch1_sitk, transform, template_sitk)
            ch2_aligned = apply_transform(ch2_sitk, transform, template_sitk)
            ch3_aligned = apply_transform(ch3_sitk, transform, template_sitk)

            tifffile.imwrite(str(output_dir / f"{base}_channel1_aligned.tif"),
                             sitk.GetArrayFromImage(ch1_aligned).astype(np.float32))
            tifffile.imwrite(str(output_dir / f"{base}_channel2_aligned.tif"),
                             sitk.GetArrayFromImage(ch2_aligned).astype(np.float32))
            tifffile.imwrite(str(output_dir / f"{base}_channel3_aligned.tif"),
                             sitk.GetArrayFromImage(ch3_aligned).astype(np.float32))

            print(f"✅ {base} aligned successfully")

        except Exception as e:
            print(f"❌ Failed to process {base}: {e}")

    print("🎉 All images aligned to DAPI template using SimpleITK rigid registration.")

if __name__ == "__main__":
    main()
