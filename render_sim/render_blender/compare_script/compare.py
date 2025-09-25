import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import argparse
def mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def psnr(imageA, imageB):
    mse_val = mse(imageA, imageB)
    if mse_val == 0:
        return float("inf")
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse_val))

def compare_render(img_path1, img_path2, diff_output="./render_diff.png"):
    # 读取两张图像
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        raise ValueError("无法读取图像，请检查路径是否正确")

    # 确保尺寸一致
    if img1.shape != img2.shape:
        raise ValueError(f"图像尺寸不一致: {img1.shape} vs {img2.shape}")

    # ---------- 像素级别对比 ----------
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    total_pixels = diff_gray.size
    nonzero_count = np.count_nonzero(diff_gray)
    diff_ratio = nonzero_count / total_pixels
    mean_diff = np.mean(diff_gray)

    # 差异区域高亮显示（红色）
    mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)[1]
    diff_highlight = img2.copy()
    diff_highlight[mask > 0] = [0, 0, 255]
    cv2.imwrite(diff_output, diff_highlight)

    # ---------- 图像质量指标 ----------
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    mse_val = mse(gray1, gray2)
    psnr_val = psnr(gray1, gray2)
    ssim_val, _ = ssim(gray1, gray2, full=True)

    # ---------- 输出结果 ----------
    print("===== 渲染图像质量对比结果 =====")
    print(f"总像素数: {total_pixels}")
    print(f"差异像素数: {nonzero_count}")
    print(f"差异比例: {diff_ratio:.4%}")
    print(f"平均差异强度 (0-255): {mean_diff:.2f}")
    print(f"MSE: {mse_val:.4f}  （越小越好）")
    print(f"PSNR: {psnr_val:.2f} dB  （越大越好）")
    print(f"SSIM: {ssim_val:.4f}  （越接近1越好）")
    print(f"差异图已保存为: {diff_output}")

    return {
        "diff_ratio": diff_ratio,
        "mean_diff": mean_diff,
        "MSE": mse_val,
        "PSNR": psnr_val,
        "SSIM": ssim_val
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two images using SSIM")
    parser.add_argument("image1", help="Path to the first image")
    parser.add_argument("image2", help="Path to the second image")
    parser.add_argument("--output", "-o", default="./render_diff.png", 
                        help="Path to save the difference image (default: ./render_diff.png)")
    
    args = parser.parse_args()
    
    compare_render(args.image1, args.image2, args.output)