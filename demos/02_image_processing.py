import cv2
import numpy as np
import os

def main():
    print("="*50)
    print("     OpenCV 3 从基础到进阶教程：图像处理篇演示")
    print("="*50)

    input_standard = "test_images/standard.jpg"
    input_shapes = "test_images/shapes.png"
    input_noisy = "test_images/noisy.jpg"
    
    output_dir = "output_images/02_processing"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_std = cv2.imread(input_standard)
    img_shapes = cv2.imread(input_shapes)
    img_noisy = cv2.imread(input_noisy)

    # 1. 颜色空间转换与色彩分割
    print("[1.1] 色彩空间转换 (BGR -> GRAY, HSV)...")
    gray = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_std, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(output_dir, "color_gray.jpg"), gray)
    cv2.imwrite(os.path.join(output_dir, "color_hsv.jpg"), hsv)

    print("[1.2] HSV 色彩分割 (inRange 提取红色区域)...")
    # OpenCV中HSV的H范围是0-180, S是0-255, V是0-255
    # 红色在HSV的H两端 (0-10 和 170-180)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    # 通过掩码提取红色部分
    red_extracted = cv2.bitwise_and(img_std, img_std, mask=red_mask)
    cv2.imwrite(os.path.join(output_dir, "color_red_mask.jpg"), red_mask)
    cv2.imwrite(os.path.join(output_dir, "color_red_extracted.jpg"), red_extracted)

    # 2. 图像阈值分割
    print("\n[2.1] 图像阈值处理 (Thresholding)...")
    # 固定阈值
    _, thresh_bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    _, thresh_bin_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # Otsu自动阈值 (适用于双峰直方图)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 自适应阈值 (解决光照不均匀问题)
    thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(output_dir, "thresh_binary.jpg"), thresh_bin)
    cv2.imwrite(os.path.join(output_dir, "thresh_otsu.jpg"), thresh_otsu)
    cv2.imwrite(os.path.join(output_dir, "thresh_adaptive.jpg"), thresh_adapt)

    # 3. 图像滤波与降噪
    print("\n[3.1] 图像平滑与去噪 (Filtering)...")
    # 均值滤波
    blur_mean = cv2.blur(img_noisy, (5, 5))
    # 高斯滤波 (对高斯噪声有效)
    blur_gaussian = cv2.GaussianBlur(img_noisy, (5, 5), 0)
    # 中值滤波 (对椒盐噪声极度有效)
    blur_median = cv2.medianBlur(img_noisy, 5)
    # 双边滤波 (保边去噪)
    blur_bilateral = cv2.bilateralFilter(img_noisy, 9, 75, 75)

    cv2.imwrite(os.path.join(output_dir, "filter_1_noisy.jpg"), img_noisy)
    cv2.imwrite(os.path.join(output_dir, "filter_2_mean.jpg"), blur_mean)
    cv2.imwrite(os.path.join(output_dir, "filter_3_gaussian.jpg"), blur_gaussian)
    cv2.imwrite(os.path.join(output_dir, "filter_4_median.jpg"), blur_median)
    cv2.imwrite(os.path.join(output_dir, "filter_5_bilateral.jpg"), blur_bilateral)

    # 4. 形态学操作
    print("\n[4.1] 形态学操作 (Morphological Operations)...")
    # 我们用带有细小噪声的二值图像做演示
    shapes_gray = cv2.cvtColor(img_shapes, cv2.COLOR_BGR2GRAY)
    _, shapes_bin = cv2.threshold(shapes_gray, 200, 255, cv2.THRESH_BINARY_INV) # 图形为白，背景为黑
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # 腐蚀: 变细，去小白点
    erode_img = cv2.erode(shapes_bin, kernel, iterations=1)
    # 膨胀: 变粗，填小黑孔
    dilate_img = cv2.dilate(shapes_bin, kernel, iterations=1)
    # 开运算: 先腐蚀后膨胀 (去除外部白噪点)
    open_img = cv2.morphologyEx(shapes_bin, cv2.MORPH_OPEN, kernel)
    # 闭运算: 先膨胀后腐蚀 (填补内部黑孔洞)
    close_img = cv2.morphologyEx(shapes_bin, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite(os.path.join(output_dir, "morph_0_bin.png"), shapes_bin)
    cv2.imwrite(os.path.join(output_dir, "morph_1_erode.png"), erode_img)
    cv2.imwrite(os.path.join(output_dir, "morph_2_dilate.png"), dilate_img)
    cv2.imwrite(os.path.join(output_dir, "morph_3_open.png"), open_img)
    cv2.imwrite(os.path.join(output_dir, "morph_4_close.png"), close_img)

    # 5. 直方图与直方图均衡化
    print("\n[5.1] 直方图均衡化 (Histogram Equalization)...")
    # 普通全局灰度直方图均衡化
    hist_eq = cv2.equalizeHist(gray)
    cv2.imwrite(os.path.join(output_dir, "hist_1_equalized.jpg"), hist_eq)

    # CLAHE (限制对比度自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, "hist_2_clahe_gray.jpg"), clahe_gray)

    # 彩色图像CLAHE (借助LAB颜色空间)
    lab = cv2.cvtColor(img_std, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l) # 仅对亮度分量做均衡化
    lab_merged = cv2.merge([l_clahe, a, b])
    clahe_color = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)
    cv2.imwrite(os.path.join(output_dir, "hist_3_clahe_color.jpg"), clahe_color)

    print(f"\n[INFO] 图像处理篇完成，结果已保存至 '{output_dir}/' 目录下。")

if __name__ == "__main__":
    main()
