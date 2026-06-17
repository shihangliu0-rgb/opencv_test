import cv2
import numpy as np
import os

def main():
    print("="*50)
    print("     OpenCV 3 从基础到进阶教程：边缘、轮廓与特征篇演示")
    print("="*50)

    input_standard = "test_images/standard.jpg"
    input_shapes = "test_images/shapes.png"

    output_dir = "output_images/03_features"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_std = cv2.imread(input_standard)
    img_shapes = cv2.imread(input_shapes)
    gray_std = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
    gray_shapes = cv2.cvtColor(img_shapes, cv2.COLOR_BGR2GRAY)

    # 1. 边缘检测
    print("[1.1] 边缘检测 (Sobel 算子)...")
    # cv2.CV_64F 能保持负的梯度值，然后再转回 uint8
    sobel_x = cv2.Sobel(gray_std, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_std, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    # 按权相加得到综合Sobel边缘
    sobel_combined = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, "edge_1_sobel_x.jpg"), abs_sobel_x)
    cv2.imwrite(os.path.join(output_dir, "edge_2_sobel_y.jpg"), abs_sobel_y)
    cv2.imwrite(os.path.join(output_dir, "edge_3_sobel_combined.jpg"), sobel_combined)

    print("[1.2] 边缘检测 (Laplacian 算子)...")
    laplacian = cv2.Laplacian(gray_std, cv2.CV_64F)
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite(os.path.join(output_dir, "edge_4_laplacian.jpg"), abs_laplacian)

    print("[1.3] 边缘检测 (Canny 算法)...")
    # Canny(image, threshold1, threshold2)
    canny_edges = cv2.Canny(gray_std, 100, 200)
    cv2.imwrite(os.path.join(output_dir, "edge_5_canny.jpg"), canny_edges)

    # 2. 轮廓查找与绘制
    print("\n[2.1] 轮廓查找与绘制 (findContours & drawContours)...")
    # 先将图像二值化
    _, shapes_bin = cv2.threshold(gray_shapes, 220, 255, cv2.THRESH_BINARY_INV)
    
    # 查找轮廓 (兼容 OpenCV 3 和 OpenCV 4)
    # OpenCV 3: image, contours, hierarchy = cv2.findContours(...)
    # OpenCV 4: contours, hierarchy = cv2.findContours(...)
    find_res = cv2.findContours(shapes_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = find_res[1] if len(find_res) == 3 else find_res[0]

    print(f"[INFO] 共找到 {len(contours)} 个外部轮廓。")

    # 绘制全部轮廓
    img_contours_drawn = img_shapes.copy()
    cv2.drawContours(img_contours_drawn, contours, -1, (0, 0, 0), 3)
    cv2.imwrite(os.path.join(output_dir, "contour_1_all.png"), img_contours_drawn)

    # 3. 轮廓特征分析与多边形拟合识别几何形状
    print("\n[3.1] 轮廓特征解析与形状识别...")
    img_analysis = img_shapes.copy()

    for i, cnt in enumerate(contours):
        # 3.1 面积与周长
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < 100: # 过滤极小噪点
            continue
        
        # 3.2 图像矩 (计算质心)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # 3.3 多边形拟合 (判断形状)
        # epsilon 是多边形逼近的最大距离精度，通常设为周长的百分比
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        num_vertices = len(approx)

        shape_name = "Unknown"
        if num_vertices == 3:
            shape_name = "Triangle"
        elif num_vertices == 4:
            # 区分矩形和其他四边形
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif num_vertices > 4:
            # 多于4个顶点，近似为圆或椭圆
            shape_name = "Circle/Ellipse"

        print(f"轮廓 #{i+1}: 形状={shape_name}, 顶点数={num_vertices}, 面积={area:.1f}, 周长={perimeter:.1f}, 质心=({cX},{cY})")

        # 绘制质心
        cv2.circle(img_analysis, (cX, cY), 7, (255, 0, 255), -1)
        # 绘制形状名称
        cv2.putText(img_analysis, shape_name, (cX - 30, cY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 3.4 最小外接正矩形 (绿色)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_analysis, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 3.5 最小旋转外接矩形 (蓝色)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        cv2.drawContours(img_analysis, [box], 0, (255, 0, 0), 2)

        # 3.6 最小外接圆 (红色)
        (center_x, center_y), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(img_analysis, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "contour_2_analysis.png"), img_analysis)
    print(f"\n[INFO] 边缘、轮廓与特征篇完成，结果已保存至 '{output_dir}/' 目录下。")

if __name__ == "__main__":
    main()
