import cv2
import numpy as np
import os

def main():
    print("="*50)
    print("     OpenCV 3 从基础到进阶教程：基础篇演示")
    print("="*50)

    input_path = "test_images/standard.jpg"
    output_dir = "output_images/01_basic"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 读取图像
    # cv2.IMREAD_COLOR (1): 读入彩色图像
    # cv2.IMREAD_GRAYSCALE (0): 读入灰度图像
    # cv2.IMREAD_UNCHANGED (-1): 读入完整图像包括alpha通道
    print("[1.1] 读取图像...")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误: 无法读取图像 {input_path}")
        return

    # 2. 图像基本属性
    print(f"\n[2.1] 图像形状 (Height, Width, Channels): {img.shape}")
    print(f"[2.2] 图像总像素数量 (size): {img.size}")
    print(f"[2.3] 数据类型 (dtype): {img.dtype}")

    # 3. 通道分离与合并
    print("\n[3.1] 通道分离 (Split)...")
    b, g, r = cv2.split(img)
    # 保存分离结果
    cv2.imwrite(os.path.join(output_dir, "channel_B.jpg"), b)
    cv2.imwrite(os.path.join(output_dir, "channel_G.jpg"), g)
    cv2.imwrite(os.path.join(output_dir, "channel_R.jpg"), r)
    
    print("[3.2] 通道合并 (Merge)...")
    merged = cv2.merge([b, g, r])
    # 将R和B通道互换，制造冷色调效果
    merged_rb_swapped = cv2.merge([r, g, b])
    cv2.imwrite(os.path.join(output_dir, "merged_rb_swapped.jpg"), merged_rb_swapped)

    # 4. 感兴趣区域 (ROI) 截取
    print("\n[4.1] 截取感兴趣区域 (ROI)...")
    # numpy 切片操作 img[y1:y2, x1:x2]
    h, w = img.shape[:2]
    roi = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]
    cv2.imwrite(os.path.join(output_dir, "roi.jpg"), roi)

    # 5. 几何变换
    print("\n[5.1] 图像缩放 (Resize)...")
    # 按绝对尺寸缩放
    resized_abs = cv2.resize(img, (300, 200))
    # 按比例缩放
    resized_rel = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(output_dir, "resized.jpg"), resized_rel)

    print("[5.2] 图像翻转 (Flip)...")
    # 1: 水平翻转, 0: 垂直翻转, -1: 水平垂直翻转
    flipped = cv2.flip(img, 1)
    cv2.imwrite(os.path.join(output_dir, "flipped.jpg"), flipped)

    print("[5.3] 图像旋转 (Rotate)...")
    center = (w // 2, h // 2)
    # 构造旋转矩阵：center, angle, scale
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    # 应用仿射变换
    rotated = cv2.warpAffine(img, M, (w, h))
    cv2.imwrite(os.path.join(output_dir, "rotated.jpg"), rotated)

    # 6. 绘图操作
    print("\n[6.1] 绘制基本几何图形与文字...")
    draw_img = img.copy()
    # 绘制直线: 图像, 起点, 终点, 颜色(BGR), 线宽
    cv2.line(draw_img, (10, 10), (w-10, h-10), (0, 255, 255), 3)
    # 绘制矩形: 图像, 左上角, 右下角, 颜色, 线宽 (-1代表填充)
    cv2.rectangle(draw_img, (50, 50), (150, 150), (255, 0, 255), 2)
    # 绘制圆形: 图像, 圆心, 半径, 颜色, 线宽
    cv2.circle(draw_img, (w-100, 100), 50, (0, 255, 0), -1)
    # 绘制多边形
    pts = np.array([[10, h-10], [100, h-100], [190, h-10]], np.int32).reshape((-1, 1, 2))
    cv2.polylines(draw_img, [pts], isClosed=True, color=(0, 0, 255), thickness=4)
    # 绘制文字
    cv2.putText(draw_img, "OpenCV 3 Drawing", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "drawing.jpg"), draw_img)

    print(f"\n[INFO] 基础篇处理完成，结果已保存至 '{output_dir}/' 目录下。")

if __name__ == "__main__":
    main()
