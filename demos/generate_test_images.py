import cv2
import numpy as np
import os

def create_test_images():
    output_dir = "test_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("[INFO] 开始生成测试图像...")

    # 1. 生成一张具有丰富渐变和几何形状的“模拟人脸/彩色标准”图
    img_standard = np.zeros((500, 500, 3), dtype=np.uint8)
    # 背景渐变
    for y in range(500):
        img_standard[y, :, 0] = int(255 * y / 500) # B
        img_standard[y, :, 1] = int(255 * (500 - y) / 500) # G
        img_standard[y, :, 2] = 120 # R
    # 绘制模拟人脸 (用于简单的Haar或几何演示)
    cv2.circle(img_standard, (250, 250), 120, (220, 220, 250), -1) # 脸
    cv2.circle(img_standard, (200, 210), 20, (0, 0, 0), -1)       # 左眼
    cv2.circle(img_standard, (300, 210), 20, (0, 0, 0), -1)       # 右眼
    cv2.ellipse(img_standard, (250, 280), (40, 20), 0, 0, 180, (0, 0, 255), 5) # 嘴巴
    cv2.putText(img_standard, "OpenCV Tutorial", (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, "standard.jpg"), img_standard)

    # 2. 生成形状图 (用于轮廓检测、边缘检测、形态学)
    img_shapes = np.zeros((400, 600, 3), dtype=np.uint8)
    img_shapes.fill(255) # 白底
    # 实心矩形
    cv2.rectangle(img_shapes, (50, 50), (200, 180), (255, 0, 0), -1)
    # 实心圆
    cv2.circle(img_shapes, (350, 115), 65, (0, 255, 0), -1)
    # 多边形 (三角形)
    pts = np.array([[480, 50], [420, 180], [540, 180]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img_shapes, [pts], (0, 0, 255))
    # 空心椭圆与带孔图形
    cv2.ellipse(img_shapes, (150, 300), (80, 40), 30, 0, 360, (0, 255, 255), -1)
    cv2.circle(img_shapes, (150, 300), 20, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "shapes.png"), img_shapes)

    # 3. 生成噪声图 (用于滤波与去噪)
    img_noisy = cv2.resize(img_standard, (300, 300))
    # 加入椒盐噪声
    num_salt = 1000
    for _ in range(num_salt):
        y = np.random.randint(0, img_noisy.shape[0])
        x = np.random.randint(0, img_noisy.shape[1])
        img_noisy[y, x] = 255
    num_pepper = 1000
    for _ in range(num_pepper):
        y = np.random.randint(0, img_noisy.shape[0])
        x = np.random.randint(0, img_noisy.shape[1])
        img_noisy[y, x] = 0
    cv2.imwrite(os.path.join(output_dir, "noisy.jpg"), img_noisy)

    # 4. 生成用于模板匹配的目标图和模板图
    img_target = np.zeros((400, 400, 3), dtype=np.uint8)
    img_target.fill(100)
    # 在特定位置放置一个特定的图案
    cv2.rectangle(img_target, (120, 150), (220, 250), (0, 128, 255), -1)
    cv2.circle(img_target, (170, 200), 25, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(output_dir, "target.png"), img_target)
    
    # 截取模板
    img_template = img_target[140:260, 110:230]
    cv2.imwrite(os.path.join(output_dir, "template.png"), img_template)

    print(f"[INFO] 测试图像已成功生成保存在 '{output_dir}/' 目录下。")

if __name__ == "__main__":
    create_test_images()
