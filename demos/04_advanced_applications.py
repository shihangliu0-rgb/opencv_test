import cv2
import numpy as np
import os

def main():
    print("="*50)
    print("     OpenCV 3 从基础到进阶教程：进阶应用篇演示")
    print("="*50)

    input_target = "test_images/target.png"
    input_template = "test_images/template.png"
    input_standard = "test_images/standard.jpg"
    input_shapes = "test_images/shapes.png"

    output_dir = "output_images/04_advanced"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 模板匹配 (Template Matching)
    print("[1.1] 模板匹配 (matchTemplate)...")
    target_img = cv2.imread(input_target)
    template_img = cv2.imread(input_template)
    
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
    th, tw = template_gray.shape[:2]

    # 执行模板匹配
    # 常用的匹配方法有: TM_CCOEFF_NORMED, TM_CCORR_NORMED, TM_SQDIFF_NORMED
    # TM_SQDIFF_NORMED 是越小越好，其余是越大越好
    result = cv2.matchTemplate(target_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # TM_CCOEFF_NORMED 的最佳匹配位置是 max_loc
    top_left = max_loc
    bottom_right = (top_left[0] + tw, top_left[1] + th)

    match_display = target_img.copy()
    cv2.rectangle(match_display, top_left, bottom_right, (0, 255, 0), 3)
    cv2.putText(match_display, f"Match: {max_val:.2f}", (top_left[0], top_left[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, "app_1_template_match.png"), match_display)
    print(f"[INFO] 模板匹配最佳得分: {max_val:.4f}, 位置: {top_left}")

    # 2. 霍夫变换检测圆形 (Hough Circles)
    print("\n[2.1] 霍夫变换检测圆形 (HoughCircles)...")
    shapes_img = cv2.imread(input_shapes)
    shapes_gray = cv2.cvtColor(shapes_img, cv2.COLOR_BGR2GRAY)
    shapes_blur = cv2.medianBlur(shapes_gray, 5)

    # cv2.HoughCircles 参数解析:
    # image, method, dp(分辨率累加器与图像比例), minDist(圆心之间最小距离), param1(Canny高阈值), param2(投票数阈值), minRadius, maxRadius
    circles = cv2.HoughCircles(shapes_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=20, maxRadius=100)

    circles_display = shapes_img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            # 绘制圆周 (绿色)
            cv2.circle(circles_display, (c[0], c[1]), c[2], (0, 255, 0), 3)
            # 绘制圆心 (红色)
            cv2.circle(circles_display, (c[0], c[1]), 2, (0, 0, 255), 3)
        print(f"[INFO] 霍夫变换共检测到 {len(circles[0])} 个圆。")
    cv2.imwrite(os.path.join(output_dir, "app_2_hough_circles.png"), circles_display)

    # 3. Haar 级联人脸检测 (Haar Cascade Face Detection)
    print("\n[3.1] Haar 级联分类器人脸检测...")
    std_img = cv2.imread(input_standard)
    std_gray = cv2.cvtColor(std_img, cv2.COLOR_BGR2GRAY)

    # 加载 OpenCV 内置的人脸级联分类器模型
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print(f"[错误] 无法加载 Haar 模型: {cascade_path}")
    else:
        # detectMultiScale(image, scaleFactor, minNeighbors, minSize)
        faces = face_cascade.detectMultiScale(std_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        face_display = std_img.copy()
        
        print(f"[INFO] 共检测到 {len(faces)} 个人脸目标。")
        for (x, y, w, h) in faces:
            cv2.rectangle(face_display, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv2.putText(face_display, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "app_3_face_detection.jpg"), face_display)

    # 4. 视频读取与摄像头处理基础 (VideoCapture Framework)
    print("\n[4.1] 视频流与摄像头操作基础 (演示代码生成)...")
    # 为了演示，我们创建一个简短的模拟视频文件并读取
    video_path = os.path.join(output_dir, "demo_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    vw, vh = 300, 300
    out = cv2.VideoWriter(video_path, fourcc, fps, (vw, vh))

    print("[INFO] 生成模拟视频文件 demo_video.mp4 ...")
    for i in range(20):
        frame = np.zeros((vh, vw, 3), dtype=np.uint8)
        # 每帧移动的圆形
        cv2.circle(frame, (15 * i, 150), 30, (0, int(255*i/20), 255), -1)
        cv2.putText(frame, f"Frame {i+1}/20", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    out.release()

    # 演示读取视频流
    print("[INFO] 使用 VideoCapture 读取刚刚生成的视频流...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # 我们保存视频的前2帧作为示例展示
        if frame_count <= 2:
            cv2.imwrite(os.path.join(output_dir, f"app_4_video_frame_{frame_count}.jpg"), frame)
    cap.release()
    print(f"[INFO] 视频读取演示完毕，成功捕获 {frame_count} 帧视频。")

    print(f"\n[INFO] 进阶应用篇完成，结果已保存至 '{output_dir}/' 目录下。")

if __name__ == "__main__":
    main()
