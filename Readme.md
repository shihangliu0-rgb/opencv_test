# OpenCV 3 计算机视觉：从基础到进阶实战全方位教程

> **撰写目标**：本教程旨在为学习计算机视觉的开发者提供一份结构清晰、原理解析透彻且具备生产力价值的 **OpenCV 3 (兼顾 Python 3 环境及 OpenCV 4 核心兼容性)** 开发指南。配套完整的演示代码，支持自闭环一键生成测试图像并运行实战演示。

---

## 目录 (Table of Contents)
1. **[前言与环境准备](#一-前言与环境准备)**
   - 1.1 OpenCV 3 简介与核心架构
   - 1.2 Python 环境安装与 OpenCV 3/4 兼容性提示
   - 1.3 教程配套源码结构
2. **[基础篇：Core 与 GUI 基础操作](#二-基础篇core-与-gui-基础操作)**
   - 2.1 图像读取、显示与保存
   - 2.2 图像在 NumPy 中的数据结构与属性
   - 2.3 色彩通道的分离与合并
   - 2.4 感兴趣区域 (ROI) 提取
   - 2.5 几何变换（缩放、翻转、仿射旋转）
   - 2.6 绘制基本图形与文本
3. **[进阶篇上：图像处理与增强 (Imgproc)](#三-进阶篇上图像处理与增强-imgproc)**
   - 3.1 色彩空间转换 (BGR vs RGB vs GRAY vs HSV)
   - 3.2 图像阈值分割 (二值化、Otsu、自适应阈值)
   - 3.3 图像平滑与滤波降噪 (均值、高斯、中值、双边滤波)
   - 3.4 数学形态学操作 (腐蚀、膨胀、开闭运算)
   - 3.5 图像直方图与自适应均衡化 (CLAHE)
4. **[进阶篇中：特征、边缘与轮廓分析](#四-进阶篇中特征边缘与轮廓分析)**
   - 4.1 图像梯度与边缘检测 (Sobel, Laplacian, Canny 算子)
   - 4.2 轮廓查找与绘制 (`findContours` 层级剖析)
   - 4.3 轮廓特征解析 (面积、质心、最小外接矩形/圆)
   - 4.4 多边形拟合与几何形状自动化识别
5. **[进阶篇下：计算机视觉经典实战应用](#五-进阶篇下计算机视觉经典实战应用)**
   - 5.1 模板匹配原理与对象定位 (Template Matching)
   - 5.2 霍夫变换检测几何圆与直线 (Hough Cycles/Lines)
   - 5.3 经典 Haar 级联分类器人脸检测 (Haar Cascade)
   - 5.4 视频流与摄像头处理架构 (`VideoCapture` / `VideoWriter`)
6. **[附录：配套演示源码快速运行指南](#六-附录配套演示源码快速运行指南)**

---

## 一、前言与环境准备

### 1.1 OpenCV 3 简介与核心架构
OpenCV (Open Source Computer Vision Library) 是全球最受推崇的计算机视觉与机器学习开源库。OpenCV 3 标志着该项目的重要里程碑：
- 引入了 `cv::Mat` 架构的深度成熟期，并在 Python 层面完美融合了 **NumPy** 数组结构。
- 将实验性和前沿算法分离至独立的 `opencv_contrib` 模块。
- **模块化架构**：核心模块包括 `core`（核心数据结构与运算）、`imgproc`（图像处理）、`highgui`（图形用户界面）、`features2d`（特征检测与描述）、`objdetect`（目标检测）以及 `video`（视频流分析）。

### 1.2 Python 环境安装与 OpenCV 3/4 兼容性提示
本教程代码采用 **Python 3** 编写。在大多数现代生产环境中，直接通过 pip 安装 `opencv-python` 即可获取主流版本。
```bash
# 安装主流 OpenCV 库
pip install opencv-python numpy
```

> **⚡ [核心指南] OpenCV 3 与 OpenCV 4 的关键差异：**
> Python 下 OpenCV 3 与 4 拥有高达 **98%** 的语法兼容度。最主要的一个断代差异在于轮廓检索函数 `cv2.findContours` 的返回值数量：
> - **OpenCV 3**: 返回 3 个值 `image, contours, hierarchy = cv2.findContours(...)`
> - **OpenCV 4**: 返回 2 个值 `contours, hierarchy = cv2.findContours(...)`
> *本教程配套源码已内置自适应判断逻辑，无论运行在 OpenCV 3 还是 4 下均能完美兼容！*

### 1.3 教程配套源码结构
为了让读者无障碍上手，我们在 `demos/` 目录下准备了完整的可执行 Python 脚本，并提供了自包含的测试图像生成脚本 `generate_test_images.py`。
```text
workspace/
 ├── OpenCV3_Python_Tutorial.md       # 本教程文档
 ├── run_all_demos.py                 # 一键执行全部演示脚本
 ├── demos/
 │    ├── generate_test_images.py     # 自动化生成配套测试图像
 │    ├── 01_basic_operations.py      # 基础篇代码演示
 │    ├── 02_image_processing.py      # 图像处理篇代码演示
 │    ├── 03_feature_edge_contour.py  # 边缘与轮廓篇代码演示
 │    └── 04_advanced_applications.py # 进阶实战篇代码演示
 ├── webcam_studio.py                 # 电影级实时摄像头工作室 (NEON VISION)
 ├── test_images/                     # 执行generate脚本后自动生成的测试图
 └── output_images/                   # 各模块演示代码的输出效果图保存路径
```

---

## 二、基础篇：Core 与 GUI 基础操作

本章重点掌握 OpenCV 最核心的输入输出及内存处理方式。

### 2.1 图像读取、显示与保存
OpenCV 通过 `cv2.imread()` 函数加载图像，返回值是一个三维的 NumPy 数组。
```python
import cv2

# 读取彩色图 (默认模式)
img_color = cv2.imread("image.jpg", cv2.IMREAD_COLOR)

# 读取灰度图
img_gray = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

# 保存图像 (OpenCV会自动根据后缀名采用对应压缩算法)
cv2.imwrite("output.png", img_color)
```

### 2.2 图像在 NumPy 中的数据结构与属性
由于 Python 中的 OpenCV 图像完全由 NumPy 处理，掌握其三剑客属性至关重要：
- `img.shape`：返回一个元组。彩色图为 `(Height, Width, Channels)`，灰度图为 `(Height, Width)`。
- `img.size`：图像中所有像素值的总数（`Height * Width * Channels`）。
- `img.dtype`：通常是 `uint8`（无符号 8 位整型，范围 0-255）。在做高精度计算（如梯度算子）时可能会转为 `float32` 或 `float64`。

### 2.3 色彩通道的分离与合并
> **⚠️ 核心陷阱**：与通用的 RGB 顺序不同，OpenCV 默认加载的彩色图像通道顺序为 **BGR (Blue, Green, Red)**。

```python
# 通道分离
b, g, r = cv2.split(img)

# 通道合并 (比如将红蓝互换变成纯冷色调)
img_bgr = cv2.merge([r, g, b])
```
*提示：`cv2.split()` 是一项耗时操作。在追求高性能的场景下，建议直接使用 NumPy 切片操作，如提取红色通道：`r = img[:, :, 2]`。*

### 2.4 感兴趣区域 (ROI) 提取
在实战中，我们往往只需要关注图像的一部分（如人脸区域、车牌区域），这被称为 **ROI (Region of Interest)**。这可以通过 NumPy 切片高效完成：
```python
# 截取 Height方向 100~300, Width方向 200~400 的子区域
roi = img[100:300, 200:400]
```

### 2.5 几何变换（缩放、翻转、仿射旋转）
OpenCV 提供了强大的几何仿射变换支持。

#### 1. 图像缩放 (`cv2.resize`)
支持按指定的绝对像素尺寸或指定的比例缩放。常用的插值算法有 `cv2.INTER_LINEAR`（默认，双线性插值）和 `cv2.INTER_AREA`（缩小图像时抗锯齿效果最好）。
```python
# 按比例缩小一半
resized = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
```

#### 2. 图像翻转 (`cv2.flip`)
- `flipCode = 1`：水平镜像翻转（极常用于摄像头自拍前置视角的调整）。
- `flipCode = 0`：垂直翻转。
- `flipCode = -1`：同时水平和垂直翻转。

#### 3. 图像旋转 (仿射变换 `cv2.warpAffine`)
任意角度旋转需要先通过 `cv2.getRotationMatrix2D()` 构造 2x3 的仿射矩阵，再进行变换。
```python
# 构造旋转矩阵：参数依次为 旋转中心、旋转角度(逆时针为正)、缩放比例
center = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)

# 执行仿射变换
rotated = cv2.warpAffine(img, M, (width, height))
```

### 2.6 绘制基本图形与文本
OpenCV 支持直接在 NumPy 内存画布上绘制几何图形，支持修改颜色（BGR）和线条粗细：
```python
# 绘制直线 (画布, 起点, 终点, 颜色, 线宽)
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

# 绘制矩形 (左上角, 右下角)
cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)

# 绘制圆形 (圆心, 半径，线宽设为 -1 表示实心填充)
cv2.circle(img, (447, 63), 63, (0, 0, 255), -1)

# 绘制文字 (画布, 文本, 左下角坐标, 字体, 缩放大小, 颜色, 线宽)
cv2.putText(img, 'OpenCV', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 2)
```

---

## 三、进阶篇上：图像处理与增强 (Imgproc)

在进行高级计算机视觉识别之前，图像预处理往往决定了算法的成败。

### 3.1 色彩空间转换 (BGR vs RGB vs GRAY vs HSV)
图像色彩空间的转换通过 `cv2.cvtColor()` 实现。
- **HSV 空间**（Hue 色相，Saturation 饱和度，Value 明度）：在物体颜色识别（如跟踪红球、绿牌）中非常强大。在 OpenCV 中，$H \in [0, 180]$，$S \in [0, 255]$，$V \in [0, 255]$。
```python
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 提取特定范围颜色构建二值掩码 (Mask)
mask = cv2.inRange(hsv, lower_bound_color, upper_bound_color)
```

### 3.2 图像阈值分割 (二值化、Otsu、自适应阈值)
阈值分割的核心在于将灰度图像转换为仅包含 0 (纯黑) 和 255 (纯白) 的二值图像。

#### 1. 固定阈值 (`cv2.threshold`)
如果像素值大于设定的阈值，则赋为最大值（255），否则赋为 0。
```python
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

#### 2. Otsu 自动二值化
面对双峰灰度直方图图像，人工猜测阈值极不稳定。Otsu 算法能基于图像像素的类间方差自动寻找最佳全局阈值。
```python
ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```

#### 3. 自适应阈值 (`cv2.adaptiveThreshold`)
面对光照严重不均（如阴影覆盖下的书本扫描图），全局阈值会失效。自适应阈值会计算像素邻域的平均值或高斯加权平均值，逐个区域独立决定阈值。
```python
thresh_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
```

### 3.3 图像平滑与滤波降噪
真实场景捕获的图像常伴随各类物理噪声。OpenCV 提供了四大经典空间滤波器：

1. **均值滤波 (`cv2.blur`)**：用局部窗口内像素的平均值代替目标像素，平滑效果生硬。
2. **高斯滤波 (`cv2.GaussianBlur`)**：根据二维高斯分布对邻域像素赋予不同权重（越近权重越大），对**高斯噪声**（如夜间拍摄的感光噪点）具有极佳的抑制效果。
3. **中值滤波 (`cv2.medianBlur`)**：用邻域像素排序后的中位数替代目标像素。它是**椒盐噪声**（随机出现的纯白或纯黑噪点）的绝对克星。
4. **双边滤波 (`cv2.bilateralFilter`)**：传统滤波在去噪的同时会模糊物体边缘。双边滤波不仅考虑像素空间距离，还结合像素灰度值差值。**能在有效滤除噪声的同时完美保留锐利的物体边缘**（常用于磨皮美颜）。

### 3.4 数学形态学操作 (腐蚀、膨胀、开闭运算)
形态学操作通常针对二值化图像，通过一个名为“结构元素 (Kernel)”的滑窗对图像进行形状修剪。

| 操作名称 | OpenCV API | 核心数学原理与实战作用 |
| :--- | :--- | :--- |
| **腐蚀 (Erosion)** | `cv2.erode()` | 削弱高亮(白色)区域。用于消除细小的白色噪点，或将粘连的物体分离。 |
| **膨胀 (Dilation)**| `cv2.dilate()`| 扩张高亮(白色)区域。用于填补二值化前景物体内部的细小裂缝或连通断开的线条。|
| **开运算 (Open)** | `cv2.morphologyEx(..., cv2.MORPH_OPEN)` | **先腐蚀，后膨胀**。极度适用于去除前景物体外部的离散白色噪点。 |
| **闭运算 (Close)**| `cv2.morphologyEx(..., cv2.MORPH_CLOSE)`| **先膨胀，后腐蚀**。极度适用于填补前景物体内部的黑色小孔洞。 |

### 3.5 图像直方图与自适应均衡化 (CLAHE)
直方图反映了图像中不同灰度级像素的分布频率。
**直方图均衡化**能够拉伸图像的灰度分布范围，从而显著提升暗淡或过度曝光图像的对比度。

> **🌟 实战黑科技：CLAHE (对比度受限自适应直方图均衡化)**
> 传统的全局直方图均衡化 (`cv2.equalizeHist`) 会导致局部高亮区域被过度放大变成刺眼的纯白。**CLAHE** 将图像划分为 8x8 的小网格分别做均衡化，若某个灰度级直方图高度超过限幅值（clipLimit），则将其裁剪并平均分配到其他灰度级中。这在医学图像（如X光透视图增强）和夜视图像处理中是行业标准。

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced_img = clahe.apply(gray_img)
```

---

## 四、进阶篇中：特征、边缘与轮廓分析

掌握了预处理后，本章将带领读者从像素过渡到几何语义的提取。

### 4.1 图像梯度与边缘检测 (Sobel, Laplacian, Canny 算子)
边缘反映了图像亮度的剧烈跃变。我们通过计算导数来寻找边缘。

#### 1. Sobel 算子
离散微分算子，结合了高斯平滑与微分操作。可分方向求导：
- 水平 Sobel (强调垂直边缘)
- 垂直 Sobel (强调水平边缘)
```python
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
```

#### 2. Canny 边缘检测
这是目前工业界公认 **效果最优** 的经典边缘检测算法。它经历了四大严苛步骤：
1. 用高斯滤波器平滑图像。
2. 计算梯度的幅值和方向。
3. **非极大值抑制 (NMS)**：仅保留局部梯度方向上幅值最大的像素，让边缘变细为单像素宽。
4. **双阈值连接 (Hysteresis Thresholding)**：利用高低两个阈值。强边缘（大于高阈值）直接保留；弱边缘（介于两阈值之间）若与强边缘连通则保留，否则抛弃。
```python
canny_edges = cv2.Canny(gray, threshold1=100, threshold2=200)
```

### 4.2 轮廓查找与绘制 (`findContours` 层级剖析)
二值图像中的连通区域边界即为**轮廓**。
```python
# 提取外部轮廓
contours, hierarchy = cv2.findContours(thresh_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓 (参数依次为：目标画布, 轮廓列表, 绘制轮廓索引(-1代表全画), 颜色, 线宽)
cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
```
- `cv2.RETR_EXTERNAL`：仅提取最外层轮廓。
- `cv2.RETR_TREE`：提取所有轮廓，并重构完整的嵌套父子层级关系。
- `cv2.CHAIN_APPROX_SIMPLE`：轮廓压缩算法，仅保存直线段的终点坐标（如矩形只存4个顶点），极大节约内存。

### 4.3 轮廓特征解析 (面积、质心、最小外接矩形/圆)

每个找到的轮廓都可以进一步提取丰富的几何数学属性：

```python
# 1. 轮廓面积
area = cv2.contourArea(cnt)

# 2. 轮廓周长 (True表示闭合轮廓)
perimeter = cv2.arcLength(cnt, True)

# 3. 空间矩 (Moments) 与 质心 (Centroid) 计算
M = cv2.moments(cnt)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# 4. 最小正外接矩形
x, y, w, h = cv2.boundingRect(cnt)

# 5. 最小旋转最小面积外接矩形 (带旋转角度)
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int64(box) # 包含4个旋转后的顶点坐标
```

### 4.4 多边形拟合与几何形状自动化识别
在制造缺陷检测或机器人视觉抓取中，我们需要判断一个轮廓是三角形、矩形还是圆形。
这通过 `cv2.approxPolyDP()` 算法实现，它基于 Ramer-Douglas-Peucker 算法，用极少的线段来逼近一条连续的曲线边界。
```python
# 精度 epsilon 设为轮廓周长的 3%
epsilon = 0.03 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

# 通过判定拟合后的多边形顶点数量判定几何物理形状！
vertices = len(approx)
if vertices == 3:
    print("这是一个三角形")
elif vertices == 4:
    print("这是一个四边形/矩形")
elif vertices > 6:
    print("这是一个圆形或圆滑的多边形")
```

---

## 五、进阶篇下：计算机视觉经典实战应用

本章将综合前述所有知识，剖析四大计算机视觉工业实战项目。

### 5.1 模板匹配原理与对象定位 (Template Matching)
> **应用场景**：在PCB电路板图上搜寻特定芯片的位置、在游戏中实现自动脚本寻路或图标识别。

**核心原理**：将目标小图像 (模板，Template) 在输入的大图像 (Target) 上像卷积核一样滑动，计算每个位置的匹配相似度。
```python
# 执行匹配
result = cv2.matchTemplate(target_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# minMaxLoc 找出全局极值及坐标点
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 如果使用 TM_CCOEFF_NORMED, 最佳匹配点是 max_loc
top_left = max_loc
```
*提示：如果目标图像受光照、大小缩放或旋转影响，基础模板匹配会失效，此时应使用 SIFT、ORB 等特征点匹配算法。*

### 5.2 霍夫变换检测几何圆与直线 (Hough Cycles/Lines)
霍夫变换能巧妙地利用坐标空间与参数空间的对偶性，将图像空间中检测连续曲线的问题转化为在参数空间中寻找峰值投票的问题。

#### 霍夫圆检测 (`cv2.HoughCircles`)
它基于梯度霍夫算法（Two-Stage Hough Transform），能够极度鲁棒地从嘈杂复杂背景中找出标准圆形目标（如硬币计数、细胞计数）。
```python
# 参数极其讲究：图像, 变换方法, 累加器分辨率比例(dp), 圆心最小间距(minDist), Canny高阈值(param1), 投票峰值门槛(param2), 最小半径, 最大半径
circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=100, param2=30, minRadius=20, maxRadius=100)
```

### 5.3 经典 Haar 级联分类器人脸检测 (Haar Cascade)
在深度学习爆发前，OpenCV 依赖基于 Haar 积分图特征（反映面部明暗边缘关系，如眼窝比鼻梁暗）的 **Adaboost 级联分类器** 统治了人脸识别领域十年之久。
即便是今天，在算力极其受限的微控制器或树莓派上，它依然拥有极高的响应速度。

```python
# 加载内置的正面人脸检测预训练模型 (.xml)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 执行多尺度人脸目标搜索
# scaleFactor: 每次搜索窗口缩小的步长比例; minNeighbors: 候选框至少保留的相邻判定次数
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

### 5.5 实战彩蛋：NEON VISION 实时摄像头工作室
根目录 `webcam_studio.py` 是一套纯 OpenCV 的电影级实时视觉引擎，打开电脑摄像头即可切换 8 种特效：

| 键 | 模式 | 技术要点 |
| :---: | :--- | :--- |
| 1 | CYBER RAIN | 双色霓虹描边 + 字符雨 + 扫描线 + Bloom |
| 2 | GLITCH | RGB 分离、横条撕裂、色块故障 |
| 3 | HOLO MESH | Delaunay 全息网格 + 透视地面 + 旋转环 |
| 4 | FLOW RIBBON | Farneback 光流丝带星云 |
| 5 | WORMHOLE | 吸积盘极坐标扭曲虫洞 |
| 6 | LIGHT PAINT | 运动能量光绘拖尾 |
| 7 | PRISM | 12 瓣棱镜万花筒 + 色散 |
| 8 | PLASMA LOCK | Magma 热成像 + 三角锁定 HUD |

```bash
python3 webcam_studio.py              # 默认摄像头 0
python3 webcam_studio.py --camera 1
python3 webcam_studio.py --demo       # 无摄像头时用合成画面
python3 webcam_studio.py --list       # 扫描本机设备
# WSL2 请看 docs/WSL2_CAMERA.md：Windows 运行 windows_camera_bridge.py 再拉流
```

`S` 截图、`R` 录像，文件写入 `output_images/webcam_studio/`。无 GUI 环境会自动导出 8 模式海报与预览视频。

### 5.4 视频流与摄像头处理架构 (`VideoCapture` / `VideoWriter`)
无论是处理离散的 `.mp4` 视频文件，还是直接连接 USB 工业相机实时捕获流画面，OpenCV 的 `VideoCapture` 框架都提供了一致的抽象。

```python
# 打开默认物理摄像头 (传入索引 0)，或者传入视频文件路径 "video.mp4"
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 逐帧捕获画面 (ret 是布尔值，表示是否成功读取)
    ret, frame = cap.read()
    if not ret:
        break
        
    # **此处可无缝接入上述提到的任何图像预处理/轮廓分析/人脸检测代码！**
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Video Stream", gray_frame)
    
    # 监听键盘输入，按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 优雅释放硬件资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
```

---

## 六、附录：配套演示源码快速运行指南

为了验证学习成果，本教程附带了开箱即用的自动化代码工程。您只需拥有 Python 3 环境，即可一键生成测试案例并查看全部算法执行效果！

### 快速一键执行步骤：
1. **下载或克隆本工程代码。**
2. 在工程根目录下打开终端，运行启动脚本：
   ```bash
   python3 run_all_demos.py
   ```
3. **见证奇迹**：
   - 脚本会自动调用 `demos/generate_test_images.py` 在 `test_images/` 目录下生成标准化测试图案。
   - 依次执行基础操作、图像处理、特征提取和进阶实战 4 个演示模块。
   - 打开 `output_images/` 目录，您将清晰地看到：
     - `01_basic/`: ROI 裁剪、旋转仿射、多边形绘画结果。
     - `02_processing/`: Otsu二值化、各种滤波去噪比对图、形态学修剪结果、CLAHE 对比度增强效果图。
     - `03_features/`: Sobel 综合边缘、Canny 提取图、形状自动化拟合与质心标注结果图。
     - `04_advanced/`: 模板匹配最佳位置框选、霍夫圆完美识别、Haar 级联人脸定位结果及模拟视频流截取。

**至此，您已经完成了 OpenCV 3 从底层图像 IO 到进阶语义检测的全套闭环修炼。祝您在计算机视觉探索之路上硕果累累！**
