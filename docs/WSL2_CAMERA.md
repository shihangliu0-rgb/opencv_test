# WSL2 Ubuntu 里用摄像头（NEON VISION）

## 这两条警告是什么意思？

```
GStreamer: pipeline have not been created
VIDEOIO(V4L2:/dev/video0): can't open camera by index
[WARN] 无法打开摄像头，自动切换到合成演示源。
```

程序打开的是 **Linux 虚拟机里的摄像头 0**，对应设备文件 `/dev/video0`。  
它**不会自动去找 Windows 任务栏里的笔记本摄像头**。

WSL2 是一台独立的 Linux 虚拟机：

| 环境 | 摄像头从哪来 |
| :--- | :--- |
| Windows 本机 Python | 直接用笔记本 / USB 摄像头 |
| 实体 Ubuntu / 双系统 | `/dev/video0` 通常就是摄像头 |
| **WSL2 Ubuntu 22.04** | **默认没有摄像头设备**，所以 `VideoCapture(0)` 必失败 |

GStreamer 那行可以忽略：OpenCV 先试 GStreamer，失败再试 V4L2，V4L2 也打不开就报第二行。

先在 WSL 里确认：

```bash
ls -l /dev/video*
# 若提示 No such file or directory —— 说明 Linux 侧根本没有摄像头
```

---

## 推荐做法（最稳）：Windows 推流，WSL 拉流

不必编译内核、不必 usbipd。Windows 把摄像头打成 MJPEG，WSL 用 HTTP 读。

### 1) Windows 安装 Python + OpenCV

在 **PowerShell（Windows 侧，不是 WSL）**：

```powershell
pip install opencv-python
```

### 2) 在仓库里启动桥接

```powershell
cd \\wsl$\Ubuntu\home\<你的用户名>\opencv_test
python windows_camera_bridge.py
```

或把 `windows_camera_bridge.py` 拷到 Windows 目录再运行。默认监听 `0.0.0.0:8765`。

浏览器打开 `http://127.0.0.1:8765/` 能看到画面即成功。

### 3) WSL 里连 Windows 主机 IP

```bash
WIN_IP=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}')
echo "Windows 主机: $WIN_IP"
python3 webcam_studio.py --source "http://${WIN_IP}:8765/mjpeg"
```

若连不上，在 **Windows 防火墙** 放行 8765，或临时：

```powershell
netsh advfirewall firewall add rule name="NEON CAM" dir=in action=allow protocol=TCP localport=8765
```

---

## 备选：usbipd 把 USB 摄像头挂进 WSL（折腾）

仅对 **USB 外接摄像头** 相对靠谱。笔记本内置摄像头很多机型挂不进去。

1. Windows 安装 [usbipd-win](https://github.com/dorssel/usbipd-win)
2. 管理员 PowerShell：

```powershell
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

3. WSL 里还需要 **带 UVC / V4L 的自定义内核**（官方内核经常没有 `uvcvideo`）。参考：  
   https://github.com/PINTO0309/wsl2_linux_kernel_usbcam_enable_conf

4. 出现 `/dev/video0` 后再：

```bash
python3 webcam_studio.py --source /dev/video0
```

内置摄像头失败、`select() timeout` 都是常见坑，这时请改用上面的 MJPEG 桥接。

---

## 其它命令

```bash
python3 webcam_studio.py --list              # 扫描本机 /dev/video* 和索引
python3 webcam_studio.py --demo              # 合成画面，不碰硬件
python3 webcam_studio.py --source video.mp4  # 用视频文件当输入
```

想零配置用真摄像头：在 **Windows 原生 Python** 里直接 `python webcam_studio.py` 即可，不要走 WSL。
