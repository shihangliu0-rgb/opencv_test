#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 摄像头 → MJPEG HTTP 桥接（给 WSL2 用）

请在 Windows 的 Python 里运行（不要在 WSL 里跑）：
    pip install opencv-python
    python windows_camera_bridge.py
    python windows_camera_bridge.py --camera 1 --port 8765

WSL2 里：
    WIN_IP=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}')
    python3 webcam_studio.py --source http://${WIN_IP}:8765/mjpeg
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import cv2
except ImportError:
    print("请先在 Windows 安装: pip install opencv-python")
    sys.exit(1)


class FrameHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg = None
        self.ok = False


HUB = FrameHub()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (
                b"<html><body style='background:#111;color:#eee;font-family:sans-serif'>"
                b"<h2>NEON VISION camera bridge</h2>"
                b"<p>WSL: --source http://&lt;WIN_IP&gt;:PORT/mjpeg</p>"
                b"<img src='/mjpeg' style='max-width:96vw'/></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path.startswith("/mjpeg") or self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with HUB.lock:
                        blob = HUB.jpeg
                    if blob:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(blob)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            return

        self.send_error(404)


def grab_loop(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[错误] Windows 打不开摄像头 {index}。试试 --camera 1")
        sys.exit(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print(f"[OK] 已打开 Windows 摄像头 {index}")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        with HUB.lock:
            HUB.jpeg = buf.tobytes()
            HUB.ok = True


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()

    t = threading.Thread(target=grab_loop, args=(args.camera, args.width, args.height), daemon=True)
    t.start()
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    ip = local_ip()
    print("=" * 60)
    print("  Windows 摄像头桥已启动")
    print(f"  本机预览:  http://127.0.0.1:{args.port}/")
    print(f"  WSL 连接:  python3 webcam_studio.py --source http://{ip}:{args.port}/mjpeg")
    print("  （若 WSL 连不上，用 /etc/resolv.conf 里的 nameserver IP）")
    print("=" * 60)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
