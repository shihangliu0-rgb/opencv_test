#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEON VISION — 电影级实时摄像头视觉工作室
==========================================
纯 OpenCV + NumPy，无需深度学习模型。

运行:
    python3 webcam_studio.py
    python3 webcam_studio.py --camera 0
    python3 webcam_studio.py --demo          # 无摄像头时用合成画面

按键:
    1-8  切换视觉模式
    [ ]  上一 / 下一模式
    s    截图保存
    r    开始 / 停止录像
    f    镜像开关
    h    帮助叠加
    q/ESC 退出
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import platform
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple, Union

import cv2
import numpy as np

CaptureSrc = Union[int, str]


def _is_wsl() -> bool:
    if os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def list_v4l_devices() -> List[str]:
    return sorted(glob.glob("/dev/video*"))


def print_wsl_camera_help() -> None:
    print()
    print("=" * 66)
    print("  WSL2 默认看不到 Windows 笔记本摄像头。")
    print("  VideoCapture(0) 打开的是 Linux 的 /dev/video0，不是任务栏里的摄像头。")
    print()
    print("  推荐（最稳）：Windows 推流 + WSL 拉流")
    print("    1) 在 Windows PowerShell（不要在 WSL）运行:")
    print("         pip install opencv-python")
    print("         python windows_camera_bridge.py")
    print("    2) 在 WSL 里:")
    print("         WIN_IP=$(grep -m1 nameserver /etc/resolv.conf | awk '{print $2}')")
    print('         python3 webcam_studio.py --source "http://${WIN_IP}:8765/mjpeg"')
    print()
    print("  说明文档: docs/WSL2_CAMERA.md")
    print("  仅看特效:   python3 webcam_studio.py --demo")
    print("=" * 66)


def try_open_source(src: CaptureSrc, width: int, height: int):
    """按后端优先级打开；成功读到一帧才算数。"""
    backends = []
    if isinstance(src, int) or (isinstance(src, str) and src.startswith("/dev/")):
        if hasattr(cv2, "CAP_V4L2"):
            backends.append(cv2.CAP_V4L2)
        if hasattr(cv2, "CAP_ANY"):
            backends.append(cv2.CAP_ANY)
    else:
        if hasattr(cv2, "CAP_FFMPEG"):
            backends.append(cv2.CAP_FFMPEG)
        backends.append(cv2.CAP_ANY)

    for be in backends:
        cap = cv2.VideoCapture(src, be)
        if not cap.isOpened():
            cap.release()
            continue
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()
    return None


def parse_source(raw: Optional[str], camera: int) -> CaptureSrc:
    if raw is None or raw == "":
        return camera
    if raw.isdigit():
        return int(raw)
    return raw


def cmd_list_cameras() -> int:
    print(f"平台: {platform.platform()}")
    print(f"WSL : {_is_wsl()}")
    devs = list_v4l_devices()
    if devs:
        print("发现 V4L 设备:")
        for d in devs:
            print(f"  {d}")
    else:
        print("未发现 /dev/video*  （WSL2 上这很正常）")
    print("尝试索引 0..3 ...")
    found = False
    for i in range(4):
        cap = try_open_source(i, 320, 240)
        if cap is not None:
            print(f"  索引 {i}: 可读")
            cap.release()
            found = True
        else:
            print(f"  索引 {i}: 不可用")
    if not found and _is_wsl():
        print_wsl_camera_help()
    return 0 if found else 1


# ---------------------------------------------------------------------------
# 视觉模式
# ---------------------------------------------------------------------------
MODES = [
    "CYBER RAIN",     # 0 黑客帝国雨 + 双色霓虹
    "GLITCH",         # 1 RGB 分离 / 数据故障
    "HOLO MESH",      # 2 全息线框人体
    "FLOW RIBBON",    # 3 光流丝带星云
    "WORMHOLE",       # 4 虫洞极坐标
    "LIGHT PAINT",    # 5 运动光绘
    "PRISM",          # 6 棱镜万花筒
    "PLASMA LOCK",    # 7 等离子锁定 HUD
]


def _find_contours(bin_img):
    """兼容 OpenCV 3 / 4 的 findContours。"""
    result = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 3:
        _, contours, hierarchy = result
    else:
        contours, hierarchy = result
    return contours, hierarchy


def neon_glow(edges: np.ndarray, color_bgr: Tuple[int, int, int], bloom: int = 15) -> np.ndarray:
    h, w = edges.shape[:2]
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    layer[edges > 0] = color_bgr
    k = bloom | 1
    glow = cv2.GaussianBlur(layer, (k, k), 0)
    glow = cv2.addWeighted(glow, 2.4, layer, 1.2, 0)
    return np.clip(glow, 0, 255).astype(np.uint8)


def hsv_shift(frame: np.ndarray, hue_delta: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def bloom(img: np.ndarray, thresh: int = 180, amount: float = 0.55) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    hi = cv2.bitwise_and(img, img, mask=bright)
    hi = cv2.GaussianBlur(hi, (0, 0), 8)
    return cv2.addWeighted(img, 1.0, hi, amount, 0)


def vignette(img: np.ndarray, strength: float = 0.55) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    mask = np.clip(1.0 - strength * np.clip(r - 0.15, 0, None) ** 1.6, 0.15, 1.0)
    return (img.astype(np.float32) * mask[..., None]).astype(np.uint8)


def film_grain(img: np.ndarray, t: float, amp: int = 14) -> np.ndarray:
    rng = np.random.default_rng(int(t * 40) % 100000)
    noise = rng.integers(-amp, amp + 1, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def chroma_split(img: np.ndarray, px: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    if px <= 0:
        return out
    out[:, :, 2] = np.roll(img[:, :, 2], px, axis=1)
    out[:, :, 0] = np.roll(img[:, :, 0], -px, axis=1)
    return out


def hue_color(h: float) -> Tuple[int, int, int]:
    c = cv2.cvtColor(np.uint8([[[int(h) % 180, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


_VIGNETTE_CACHE = {}


def cached_vignette_mask(h, w, strength=0.55):
    key = (h, w, strength)
    if key not in _VIGNETTE_CACHE:
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
        _VIGNETTE_CACHE[key] = np.clip(1.0 - strength * np.clip(r - 0.12, 0, None) ** 1.5, 0.12, 1.0).astype(np.float32)
    return _VIGNETTE_CACHE[key]


# ---------------------------------------------------------------------------
# 各模式渲染
# ---------------------------------------------------------------------------
class CyberRain:
    def __init__(self):
        self.cols = None
        self.ys = None
        self.speeds = None
        self.glyphs = list("01アイウエオカキクケコサシスセソ01#$%&<>/\\|")

    def _ensure(self, w, h):
        n = max(24, w // 18)
        if self.cols is not None and len(self.cols) == n:
            return
        rng = np.random.default_rng(7)
        self.cols = rng.integers(0, w, n)
        self.ys = rng.integers(-h, h, n).astype(np.float32)
        self.speeds = rng.uniform(6, 22, n)

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        self._ensure(w, h)
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        c1, c2 = hue_color(t * 35), hue_color(t * 35 + 90)
        layer = np.zeros((h, w, 3), dtype=np.uint8)
        layer[edges > 0] = c1
        thick = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
        layer[thick > 0] = np.maximum(layer[thick > 0], c2)
        glow = cv2.GaussianBlur(layer, (0, 0), 3)
        glow2 = cv2.GaussianBlur(layer, (0, 0), 11)
        dark = (frame.astype(np.float32) * 0.12)
        out = np.clip(dark + glow2 * 1.6 + glow * 0.9 + layer * 0.7, 0, 255).astype(np.uint8)

        rain = np.zeros_like(out)
        for i, x in enumerate(self.cols):
            self.ys[i] = (self.ys[i] + self.speeds[i]) % (h + 80) - 40
            y = int(self.ys[i])
            ch = self.glyphs[(i + int(t * 12)) % len(self.glyphs)]
            g = 80 + int(175 * ((i * 17) % 10) / 10)
            cv2.putText(rain, ch, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, g, 40), 1, cv2.LINE_AA)
            cv2.putText(rain, ch, (int(x), y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
        rain = cv2.GaussianBlur(rain, (3, 3), 0)
        out = cv2.add(out, rain)
        scan = int((t * 140) % h)
        out[scan:scan + 3] = np.clip(out[scan:scan + 3].astype(np.int16) + 70, 0, 255)
        for y in range(0, h, 3):
            out[y] = (out[y].astype(np.int16) * 0.82).astype(np.uint8)
        vm = cached_vignette_mask(h, w, 0.7)
        out = (out.astype(np.float32) * vm[..., None]).astype(np.uint8)
        return bloom(out, 160, 0.7)


class GlitchCore:
    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        out = chroma_split(frame, int(6 + 10 * abs(math.sin(t * 7))))
        # 随机横条撕裂
        rng = np.random.default_rng(int(t * 18) % 99991)
        for _ in range(7):
            y = int(rng.integers(0, h - 12))
            bh = int(rng.integers(4, 18))
            shift = int(rng.integers(-40, 41))
            out[y:y + bh] = np.roll(out[y:y + bh], shift, axis=1)
        # 色块故障
        if rng.random() > 0.35:
            x, y = int(rng.integers(0, w - 80)), int(rng.integers(0, h - 50))
            bw, bh = int(rng.integers(30, 120)), int(rng.integers(12, 50))
            patch = out[y:y + bh, x:x + bw]
            patch = hsv_shift(patch, int(rng.integers(0, 180)))
            out[y:y + bh, x:x + bw] = patch
        # 高对比 + 扫描
        out = cv2.convertScaleAbs(out, alpha=1.25, beta=-10)
        out = chroma_split(out, 4)
        for y in range(0, h, 2):
            out[y] = np.clip(out[y].astype(np.int16) - 18, 0, 255)
        return bloom(vignette(out, 0.45), 200, 0.4)


class HoloMesh:
    def __init__(self):
        self.prev = None

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 取最大前景
        contours, _ = _find_contours(mask)
        canvas = np.zeros_like(frame)
        canvas[:] = (8, 4, 18)
        # 透视网格地面
        for i in range(1, 14):
            y = int(h * 0.45 + i * i * 1.8)
            if y >= h:
                break
            a = int(40 + 180 * (i / 14))
            cv2.line(canvas, (0, y), (w, y), (a, 40, a + 40), 1, cv2.LINE_AA)
        for x in range(-w, w * 2, 50):
            x2 = int(w / 2 + (x - w / 2) * 2.2)
            cv2.line(canvas, (x, int(h * 0.48)), (x2, h), (70, 20, 90), 1, cv2.LINE_AA)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1200:
                hull = cv2.convexHull(c)
                pts = cv2.goodFeaturesToTrack(gray, 90, 0.02, 12)
                if pts is not None and len(pts) >= 4:
                    pts2 = pts.reshape(-1, 2).astype(np.float32)
                    try:
                        rect = (0, 0, w, h)
                        subdiv = cv2.Subdiv2D(rect)
                        for p in pts2:
                            subdiv.insert((float(p[0]), float(p[1])))
                        tris = subdiv.getTriangleList()
                        cyan = hue_color(90 + 20 * math.sin(t * 3))
                        mag = hue_color(150)
                        for tri in tris:
                            p = tri.reshape(3, 2).astype(np.int32)
                            if np.any(p[:, 0] < 0) or np.any(p[:, 0] >= w) or np.any(p[:, 1] < 0) or np.any(p[:, 1] >= h):
                                continue
                            col = cyan if (p[0, 0] + p[0, 1]) % 2 == 0 else mag
                            cv2.polylines(canvas, [p], True, col, 1, cv2.LINE_AA)
                    except cv2.error:
                        pass
                cv2.drawContours(canvas, [hull], -1, (255, 180, 80), 2, cv2.LINE_AA)
                # 旋转环
                M = cv2.moments(c)
                if M["m00"] > 1:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    for k, rad in enumerate((70, 100, 140)):
                        ang = t * (80 + k * 40)
                        box = cv2.boxPoints(((cx, cy), (rad * 2, rad), ang))
                        cv2.polylines(canvas, [np.int32(box)], True, hue_color(40 + k * 40 + t * 20), 1, cv2.LINE_AA)
                    cv2.drawMarker(canvas, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

        edges = cv2.Canny(gray, 50, 130)
        canvas = cv2.add(canvas, neon_glow(edges, (255, 90, 220), 13))
        self.prev = gray
        return bloom(vignette(canvas, 0.5), 140, 0.65)


class FlowRibbon:
    def __init__(self):
        self.prev = None
        self.paint = None

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.paint is None or self.paint.shape[:2] != (h, w):
            self.paint = np.zeros_like(frame, dtype=np.float32)
        self.paint *= 0.88
        if self.prev is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, 0.5, 3, 13, 3, 5, 1.1, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(mag * 18, 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)
            self.paint += rgb
            # 稀疏亮点
            ys, xs = np.where(mag > 2.2)
            if len(xs) > 0:
                step = max(1, len(xs) // 180)
                for x, y in zip(xs[::step], ys[::step]):
                    dx, dy = flow[y, x]
                    col = tuple(int(c) for c in rgb[y, x])
                    cv2.line(self.paint, (x, y), (int(x + dx * 4), int(y + dy * 4)), col, 2, cv2.LINE_AA)
        self.prev = gray
        base = frame.astype(np.float32) * 0.18
        out = np.clip(base + self.paint * 0.085 + cv2.GaussianBlur(self.paint, (0, 0), 7) * 0.05, 0, 255)
        return bloom(vignette(out.astype(np.uint8), 0.5), 150, 0.6)


class Wormhole:
    def __init__(self):
        self.map_x = None
        self.map_y = None
        self.key = None

    def _maps(self, h, w, t):
        key = (h, w, int(t * 8))
        if self.key == key:
            return
        self.key = key
        cy, cx = h / 2, w / 2
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        r = np.sqrt(dx * dx + dy * dy) + 1e-5
        a = np.arctan2(dy, dx) + 0.35 * math.sin(t * 1.4)
        # 吸积盘扭曲
        r2 = np.power(r, 0.72) * (0.85 + 0.15 * np.sin(a * 6 + t * 3))
        self.map_x = (cx + r2 * np.cos(a + 0.0008 * r * math.sin(t))).astype(np.float32)
        self.map_y = (cy + r2 * np.sin(a + 0.0008 * r * math.cos(t * 0.8))).astype(np.float32)

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        self._maps(h, w, t)
        warp = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        warp = hsv_shift(warp, int(t * 22) % 180)
        warp = chroma_split(warp, 5)
        # 中心亮环
        cy, cx = h // 2, w // 2
        pulse = 18 + int(10 * math.sin(t * 5))
        cv2.circle(warp, (cx, cy), pulse, (255, 240, 200), 2, cv2.LINE_AA)
        cv2.circle(warp, (cx, cy), pulse + 30, (80, 40, 255), 1, cv2.LINE_AA)
        return bloom(vignette(warp, 0.65), 170, 0.7)


class LightPaint:
    def __init__(self):
        self.prev = None
        self.canvas = None

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.canvas is None or self.canvas.shape[:2] != (h, w):
            self.canvas = np.zeros((h, w, 3), dtype=np.float32)
        self.canvas *= 0.965
        if self.prev is not None:
            diff = cv2.absdiff(gray, self.prev)
            diff = cv2.GaussianBlur(diff, (7, 7), 0)
            heat = np.clip(diff.astype(np.float32) * 3.2, 0, 255)
            col = np.zeros((h, w, 3), dtype=np.float32)
            hue = (t * 40) % 180
            c = hue_color(hue)
            c2 = hue_color(hue + 70)
            m = heat / 255.0
            col[..., 0] = c[0] * m + c2[0] * (1 - m) * m
            col[..., 1] = c[1] * m
            col[..., 2] = c[2] * m
            self.canvas += col * 1.8
        self.prev = gray
        bg = (frame.astype(np.float32) * 0.08)
        glow = cv2.GaussianBlur(self.canvas, (0, 0), 9)
        out = np.clip(bg + self.canvas + glow * 0.55, 0, 255).astype(np.uint8)
        return bloom(vignette(out, 0.45), 120, 0.8)


class PrismGod:
    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        size = min(h, w)
        y0, x0 = (h - size) // 2, (w - size) // 2
        crop = cv2.resize(frame[y0:y0 + size, x0:x0 + size], (size, size))
        cx = cy = size // 2
        M = cv2.getRotationMatrix2D((cx, cy), (t * 22) % 360, 1.05)
        rot = cv2.warpAffine(crop, M, (size, size))
        yy, xx = np.ogrid[:size, :size]
        ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360) % 360
        sector = ((ang < 30) | (ang > 355)).astype(np.uint8) * 255
        piece = cv2.bitwise_and(rot, rot, mask=sector)
        canvas = np.zeros_like(rot)
        for k in range(12):
            Rm = cv2.getRotationMatrix2D((cx, cy), k * 30, 1.0)
            spun = cv2.warpAffine(piece, Rm, (size, size))
            canvas = np.maximum(canvas, spun)
            canvas = np.maximum(canvas, cv2.flip(spun, 1))
        canvas = hsv_shift(canvas, int(t * 18) % 180)
        canvas = chroma_split(canvas, 6)
        out = np.zeros_like(frame)
        out[y0:y0 + size, x0:x0 + size] = canvas
        # 星芒
        out = cv2.addWeighted(out, 0.85, cv2.GaussianBlur(out, (0, 0), 12), 0.45, 0)
        return bloom(vignette(out, 0.55), 160, 0.75)


class PlasmaLock:
    def __init__(self):
        self.cascade = None
        self.smooth = []
        clf = getattr(cv2, "CascadeClassifier", None)
        haar = getattr(cv2, "data", None)
        if haar is not None and clf is not None:
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            loaded = clf(path)
            if loaded is not None and not loaded.empty():
                self.cascade = loaded

    def _boxes(self, gray):
        faces = []
        if self.cascade is not None:
            faces = self.cascade.detectMultiScale(gray, 1.15, 5, minSize=(50, 50))
        else:
            _, thr = cv2.threshold(cv2.GaussianBlur(gray, (15, 15), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = _find_contours(thr)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 800:
                    faces = [cv2.boundingRect(c)]
        if len(faces):
            self.smooth = [tuple(map(int, f)) for f in faces]
        return self.smooth

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        heat = cv2.applyColorMap(cv2.createCLAHE(3.0, (8, 8)).apply(gray), cv2.COLORMAP_MAGMA)
        heat = cv2.addWeighted(heat, 0.75, frame, 0.15, 0)
        # 六边形 HUD
        out = heat.copy()
        for y in range(0, h, 36):
            off = 18 if (y // 36) % 2 else 0
            for x in range(-off, w, 42):
                cv2.circle(out, (x, y), 16, (40, 10, 50), 1, cv2.LINE_AA)
        boxes = self._boxes(gray)
        for i, (x, y, fw, fh) in enumerate(boxes):
            cx, cy = x + fw // 2, y + fh // 2
            pulse = 0.5 + 0.5 * math.sin(t * 6)
            col = (80, 255, 255)
            for k in range(3):
                cv2.circle(out, (cx, cy), int((max(fw, fh) // 2 + 10 + k * 14) * (0.92 + 0.08 * pulse)), col, 1, cv2.LINE_AA)
            # 三角锁定
            r = max(fw, fh) // 2 + 24
            tri = []
            for a in range(3):
                ang = math.radians(t * 50 + a * 120)
                tri.append((int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))))
            cv2.polylines(out, [np.array(tri)], True, (255, 80, 200), 2, cv2.LINE_AA)
            cv2.putText(out, f"LOCK {i+1}  {fw}px", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            # 侧边频谱
            for b in range(12):
                bh = int(8 + 28 * abs(math.sin(t * 8 + b)))
                cv2.rectangle(out, (x + fw + 10 + b * 7, y + fh - bh), (x + fw + 15 + b * 7, y + fh), (255, 180, 40), -1)
        out = chroma_split(out, 3)
        return bloom(vignette(out, 0.4), 180, 0.5)


# ---------------------------------------------------------------------------
# 合成演示源（无摄像头）
# ---------------------------------------------------------------------------
def synth_frame(t: float, w=960, h=540) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 渐变背景
    for y in range(h):
        img[y, :] = (20, int(18 + 40 * y / h), int(30 + 50 * y / h))
    # 移动的“人脸”椭圆 + 眼睛
    cx = int(w * 0.5 + math.sin(t * 0.8) * w * 0.18)
    cy = int(h * 0.48 + math.cos(t * 0.6) * h * 0.08)
    cv2.ellipse(img, (cx, cy), (90, 120), 0, 0, 360, (70, 90, 180), -1)
    cv2.ellipse(img, (cx, cy), (90, 120), 0, 0, 360, (40, 50, 90), 3)
    cv2.circle(img, (cx - 30, cy - 20), 12, (240, 240, 240), -1)
    cv2.circle(img, (cx + 30, cy - 20), 12, (240, 240, 240), -1)
    cv2.circle(img, (cx - 30, cy - 20), 5, (20, 20, 20), -1)
    cv2.circle(img, (cx + 30, cy - 20), 5, (20, 20, 20), -1)
    cv2.ellipse(img, (cx, cy + 40), (28, 12), 0, 0, 180, (30, 30, 60), 2)
    # 飘动色块供光流
    for i in range(6):
        px = int((t * 80 + i * 140) % (w + 80) - 40)
        py = int(80 + 60 * math.sin(t + i) + i * 50)
        cv2.circle(img, (px, py), 18 + i * 2, (40, 80 + i * 20, 220), -1)
    cv2.putText(img, "DEMO SIGNAL", (24, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
    return img


# ---------------------------------------------------------------------------
# UI 叠加
# ---------------------------------------------------------------------------
def draw_chrome(frame: np.ndarray, mode_idx: int, fps: float, rec: bool, help_on: bool, t: float) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()
    # 底部模式条
    bar_h = 52
    cv2.rectangle(out, (0, h - bar_h), (w, h), (8, 8, 12), -1)
    x = 16
    for i, name in enumerate(MODES):
        active = i == mode_idx
        color = (80, 255, 220) if active else (90, 90, 100)
        label = f"{i+1}:{name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        if active:
            cv2.rectangle(out, (x - 6, h - 42), (x + tw + 6, h - 12), (30, 50, 45), -1)
            cv2.rectangle(out, (x - 6, h - 42), (x + tw + 6, h - 12), color, 1)
        cv2.putText(out, label, (x, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        x += tw + 22

    # 左上角品牌
    cv2.putText(out, "NEON VISION", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (80, 255, 220), 2, cv2.LINE_AA)
    cv2.putText(out, f"{fps:5.1f} FPS", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 210), 1, cv2.LINE_AA)
    if rec:
        if int(t * 2) % 2 == 0:
            cv2.circle(out, (w - 36, 28), 8, (40, 40, 255), -1)
        cv2.putText(out, "REC", (w - 78, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2)

    if help_on:
        lines = [
            "1-8 mode   [ ] prev/next   F mirror",
            "S snapshot   R record   H help   Q quit",
        ]
        y = 86
        for line in lines:
            cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 240), 1, cv2.LINE_AA)
            y += 22
    return out


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
@dataclass
class Studio:
    source: CaptureSrc = 0
    demo: bool = False
    mirror: bool = True
    mode: int = 0
    help_on: bool = True
    width: int = 960
    height: int = 540

    def build_fx(self):
        return [
            CyberRain(),
            GlitchCore(),
            HoloMesh(),
            FlowRibbon(),
            Wormhole(),
            LightPaint(),
            PrismGod(),
            PlasmaLock(),
        ]

    def open_capture(self):
        if self.demo:
            print("[INFO] --demo：使用合成画面，不打开硬件。")
            return None
        candidates = [self.source]
        if self.source == 0:
            candidates.extend(list_v4l_devices())
            candidates.extend(range(1, 4))
        seen, uniq = set(), []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        print(f"[INFO] 尝试打开视频源: {uniq}")
        for src in uniq:
            print(f"[INFO]   -> {src!r}")
            cap = try_open_source(src, self.width, self.height)
            if cap is not None:
                print(f"[OK] 已打开: {src!r}")
                return cap
        print("[WARN] 无法打开任何摄像头 / 视频源，切换到合成演示画面。")
        if _is_wsl() or not list_v4l_devices():
            print_wsl_camera_help()
        self.demo = True
        return None

    def run(self):
        os.makedirs("output_images/webcam_studio", exist_ok=True)
        fx = self.build_fx()
        cap = self.open_capture()
        writer = None
        recording = False
        t0 = time.time()
        fps = 0.0
        ema = None
        snap_i = 0

        win = "NEON VISION — OpenCV Studio"
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            print("[WARN] 未检测到图形界面，生成预览视频后退出。本机运行请直接: python3 webcam_studio.py")
            return self._headless_preview(fx)
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error:
            print("[WARN] 无 GUI 后端，将写入预览视频后退出。")
            return self._headless_preview(fx)

        print("=" * 60)
        print("  NEON VISION  已启动")
        print("  1-8 切换模式 | S 截图 | R 录像 | Q 退出")
        print("=" * 60)

        while True:
            t = time.time() - t0
            tick = time.perf_counter()
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] 读帧失败，改用演示源。")
                    cap.release()
                    cap = None
                    self.demo = True
                    continue
            else:
                frame = synth_frame(t, self.width, self.height)

            if self.mirror and not self.demo:
                frame = cv2.flip(frame, 1)

            # 统一尺寸
            if frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))

            rendered = fx[self.mode](frame, t, None)
            rendered = draw_chrome(rendered, self.mode, fps, recording, self.help_on, t)

            if recording and writer is not None:
                writer.write(rendered)

            cv2.imshow(win, rendered)
            dt = time.perf_counter() - tick
            inst = 1.0 / max(dt, 1e-6)
            ema = inst if ema is None else ema * 0.9 + inst * 0.1
            fps = ema

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("h"), ord("H")):
                self.help_on = not self.help_on
            if key in (ord("f"), ord("F")):
                self.mirror = not self.mirror
            if key in (ord("["),):
                self.mode = (self.mode - 1) % len(MODES)
            if key in (ord("]"),):
                self.mode = (self.mode + 1) % len(MODES)
            if ord("1") <= key <= ord("8"):
                self.mode = key - ord("1")
            if key in (ord("s"), ord("S")):
                path = f"output_images/webcam_studio/snap_{snap_i:03d}_{MODES[self.mode].replace(' ', '_')}.jpg"
                cv2.imwrite(path, rendered)
                print(f"[SNAP] {path}")
                snap_i += 1
            if key in (ord("r"), ord("R")):
                if not recording:
                    path = f"output_images/webcam_studio/rec_{int(time.time())}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(path, fourcc, 24, (rendered.shape[1], rendered.shape[0]))
                    recording = True
                    print(f"[REC] 开始录像 -> {path}")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print("[REC] 已停止")

        if writer:
            writer.release()
        if cap:
            cap.release()
        cv2.destroyAllWindows()

    def _headless_preview(self, fx):
        """无显示器环境：把 8 种模式各渲若干帧写成预览视频 + 海报图。"""
        out_dir = "output_images/webcam_studio"
        os.makedirs(out_dir, exist_ok=True)
        poster_w, poster_h = self.width, self.height
        tiles = []
        path = os.path.join(out_dir, "preview_all_modes.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (poster_w, poster_h))
        for mi, mode in enumerate(MODES):
            self.mode = mi
            for k in range(24):
                t = mi * 1.2 + k / 20.0
                frame = synth_frame(t, poster_w, poster_h)
                rendered = fx[mi](frame, t, None)
                rendered = draw_chrome(rendered, mi, 24.0, False, False, t)
                writer.write(rendered)
                if k == 16:
                    tiles.append(rendered.copy())
                    cv2.imwrite(os.path.join(out_dir, f"mode_{mi+1}_{mode.replace(' ', '_')}.jpg"), rendered)
        writer.release()
        # 2x4 海报
        if len(tiles) == 8:
            row1 = np.hstack([cv2.resize(t, (480, 270)) for t in tiles[:4]])
            row2 = np.hstack([cv2.resize(t, (480, 270)) for t in tiles[4:]])
            poster = np.vstack([row1, row2])
            cv2.imwrite(os.path.join(out_dir, "poster_8modes.jpg"), poster)
        print(f"[INFO] 无摄像头/无 GUI，已生成预览: {out_dir}/")
        return 0


def parse_args():
    p = argparse.ArgumentParser(
        description="NEON VISION — OpenCV 实时摄像头工作室",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python3 webcam_studio.py\n"
            "  python3 webcam_studio.py --source 0\n"
            "  python3 webcam_studio.py --source /dev/video0\n"
            "  python3 webcam_studio.py --source http://172.x.x.x:8765/mjpeg\n"
            "  python3 webcam_studio.py --demo\n"
            "WSL2 说明见 docs/WSL2_CAMERA.md"
        ),
    )
    p.add_argument("--camera", type=int, default=0, help="摄像头索引（兼容旧参数）")
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="视频源：索引 / /dev/video0 / 视频文件 / http MJPEG",
    )
    p.add_argument("--demo", action="store_true", help="强制使用合成演示画面")
    p.add_argument("--list", action="store_true", help="列出本机可打开的摄像头后退出")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--mode", type=int, default=0, help="初始模式 0-7")
    return p.parse_args()


def main():
    args = parse_args()
    if args.list:
        sys.exit(cmd_list_cameras())
    studio = Studio(
        source=parse_source(args.source, args.camera),
        demo=args.demo,
        mirror=not args.no_mirror,
        mode=args.mode % 8,
    )
    studio.run()


if __name__ == "__main__":
    main()
