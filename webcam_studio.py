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
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 视觉模式
# ---------------------------------------------------------------------------
MODES = [
    "NEON EDGE",      # 0 赛博霓虹描边
    "STAR FLOW",      # 1 光流星尘
    "FACE HUD",       # 2 科幻人脸 HUD
    "KALEIDO",        # 3 万花筒
    "THERMAL",        # 4 热成像伪彩
    "GHOST TRAIL",    # 5 残影拖尾
    "INK CARTOON",    # 6 水墨卡通
    "AURORA MIX",     # 7 极光混合
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
    """把二值边缘变成带辉光的霓虹层。"""
    h, w = edges.shape[:2]
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    layer[edges > 0] = color_bgr
    k = bloom | 1
    glow = cv2.GaussianBlur(layer, (k, k), 0)
    glow = cv2.addWeighted(glow, 1.8, layer, 1.0, 0)
    return np.clip(glow, 0, 255).astype(np.uint8)


def hsv_shift(frame: np.ndarray, hue_delta: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta) % 180
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# 各模式渲染
# ---------------------------------------------------------------------------
class NeonEdge:
    def __call__(self, frame, t, state):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 140)
        hue = int((t * 40) % 180)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        glow = neon_glow(edges, tuple(int(c) for c in color), bloom=21)
        dark = (frame.astype(np.float32) * 0.18).astype(np.uint8)
        out = cv2.add(dark, glow)
        # 扫描线
        scan = int((t * 80) % frame.shape[0])
        out[scan : scan + 2] = np.clip(out[scan : scan + 2].astype(np.int16) + 40, 0, 255)
        return out


class StarFlow:
    def __init__(self):
        self.prev_gray = None
        self.pts: Optional[np.ndarray] = None
        self.trails: List[Deque] = []
        self.lk = dict(
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03),
        )

    def __call__(self, frame, t, state):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        canvas = (frame.astype(np.float32) * 0.25).astype(np.uint8)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.pts = cv2.goodFeaturesToTrack(gray, 280, 0.01, 8)
            self.trails = [deque(maxlen=14) for _ in range(len(self.pts) or 0)]
            return canvas

        if self.pts is None or len(self.pts) < 40:
            self.pts = cv2.goodFeaturesToTrack(gray, 280, 0.01, 8)
            self.trails = [deque(maxlen=14) for _ in range(len(self.pts) or 0)]

        if self.pts is not None and len(self.pts):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.pts, None, **self.lk)
            if nxt is not None and st is not None:
                good_new = nxt[st.flatten() == 1]
                good_old = self.pts[st.flatten() == 1]
                # 对齐 trails
                new_trails = []
                gi = 0
                for i, keep in enumerate(st.flatten()):
                    if keep and gi < len(good_new):
                        p = tuple(good_new[gi].ravel())
                        if i < len(self.trails):
                            tr = self.trails[i]
                            tr.append(p)
                            new_trails.append(tr)
                        else:
                            new_trails.append(deque([p], maxlen=14))
                        gi += 1
                self.trails = new_trails

                for i, (n, o) in enumerate(zip(good_new, good_old)):
                    x1, y1 = n.ravel()
                    x0, y0 = o.ravel()
                    speed = math.hypot(x1 - x0, y1 - y0)
                    hue = int(min(179, 20 + speed * 18))
                    col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
                    col = tuple(int(c) for c in col)
                    if i < len(self.trails) and len(self.trails[i]) > 1:
                        pts = np.array(self.trails[i], dtype=np.int32)
                        cv2.polylines(canvas, [pts], False, col, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (int(x1), int(y1)), 2, (255, 255, 255), -1, cv2.LINE_AA)
                self.pts = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray
        # 星空闪点
        rng = np.random.default_rng(int(t * 3) % 10000)
        for _ in range(18):
            sx, sy = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.circle(canvas, (sx, sy), 1, (180, 220, 255), -1)
        return canvas


class FaceHUD:
    def __init__(self):
        self.cascade = None
        self.smooth = []  # list of (x,y,w,h)
        haar = getattr(cv2, "data", None)
        clf = getattr(cv2, "CascadeClassifier", None)
        if haar is not None and clf is not None:
            path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            loaded = clf(path)
            if loaded is not None and not loaded.empty():
                self.cascade = loaded

    def _smooth(self, faces):
        if len(faces) == 0:
            self.smooth = [(int(x * 0.7 + sx * 0.3), int(y * 0.7 + sy * 0.3),
                            int(w * 0.7 + sw * 0.3), int(h * 0.7 + sh * 0.3))
                           for (x, y, w, h), (sx, sy, sw, sh) in zip(self.smooth, self.smooth)]
            self.smooth = [b for b in self.smooth if b[2] > 20]
            return self.smooth
        self.smooth = [tuple(map(int, f)) for f in faces]
        return self.smooth

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        out = frame.copy()
        # 暗角 + 青蓝调色
        out = cv2.convertScaleAbs(out, alpha=0.85, beta=-8)
        overlay = out.copy()
        overlay[:, :, 0] = np.clip(overlay[:, :, 0].astype(np.int16) + 18, 0, 255)
        out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if self.cascade is not None:
            faces = self.cascade.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
        else:
            # OpenCV 5 无 Haar 时：用运动/亮度椭圆当锁定目标
            _, thr = cv2.threshold(cv2.GaussianBlur(gray, (15, 15), 0), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = _find_contours(thr)
            if contours:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 800:
                    faces = [cv2.boundingRect(c)]
        boxes = self._smooth(faces)

        # 网格
        for x in range(0, w, 48):
            cv2.line(out, (x, 0), (x, h), (40, 70, 70), 1)
        for y in range(0, h, 48):
            cv2.line(out, (0, y), (w, y), (40, 70, 70), 1)

        cyan = (255, 220, 80)
        for i, (x, y, fw, fh) in enumerate(boxes):
            # 角标框
            l = max(12, fw // 8)
            pts = [
                ((x, y), (x + l, y), (x, y + l)),
                ((x + fw, y), (x + fw - l, y), (x + fw, y + l)),
                ((x, y + fh), (x + l, y + fh), (x, y + fh - l)),
                ((x + fw, y + fh), (x + fw - l, y + fh), (x + fw, y + fh - l)),
            ]
            for a, b, c in pts:
                cv2.line(out, a, b, cyan, 2, cv2.LINE_AA)
                cv2.line(out, a, c, cyan, 2, cv2.LINE_AA)
            # 十字准星
            cx, cy = x + fw // 2, y + fh // 2
            cv2.drawMarker(out, (cx, cy), cyan, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), max(fw, fh) // 2 + 8, cyan, 1, cv2.LINE_AA)
            pulse = 0.5 + 0.5 * math.sin(t * 4)
            cv2.circle(out, (cx, cy), int((max(fw, fh) // 2 + 16) * (0.9 + 0.1 * pulse)), (180, 160, 40), 1)
            label = f"SUBJ-{i+1:02d}  LOCK  {fw}x{fh}"
            cv2.putText(out, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 1, cv2.LINE_AA)
            # 侧边数据条
            cv2.putText(out, f"ID {hash((x, y)) % 9000 + 1000}", (x + fw + 8, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
            cv2.putText(out, f"CONF {0.82 + 0.08 * pulse:.2f}", (x + fw + 8, y + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

        # 顶部 HUD 条
        cv2.rectangle(out, (0, 0), (w, 36), (10, 18, 18), -1)
        cv2.putText(out, f"NEON VISION  //  FACIAL ACQUISITION  //  T+{t:06.1f}s  //  TARGETS {len(boxes)}",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan, 1, cv2.LINE_AA)
        return out


class Kaleido:
    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        size = min(h, w)
        y0, x0 = (h - size) // 2, (w - size) // 2
        crop = frame[y0:y0 + size, x0:x0 + size]
        # 取三角扇区并镜像拼成 8 瓣
        cx, cy = size // 2, size // 2
        angle = (t * 18) % 360
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(crop, M, (size, size), flags=cv2.INTER_LINEAR)
        mask = np.zeros((size, size), dtype=np.uint8)
        wedge = np.array([[cx, cy], [size, 0], [size, size // 6]], dtype=np.int32)
        # 更稳：用极坐标扇区
        yy, xx = np.ogrid[:size, :size]
        ang = (np.degrees(np.arctan2(yy - cy, xx - cx)) + 360) % 360
        sector = ((ang < 45) | (ang > 360 - 1)).astype(np.uint8) * 255
        piece = cv2.bitwise_and(rot, rot, mask=sector)

        canvas = np.zeros_like(rot)
        for k in range(8):
            Rm = cv2.getRotationMatrix2D((cx, cy), k * 45, 1.0)
            spun = cv2.warpAffine(piece, Rm, (size, size))
            canvas = np.maximum(canvas, spun)
            flipped = cv2.flip(spun, 1)
            canvas = np.maximum(canvas, flipped)

        # 放回画布并加暗角
        out = np.zeros_like(frame)
        out[y0:y0 + size, x0:x0 + size] = canvas
        out = hsv_shift(out, int(t * 12) % 180)
        return out


class Thermal:
    def __call__(self, frame, t, state):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # 轻微 CLAHE 让层次更戏剧
        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray = clahe.apply(gray)
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        # 热点等高线
        edges = cv2.Canny(gray, 80, 160)
        heat[edges > 0] = (255, 255, 255)
        return heat


class GhostTrail:
    def __init__(self):
        self.buf: Deque[np.ndarray] = deque(maxlen=12)

    def __call__(self, frame, t, state):
        self.buf.append(frame.copy())
        acc = np.zeros_like(frame, dtype=np.float32)
        n = len(self.buf)
        for i, f in enumerate(self.buf):
            w = (i + 1) / n
            # 越旧越偏品红
            tint = f.astype(np.float32)
            tint[:, :, 2] *= 0.6 + 0.4 * w
            tint[:, :, 0] *= 1.2 - 0.2 * w
            acc += tint * w
        acc /= sum((i + 1) / n for i in range(n))
        out = np.clip(acc, 0, 255).astype(np.uint8)
        return cv2.addWeighted(out, 0.85, frame, 0.15, 0)


class InkCartoon:
    def __call__(self, frame, t, state):
        # 双边滤波多次 = 扁平色块
        smooth = frame
        for _ in range(2):
            smooth = cv2.bilateralFilter(smooth, 7, 50, 50)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 2)
        edges_c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(smooth, edges_c)
        # 水墨纸感
        paper = cv2.cvtColor(cv2.cvtColor(cartoon, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        mix = cv2.addWeighted(cartoon, 0.65, paper, 0.35, 0)
        return mix


class AuroraMix:
    def __init__(self):
        self.prev = None

    def __call__(self, frame, t, state):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 流动极光层
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        wave = (np.sin(xx * 0.01 + t * 1.7) + np.sin(yy * 0.02 - t * 1.1)) * 0.5
        aurora = np.zeros_like(frame, dtype=np.float32)
        aurora[:, :, 1] = (80 + 80 * wave).clip(0, 255)          # G
        aurora[:, :, 0] = (40 + 60 * np.sin(wave + t)).clip(0, 255)  # B
        aurora[:, :, 2] = (20 + 40 * np.cos(wave - t)).clip(0, 255)  # R

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag = np.clip(mag * 12, 0, 1)
            motion = np.dstack([mag, mag, mag])
            aurora = aurora * (0.35 + 0.65 * motion)
        self.prev = gray

        base = (frame.astype(np.float32) * 0.55)
        out = np.clip(base + aurora * 0.85, 0, 255).astype(np.uint8)
        edges = cv2.Canny(gray, 60, 130)
        out = cv2.add(out, neon_glow(edges, (180, 255, 120), bloom=11))
        return out


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
    camera: int = 0
    demo: bool = False
    mirror: bool = True
    mode: int = 0
    help_on: bool = True
    width: int = 960
    height: int = 540

    def build_fx(self):
        return [
            NeonEdge(),
            StarFlow(),
            FaceHUD(),
            Kaleido(),
            Thermal(),
            GhostTrail(),
            InkCartoon(),
            AuroraMix(),
        ]

    def open_capture(self):
        if self.demo:
            return None
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            print("[WARN] 无法打开摄像头，自动切换到合成演示源。")
            self.demo = True
            return None
        return cap

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
    p = argparse.ArgumentParser(description="NEON VISION — OpenCV 实时摄像头工作室")
    p.add_argument("--camera", type=int, default=0, help="摄像头索引")
    p.add_argument("--demo", action="store_true", help="强制使用合成演示画面")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--mode", type=int, default=0, help="初始模式 0-7")
    return p.parse_args()


def main():
    args = parse_args()
    studio = Studio(camera=args.camera, demo=args.demo, mirror=not args.no_mirror, mode=args.mode % 8)
    studio.run()


if __name__ == "__main__":
    main()
