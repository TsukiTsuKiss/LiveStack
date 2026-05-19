"""
表示ユーティリティ

live_stack.py / raw_live_view.py / raw_live_stack.py で共通利用する
画面サイズ・表示フレーム関連の汎用ユーティリティ。
"""

import os

import cv2
import numpy as np


def get_screen_size():
    """画面サイズを取得（取得不可時はNone）"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception:
            pass
    return None


def fit_display_frame(frame, screen_size=None, ratio=0.85, fallback_height=600):
    """表示フレームのみ画面サイズに収まるよう縮小（内部処理用フレームは変更しない）"""
    h, w = frame.shape[:2]
    if screen_size is not None:
        max_w = int(screen_size[0] * ratio)
        max_h = int(screen_size[1] * ratio)
    else:
        max_h = fallback_height
        max_w = int((w / max(1, h)) * max_h)
    if w <= max_w and h <= max_h:
        return frame
    scale = min(max_w / max(1, w), max_h / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_info_lines(img, lines, font_scale=0.60, color=(0, 255, 0), thickness=2):
    """複数行のテキストを左上に重ねる（1行24px間隔）。"""
    for i, line in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def clamp(value, low, high):
    return max(low, min(high, value))
