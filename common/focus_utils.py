"""フォーカスアシストの指標計算とパネル描画（raw_live_stack.py / live_stack.py 共通）"""
from collections import deque

import cv2
import numpy as np

FOCUS_HISTORY = 80       # 履歴フレーム数
FOCUS_SAT_RATIO = 0.95   # ピークがこの割合以上なら飽和扱い
FOCUS_MIN_SIZE = 16      # 測定領域の最小の辺(px)
FOCUS_PANEL_HEIGHT = 84


def new_focus_state():
    return {"on": False, "hist": deque(maxlen=FOCUS_HISTORY), "best": 0.0, "key": None, "fc": -1,
            "score": 0.0, "peak": 0.0, "sat": False}


def reset_focus_state(state):
    state.update(best=0.0, key=None, fc=-1)
    state["hist"].clear()


def focus_metric(img, region, max_value, bin2=True):
    """領域(x0,y0,x1,y1)の(勾配RMS, ピーク, 飽和か)を返す。小さすぎるときはNone。
    bin2=True は生Bayer用に2x2ビニングして色配列の影響を消す。"""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in region)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if bin2:
        x0, y0, x1, y1 = x0 // 2 * 2, y0 // 2 * 2, x1 // 2 * 2, y1 // 2 * 2
    if x1 - x0 < FOCUS_MIN_SIZE or y1 - y0 < FOCUS_MIN_SIZE:
        return None
    crop = img[y0:y1, x0:x1].astype(np.float32)
    if bin2:
        ch, cw = crop.shape
        crop = crop.reshape(ch // 2, 2, cw // 2, 2).mean(axis=(1, 3))
    blur = cv2.GaussianBlur(crop, (3, 3), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    score = float(np.sqrt(np.mean(gx * gx + gy * gy)))
    peak = float(crop.max())
    return score, peak, peak >= FOCUS_SAT_RATIO * max_value


def update_focus_state(state, img, region, max_value, stamp, bin2=True):
    """測定領域が変わったら履歴をリセットし、stamp(フレーム番号等)が新しいときだけ指標を更新する。"""
    if region != state["key"]:
        reset_focus_state(state)
        state["key"] = region
    if stamp == state["fc"]:
        return
    state["fc"] = stamp
    res = focus_metric(img, region, max_value, bin2)
    if res is not None:
        state["score"], state["peak"], state["sat"] = res
        state["best"] = max(state["best"], res[0])
        state["hist"].append(res[0])


def draw_focus_panel(img, x, y, focus, width=230):
    """フォーカスパネル(現在値・ベスト・バー・履歴)を img の (x, y) に描く。"""
    ph, pw = img.shape[:2]
    height = FOCUS_PANEL_HEIGHT
    x = max(0, min(x, pw - width))
    y = max(0, min(y, ph - height))
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 255), 1)
    best = max(focus["best"], 1e-6)
    sat = focus["sat"]
    cv2.putText(img, f"FOCUS {focus['score']:.2f}", (x + 6, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"best {focus['best']:.2f}", (x + 6, y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "SAT" if sat else f"pk {focus['peak']:.0f}", (x + width - 70, y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if sat else (200, 200, 200), 1, cv2.LINE_AA)
    bar = int((width - 12) * min(1.0, focus["score"] / best))
    cv2.rectangle(img, (x + 6, y + 42), (x + 6 + bar, y + 50), (0, 255, 0), -1)
    hist = list(focus["hist"])
    if len(hist) >= 2:
        gx0, gy0, gh = x + 6, y + 56, 22
        step = (width - 12) / (FOCUS_HISTORY - 1)
        pts = np.array([[gx0 + round(i * step), gy0 + gh - round(gh * min(1.0, v / best))]
                        for i, v in enumerate(hist)], dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 220, 0), 1, cv2.LINE_AA)
