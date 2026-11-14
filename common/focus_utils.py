"""フォーカスアシストの指標計算とパネル描画（raw_live_stack.py / live_stack.py 共通）"""
from collections import deque

import cv2
import numpy as np

FOCUS_HISTORY = 80       # 履歴フレーム数
FOCUS_SAT_RATIO = 0.95   # ピークがこの割合以上なら飽和扱い
FOCUS_MIN_SIZE = 16      # 測定領域の最小の辺(px)
FOCUS_PANEL_HEIGHT = 84
FOCUS_STARS = 5          # 星モードで同時に監視する星の数
STAR_RADIUS = 12         # 星の測定窓の半径(ビニング後px)
STAR_DETECT_SIGMA = 8.0  # 星候補の検出しきい値(平滑化後の背景ノイズσ)
STAR_MIN_SNR = 5.0       # 測定窓内のピークの最小SNR(未満は見失い扱い)
STAR_MIN_HFR = 0.5       # これ未満はホットピクセル扱い(ビニング後px)
STAR_REDETECT = 15       # 星が足りないとき再検出する間隔(フレーム)


def new_focus_state():
    return {"on": False, "mode": "star", "shape": "circle", "hist": deque(maxlen=FOCUS_HISTORY), "best": 0.0,
            "key": None, "fc": -1, "score": 0.0, "peak": 0.0, "sat": False,
            "stars": [], "star_hfr": [], "k": 1, "age": STAR_REDETECT}


def reset_focus_state(state):
    state.update(best=0.0, key=None, fc=-1, score=0.0, sat=False, stars=[], star_hfr=[], age=STAR_REDETECT)
    state["hist"].clear()


def focus_mode_label(state):
    """メニュー表示用のモード名(cv2.putTextは日本語不可なのでASCII)。"""
    if not state["on"]:
        return "OFF"
    if state["mode"] == "grad":
        return "Gradient"
    return "Star (circle)" if state["shape"] == "circle" else "Star (box)"


def cycle_focus_mode(state):
    """OFF→星(円)→星(四角)→勾配→OFF と切り替え、状態名を返す。"""
    if not state["on"]:
        state.update(on=True, mode="star", shape="circle")
    elif state["mode"] == "star" and state["shape"] == "circle":
        state["shape"] = "box"
    elif state["mode"] == "star":
        state["mode"] = "grad"
    else:
        state["on"] = False
    reset_focus_state(state)
    if not state["on"]:
        return "OFF"
    if state["mode"] == "grad":
        return "勾配"
    return "星(円)" if state["shape"] == "circle" else "星(四角)"


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


def _crop_binned(img, bx0, by0, bx1, by1, k):
    """ビニング後座標の矩形を切り出し、(float32, 元画素の最大値, 実x0, 実y0)を返す。"""
    h, w = img.shape[:2]
    bx0, by0 = max(0, bx0), max(0, by0)
    bx1, by1 = min(w // k, bx1), min(h // k, by1)
    if bx1 <= bx0 or by1 <= by0:
        return None
    raw = img[by0 * k:by1 * k, bx0 * k:bx1 * k]
    crop = raw.astype(np.float32)
    if k == 2:
        crop = crop.reshape(by1 - by0, 2, bx1 - bx0, 2).mean(axis=(1, 3))
    return crop, float(raw.max()), bx0, by0


def _measure_star(img, cx, cy, k, max_value):
    """(cx, cy)周りの星の(HFR, 重心x, 重心y, 飽和か)をビニング後pxで返す。見失い・端に近いときはNone。"""
    r = STAR_RADIUS
    ix, iy = int(round(cx)), int(round(cy))
    res = _crop_binned(img, ix - r, iy - r, ix + r + 1, iy + r + 1, k)
    n = 2 * r + 1
    if res is None or res[0].shape != (n, n):
        return None
    crop, peak, x0, y0 = res
    ring = np.concatenate([crop[0], crop[-1], crop[1:-1, 0], crop[1:-1, -1]])
    bg = float(np.median(ring))
    sigma = max(1.4826 * float(np.median(np.abs(ring - bg))), 0.5)
    sub = crop - bg
    sub[sub < 2 * sigma] = 0
    flux = float(sub.sum())
    if flux <= 0 or float(sub.max()) < STAR_MIN_SNR * sigma:
        return None
    yy, xx = np.indices((n, n), dtype=np.float32)
    mx, my = float((sub * xx).sum()) / flux, float((sub * yy).sum()) / flux
    dist = np.hypot(xx - mx, yy - my).ravel()
    order = np.argsort(dist)
    cum = np.cumsum(sub.ravel()[order])
    hfr = float(np.interp(0.5 * flux, cum, dist[order]))
    return hfr, x0 + mx, y0 + my, peak >= FOCUS_SAT_RATIO * max_value


def _detect_stars(img, bounds, k, max_value, exclude, limit):
    """bounds(ビニング後座標)内の孤立した非飽和の星を明るい順に最大limit個、(x, y, HFR)で返す。"""
    res = _crop_binned(img, *bounds, k)
    if res is None or limit <= 0:
        return []
    f, _, ox, oy = res
    if min(f.shape) < 8:
        return []
    h, w = f.shape
    small = cv2.resize(f, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4), (w, h), interpolation=cv2.INTER_LINEAR)
    d = cv2.GaussianBlur(f, (0, 0), 1.5) - bg
    med = float(np.median(d))
    sigma = max(1.4826 * float(np.median(np.abs(d - med))), 0.5)
    peaks = (d >= cv2.dilate(d, np.ones((7, 7), np.uint8))) & (d > med + STAR_DETECT_SIGMA * sigma)
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-d[ys, xs])[:200]
    vals = d[ys[order], xs[order]]
    xs, ys = xs[order] + ox, ys[order] + oy
    found = []
    taken = [tuple(p) for p in exclude]
    for i in range(len(xs)):
        x, y = float(xs[i]), float(ys[i])
        near = np.hypot(xs - x, ys - y) < 1.5 * STAR_RADIUS
        near[i] = False
        if np.any(vals[near] >= 0.2 * vals[i]):
            continue
        if any(np.hypot(px - x, py - y) < 2 * STAR_RADIUS for px, py in taken):
            continue
        m = _measure_star(img, x, y, k, max_value)
        if m is None or m[3] or m[0] < STAR_MIN_HFR:
            continue
        found.append((m[1], m[2], m[0]))
        taken.append((m[1], m[2]))
        if len(found) >= limit:
            break
    return found


def _update_stars(state, img, region, max_value, bin2):
    """監視中の星を再測定して中央値HFR(元画素px)を更新する。足りなければ検出し直す。"""
    k = 2 if bin2 else 1
    state["k"] = k
    kept, infos, hfrs = [], [], []
    for cx, cy in state["stars"]:
        m = _measure_star(img, cx, cy, k, max_value)
        if m is None or np.hypot(m[1] - cx, m[2] - cy) > STAR_RADIUS / 2:
            continue
        kept.append((m[1], m[2]))
        infos.append(None if m[3] else m[0] * k)
        if not m[3]:
            hfrs.append(m[0] * k)
    target = FOCUS_STARS if region is None else 1
    state["age"] += 1
    if len(kept) < target and state["age"] >= STAR_REDETECT:
        state["age"] = 0
        h, w = img.shape[:2]
        bounds = (0, 0, w // k, h // k) if region is None else tuple(int(v) // k for v in region)
        for x, y, hfr in _detect_stars(img, bounds, k, max_value, kept, target - len(kept)):
            kept.append((x, y))
            infos.append(hfr * k)
            hfrs.append(hfr * k)
    state["stars"], state["star_hfr"] = kept, infos
    state["sat"] = bool(kept) and not hfrs
    if not hfrs:
        state["score"] = 0.0
        return
    score = float(np.median(hfrs))
    state["score"] = score
    state["best"] = score if state["best"] <= 0 else min(state["best"], score)
    state["hist"].append(score)


def update_focus_state(state, img, region, max_value, stamp, bin2=True):
    """測定領域が変わったら履歴をリセットし、stamp(フレーム番号等)が新しいときだけ指標を更新する。
    星モードでは region=None で画面全体から複数の星を自動検出し、指定時はその中の最も明るい1個を測る。"""
    if region != state["key"]:
        reset_focus_state(state)
        state["key"] = region
    if stamp == state["fc"]:
        return
    state["fc"] = stamp
    if state["mode"] == "star":
        _update_stars(state, img, region, max_value, bin2)
        return
    res = focus_metric(img, region, max_value, bin2)
    if res is not None:
        state["score"], state["peak"], state["sat"] = res
        state["best"] = max(state["best"], res[0])
        state["hist"].append(res[0])


def draw_focus_stars(img, focus, to_view, scale, gain=3.0, min_r=5, color=(0, 255, 255), sat_color=(0, 0, 255)):
    """監視中の星を円(半径=HFR×scale×gain)または測定窓の四角で囲みHFR(元画素px)を添える。
    to_view(x, y)は入力画像座標→img座標、scaleは入力1pxあたりのimg上のpx。"""
    k = focus["k"]
    for (cx, cy), hfr in zip(focus["stars"], focus["star_hfr"]):
        px, py = to_view(cx * k, cy * k)
        col = sat_color if hfr is None else color
        if focus["shape"] == "circle":
            half = min_r if hfr is None else max(min_r, round(hfr * scale * gain))
            cv2.circle(img, (px, py), half, col, 1, cv2.LINE_AA)
        else:
            half = max(4, round(STAR_RADIUS * k * scale))
            cv2.rectangle(img, (px - half, py - half), (px + half, py + half), col, 1)
        cv2.putText(img, "SAT" if hfr is None else f"{hfr:.1f}", (px - half, py - half - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)


def draw_focus_stars_pip(img, focus, pip_pos, zoom_region, frame_size, flip_h, flip_v):
    """Zoom PiP内(拡大表示)に実寸に近い円(HFR×2)を描く。白い星でも見えるよう赤、SATはマゼンタ。zoom_regionは表示(反転後)px。"""
    ox, oy, pw, ph = pip_pos
    zx0, zy0, zx1, zy1 = zoom_region
    w, h = frame_size
    sx, sy = pw / (zx1 - zx0), ph / (zy1 - zy0)
    draw_focus_stars(
        img[oy:oy + ph, ox:ox + pw], focus,
        lambda x, y: (round(((w - x if flip_h else x) - zx0) * sx), round(((h - y if flip_v else y) - zy0) * sy)),
        sx, gain=2.0, min_r=3, color=(0, 0, 255), sat_color=(255, 0, 255))


def draw_focus_panel(img, x, y, focus, width=230):
    """フォーカスパネル(現在値・ベスト・バー・履歴)を img の (x, y) に描く。星モードはHFR(小さいほど良い)。"""
    ph, pw = img.shape[:2]
    height = FOCUS_PANEL_HEIGHT
    x = max(0, min(x, pw - width))
    y = max(0, min(y, ph - height))
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 255), 1)
    star = focus["mode"] == "star"
    best = max(focus["best"], 1e-6)
    score = focus["score"]
    sat = focus["sat"]
    # 勾配は大きいほど、HFRは小さいほど良い。どちらも1.0=ベストに揃える
    ratio = (lambda v: min(1.0, best / v) if v > 0 else 0.0) if star else (lambda v: min(1.0, v / best))
    if star:
        title = f"HFR {score:.2f}px" if score > 0 else "HFR --"
        right, right_color = ("SAT", (0, 0, 255)) if sat else ((f"n={len(focus['stars'])}", (200, 200, 200)) if focus["stars"] else ("no star", (0, 0, 255)))
    else:
        title = f"FOCUS {score:.2f}"
        right, right_color = ("SAT", (0, 0, 255)) if sat else (f"pk {focus['peak']:.0f}", (200, 200, 200))
    cv2.putText(img, title, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"best {focus['best']:.2f}", (x + 6, y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, right, (x + width - 70, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1, cv2.LINE_AA)
    bar = int((width - 12) * ratio(score))
    cv2.rectangle(img, (x + 6, y + 42), (x + 6 + bar, y + 50), (0, 255, 0), -1)
    hist = list(focus["hist"])
    if len(hist) >= 2:
        gx0, gy0, gh = x + 6, y + 56, 22
        step = (width - 12) / (FOCUS_HISTORY - 1)
        pts = np.array([[gx0 + round(i * step), gy0 + gh - round(gh * ratio(v))]
                        for i, v in enumerate(hist)], dtype=np.int32)
        cv2.polylines(img, [pts], False, (0, 220, 0), 1, cv2.LINE_AA)
