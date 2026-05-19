import cv2
import numpy as np


def draw_hist_ccdf_overlay(target_frame, source_frame, brightness_threshold=255, stop_ratio=0.10, bits=8, native_source=None, stretch_lo=None, stretch_hi=None):
    """ヒストグラム + 累積ヒストグラム(CCDF)オーバーレイを描画
    - native_source: 2D配列(float32/uint16)を渡すとネイティブbit深度でヒストグラムを計算
    - native_source=None のときは source_frame(uint8 BGR)をフォールバックとして使用
    - stretch_lo/stretch_hi: ストレッチ範囲（ネイティブbit値）を橙色マーカーで表示
    - 垂直線: 輝度しきい値
    - 水平線: 停止比率
    """
    if target_frame is None or source_frame is None:
        return target_frame

    th, tw = target_frame.shape[:2]
    graph_w, graph_h = 340, 190
    margin = 12
    x0 = tw - graph_w - margin
    y0 = margin

    if x0 < 0 or y0 + graph_h >= th:
        return target_frame

    overlay = target_frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + graph_w, y0 + graph_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, target_frame, 0.45, 0, target_frame)

    max_native = (1 << bits) - 1
    ratio = float(np.clip(stop_ratio, 0.0, 1.0))

    if native_source is not None and bits > 8:
        # ネイティブbit深度（12/16bit）でヒストグラムを計算
        data = native_source.flatten().astype(np.float32)
        hist, _ = np.histogram(data, bins=256, range=(0.0, float(max_native + 1)))
        hist = hist.astype(np.float32)
        thr_native = int(np.clip(brightness_threshold, 0, max_native))
        thr_bin = int(thr_native * 255 / max_native)  # 0-255のbin indexに変換
    else:
        # フォールバック: uint8 BGR source_frame を使用
        if len(source_frame.shape) == 3 and source_frame.shape[2] >= 3:
            metric = np.max(source_frame[:, :, :3], axis=2).astype(np.uint8)
        else:
            metric = source_frame.astype(np.uint8)
        hist = cv2.calcHist([metric], [0], None, [256], [0, 256]).flatten()
        thr_bin = int(np.clip(brightness_threshold, 0, 255))
        thr_native = thr_bin << max(0, bits - 8)

    total = float(np.sum(hist))
    if total <= 0:
        return target_frame

    ccdf = np.cumsum(hist[::-1])[::-1] / total
    hist_log = np.log1p(hist)
    hist_norm = hist_log / max(1.0, float(np.max(hist_log)))

    pad_l, pad_r, pad_t, pad_b = 36, 16, 22, 42
    gx = x0 + pad_l
    gy = y0 + pad_t
    gw = graph_w - pad_l - pad_r
    gh = graph_h - pad_t - pad_b

    cv2.rectangle(target_frame, (gx, gy), (gx + gw, gy + gh), (120, 120, 120), 2)

    hist_points = []
    for i in range(256):
        px = gx + int(i * (gw - 1) / 255)
        py = gy + gh - int(hist_norm[i] * (gh - 1))
        hist_points.append([px, py])
    cv2.polylines(target_frame, [np.array(hist_points, dtype=np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)

    points = []
    for i in range(256):
        px = gx + int(i * (gw - 1) / 255)
        py = gy + gh - int(ccdf[i] * (gh - 1))
        points.append([px, py])
    cv2.polylines(target_frame, [np.array(points, dtype=np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)

    x_thr = gx + int(thr_bin * (gw - 1) / 255)
    y_ratio = gy + gh - int(ratio * (gh - 1))

    cv2.line(target_frame, (x_thr, gy), (x_thr, gy + gh), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(target_frame, (gx, y_ratio), (gx + gw, y_ratio), (0, 255, 255), 2, cv2.LINE_AA)

    # ストレッチ範囲マーカー: 半透明エリア + x軸目盛エリアに ▲(lo) △(hi) を描画
    if stretch_lo is not None and stretch_hi is not None and bits > 8:
        lo_bin = int(np.clip(float(stretch_lo) * 255 / max_native, 0, 255))
        hi_bin = int(np.clip(float(stretch_hi) * 255 / max_native, 0, 255))
        x_lo = gx + int(lo_bin * (gw - 1) / 255)
        x_hi = gx + int(hi_bin * (gw - 1) / 255)
        # 半透明塗りつぶし
        stretch_overlay = target_frame.copy()
        cv2.rectangle(stretch_overlay, (x_lo, gy), (x_hi, gy + gh), (0, 128, 255), -1)
        cv2.addWeighted(stretch_overlay, 0.15, target_frame, 0.85, 0, target_frame)
        # ▲ lo: 塗りつぶし三角（頂点が上、x軸を指す）
        ty = gy + gh + 2
        ts = 6
        tri_lo = np.array([[x_lo, ty], [x_lo - ts, ty + ts], [x_lo + ts, ty + ts]], dtype=np.int32)
        cv2.fillPoly(target_frame, [tri_lo], (0, 165, 255))
        # △ hi: 輪郭のみ三角
        tri_hi = np.array([[x_hi, ty], [x_hi - ts, ty + ts], [x_hi + ts, ty + ts]], dtype=np.int32)
        cv2.polylines(target_frame, [tri_hi], True, (0, 165, 255), 1, cv2.LINE_AA)

    overflow_at_thr = float(ccdf[min(thr_bin, 255)])
    y_ccdf = gy + gh - int(overflow_at_thr * (gh - 1))
    cv2.circle(target_frame, (x_thr, y_ccdf), 4, (0, 255, 255), -1)

    cv2.putText(target_frame, "Hist + CCDF", (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(target_frame, "Hist(norm)", (x0 + 116, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    cv2.putText(target_frame, "CCDF", (x0 + 192, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
    cv2.putText(target_frame, "0", (gx - 12, gy + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(target_frame, str(max_native), (gx + gw - 22, gy + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(target_frame, "1.0", (gx - 28, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(target_frame, "0.0", (gx - 28, gy + gh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    info_y = gy + gh + 30
    cv2.putText(target_frame, f"T={thr_native}", (x0 + 8, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(
        target_frame,
        f"R={int(ratio * 100)}%  P(X>=T)={overflow_at_thr * 100:.2f}%",
        (x0 + 68, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
    )

    return target_frame
