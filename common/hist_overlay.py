import cv2
import numpy as np


def draw_hist_ccdf_overlay(target_frame, source_frame, brightness_threshold=255, stop_ratio=0.10):
    """ヒストグラム + 累積ヒストグラム(CCDF)オーバーレイを描画
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

    if len(source_frame.shape) == 3 and source_frame.shape[2] >= 3:
        metric = np.min(source_frame[:, :, :3], axis=2).astype(np.uint8)
    else:
        metric = source_frame.astype(np.uint8)

    hist = cv2.calcHist([metric], [0], None, [256], [0, 256]).flatten()
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

    thr = int(np.clip(brightness_threshold, 0, 255))
    ratio = float(np.clip(stop_ratio, 0.0, 1.0))
    x_thr = gx + int(thr * (gw - 1) / 255)
    y_ratio = gy + gh - int(ratio * (gh - 1))

    cv2.line(target_frame, (x_thr, gy), (x_thr, gy + gh), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(target_frame, (gx, y_ratio), (gx + gw, y_ratio), (0, 255, 255), 2, cv2.LINE_AA)

    overflow_at_thr = float(ccdf[thr])
    y_ccdf = gy + gh - int(overflow_at_thr * (gh - 1))
    cv2.circle(target_frame, (x_thr, y_ccdf), 4, (0, 255, 255), -1)

    cv2.putText(target_frame, "Hist + CCDF", (x0 + 8, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(target_frame, "Hist(norm)", (x0 + 116, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    cv2.putText(target_frame, "CCDF", (x0 + 192, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
    cv2.putText(target_frame, "0", (gx - 12, gy + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(target_frame, "255", (gx + gw - 22, gy + gh + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(target_frame, "1.0", (gx - 28, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(target_frame, "0.0", (gx - 28, gy + gh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    info_y = gy + gh + 30
    cv2.putText(target_frame, f"T={thr}", (x0 + 8, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
    cv2.putText(
        target_frame,
        f"R={int(ratio * 100)}%  P(X>=T)={overflow_at_thr * 100:.1f}%",
        (x0 + 68, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
    )

    return target_frame
