"""ズームPiP(拡大した切り出し画像を隅に重ねる)の描画（raw_live_stack.py / live_stack.py 共通）"""
import cv2


def draw_zoom_pip(preview_frame, pip_crop, pip_size, corner, margin=8):
    """pip_crop を最大8倍(1倍以上)に拡大して preview_frame の隅に重ね、(ox, oy, 幅, 高さ) を返す。収まらなければNone。"""
    ph, pw = preview_frame.shape[:2]
    scale_f = max(1.0, min(8.0, pip_size / max(1, max(pip_crop.shape[0], pip_crop.shape[1]))))
    new_w = max(1, round(pip_crop.shape[1] * scale_f))
    new_h = max(1, round(pip_crop.shape[0] * scale_f))
    zoomed = cv2.resize(pip_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    zh, zw = zoomed.shape[:2]
    ox = margin if corner in ("tl", "bl") else pw - zw - margin
    oy = margin if corner in ("tl", "tr") else ph - zh - margin
    if ox < 0 or oy < 0:
        return None
    cv2.rectangle(preview_frame, (ox - 1, oy - 1), (ox + zw, oy + zh), (0, 255, 255), 1)
    preview_frame[oy:oy + zh, ox:ox + zw] = zoomed
    return ox, oy, zw, zh
