"""
RAW TCP ストリーム共通ユーティリティ

raw_live_view.py / raw_live_stack.py で共通利用する関数・定数をまとめたモジュール。
"""

import time

import cv2
import numpy as np
import urllib.parse

from display_utils import clamp


# ---------------------------------------------------------------------------
# WB / ガンマ
# ---------------------------------------------------------------------------

def apply_white_balance(bgr8, gains):
    """BGR画像にチャンネルごとのゲインを適用する。RAWデータ自体は変更しない。"""
    f = bgr8.astype(np.float32)
    f[:, :, 0] *= gains[0]  # B
    f[:, :, 1] *= gains[1]  # G
    f[:, :, 2] *= gains[2]  # R
    return np.clip(f, 0, 255).astype(np.uint8)


def apply_gamma_correction(bgr8, gamma):
    """BGR画像に表示ガンマ補正を適用する。"""
    gamma = clamp(float(gamma), 0.10, 4.00)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr8, table)


def compute_wb_gains_from_patch(bgr8, cx, cy, radius=12):
    """クリック周辺パッチから G 基準の WB ゲインを計算する。"""
    h, w = bgr8.shape[:2]
    x0 = clamp(cx - radius, 0, w - 1)
    x1 = clamp(cx + radius + 1, 1, w)
    y0 = clamp(cy - radius, 0, h - 1)
    y1 = clamp(cy + radius + 1, 1, h)
    patch = bgr8[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    means = patch.reshape(-1, 3).mean(axis=0).astype(np.float64)
    b_mean, g_mean, r_mean = float(means[0]), float(means[1]), float(means[2])
    eps = 1e-6
    if g_mean <= eps:
        return None
    b_gain = clamp(g_mean / max(b_mean, eps), 0.10, 4.00)
    g_gain = 1.00
    r_gain = clamp(g_mean / max(r_mean, eps), 0.10, 4.00)
    return [b_gain, g_gain, r_gain]


# ---------------------------------------------------------------------------
# RAW デコード
# ---------------------------------------------------------------------------

def _align32(n):
    return (n + 31) & ~31


def min_stride_for_bits(width, bits):
    """指定ビット深度で必要な最小strideを返す。"""
    if bits == 8:
        return width
    if bits == 10:
        return width * 5 // 4
    if bits == 12:
        return width * 3 // 2
    if bits == 16:
        return width * 2
    raise ValueError(f"未対応ビット数: {bits}")


def calc_candidates(width, height):
    """解像度からフォーマット候補リストを返す（小さい順）。"""
    w, h = width, height
    candidates = []

    stride = w
    candidates.append(dict(name="8bit  SRGGB8     ", bits=8, stride=stride, frame_size=stride * h))

    stride_exact = w * 10 // 8
    stride_aligned = _align32(stride_exact)
    candidates.append(dict(name="10bit CSI2P     ", bits=10, stride=stride_exact, frame_size=stride_exact * h))
    if stride_aligned != stride_exact:
        candidates.append(dict(name="10bit CSI2P+pad", bits=10, stride=stride_aligned, frame_size=stride_aligned * h))

    stride_exact = w * 12 // 8
    stride_aligned = _align32(stride_exact)
    candidates.append(dict(name="12bit CSI2P     ", bits=12, stride=stride_exact, frame_size=stride_exact * h))
    if stride_aligned != stride_exact:
        candidates.append(dict(name="12bit CSI2P+pad", bits=12, stride=stride_aligned, frame_size=stride_aligned * h))

    stride = w * 2
    candidates.append(dict(name="16bit SRGGB16   ", bits=16, stride=stride, frame_size=stride * h))
    return candidates


def unpack_10bit_csi2p(raw_bytes, width, height, stride):
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, stride)
    cols = width * 5 // 4
    d = raw[:, :cols]
    b0 = d[:, 0::5].astype(np.uint16)
    b1 = d[:, 1::5].astype(np.uint16)
    b2 = d[:, 2::5].astype(np.uint16)
    b3 = d[:, 3::5].astype(np.uint16)
    b4 = d[:, 4::5].astype(np.uint16)
    p0 = (b0 << 2) | (b4 & 0x03)
    p1 = (b1 << 2) | ((b4 >> 2) & 0x03)
    p2 = (b2 << 2) | ((b4 >> 4) & 0x03)
    p3 = (b3 << 2) | ((b4 >> 6) & 0x03)
    out = np.empty((height, width), dtype=np.uint16)
    out[:, 0::4] = p0
    out[:, 1::4] = p1
    out[:, 2::4] = p2
    out[:, 3::4] = p3
    return out


def unpack_12bit_csi2p(raw_bytes, width, height, stride):
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, stride)
    cols = width * 3 // 2
    d = raw[:, :cols]
    b0 = d[:, 0::3].astype(np.uint16)
    b1 = d[:, 1::3].astype(np.uint16)
    b2 = d[:, 2::3].astype(np.uint16)
    p0 = (b0 << 4) | (b2 & 0x0F)
    p1 = (b1 << 4) | ((b2 >> 4) & 0x0F)
    out = np.empty((height, width), dtype=np.uint16)
    out[:, 0::2] = p0
    out[:, 1::2] = p1
    return out


def decode_to_raw16(frame_bytes, fmt, width, height):
    bits = fmt["bits"]
    stride = fmt["stride"]
    if bits == 8:
        raw8 = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, stride)
        return raw8[:, :width].astype(np.uint16)  # 下位8bitに格納（10/12bitと統一）
    if bits == 10:
        return unpack_10bit_csi2p(frame_bytes, width, height, stride)
    if bits == 12:
        return unpack_12bit_csi2p(frame_bytes, width, height, stride)
    if bits == 16:
        return np.frombuffer(frame_bytes, dtype=np.uint16).reshape(height, width)
    raise ValueError(f"未対応ビット数: {bits}")


def raw16_to_bgr8(raw16, bits, bayer_code):
    shift = bits - 8
    bayer8 = np.clip(raw16 >> shift, 0, 255).astype(np.uint8)
    return cv2.cvtColor(bayer8, bayer_code)


def raw16_to_bgr8_stretched(raw16, bayer_code, percentile=99.5, gamma=1.0):
    src = raw16.astype(np.float32)
    lo = float(np.percentile(src, 0.5))
    hi = float(np.percentile(src, percentile))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((src - lo) / (hi - lo), 0.0, 1.0)
    gamma_corrected = np.power(normalized, 1.0 / gamma)
    bayer8 = (gamma_corrected * 255.0).astype(np.uint8)
    return cv2.cvtColor(bayer8, bayer_code), lo, hi


# ---------------------------------------------------------------------------
# TCP 受信
# ---------------------------------------------------------------------------

def recv_exact(sock, n, label=""):
    buf = bytearray()
    t_last = time.time()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("接続が切れました")
        buf.extend(chunk)
        now = time.time()
        if now - t_last >= 2.0:
            pct = len(buf) / n * 100
            print(f"  {label}受信中 {len(buf):,} / {n:,} bytes ({pct:.1f}%)", flush=True)
            t_last = now
    return bytes(buf)


def probe_format(sock, width, height, forced_bits=None):
    candidates = calc_candidates(width, height)
    if forced_bits is not None:
        candidates = [c for c in candidates if c["bits"] == forced_bits]
        if not candidates:
            raise ValueError(f"--bits {forced_bits} に一致する候補がありません")

    max_size = max(c["frame_size"] for c in candidates)
    print("\n[probe] フォーマット候補:")
    for c in candidates:
        print(f"  {c['name']}  stride={c['stride']:5d}  frame={c['frame_size']:10d} bytes")

    print(f"\n[probe] {max_size} bytes 受信中... (タイムアウト={sock.gettimeout()}s)")
    t0 = time.time()
    data = recv_exact(sock, max_size, "probe ")
    elapsed = time.time() - t0
    print(f"[probe] 受信完了: {len(data)} bytes  ({elapsed:.1f}s)")
    print(f"[probe] 先頭32bytes (hex): {data[:32].hex(' ')}")

    best = min(candidates, key=lambda c: abs(c["frame_size"] - len(data)))
    print(f"\n[probe] 推定フォーマット: {best['name'].strip()}  (frame_size={best['frame_size']})")

    remaining = bytearray(data[best["frame_size"]:]) if len(data) > best["frame_size"] else bytearray()
    return best, candidates, data[:best["frame_size"]], remaining


def parse_tcp_source(source):
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme not in ("tcp", "udp", ""):
        raise ValueError(f"対応していないスキーム: {parsed.scheme} (例: tcp://192.168.1.17:8888)")
    if not parsed.hostname:
        raise ValueError(f"ホストが指定されていません: {source}")
    if not parsed.port:
        raise ValueError(f"ポートが指定されていません: {source}")
    return parsed.hostname, parsed.port


# ---------------------------------------------------------------------------
# Bayer マップ
# ---------------------------------------------------------------------------

BAYER_MAP = {
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}
