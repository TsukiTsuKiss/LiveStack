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
    stride_aligned = _align32(stride)
    candidates.append(dict(name="16bit SRGGB16   ", bits=16, stride=stride, frame_size=stride * h))
    if stride_aligned != stride:
        candidates.append(dict(name="16bit SRGGB16+pad", bits=16, stride=stride_aligned, frame_size=stride_aligned * h))
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
        raw16 = np.frombuffer(frame_bytes, dtype=np.uint16).reshape(height, stride // 2)
        return raw16[:, :width]
    raise ValueError(f"未対応ビット数: {bits}")


def apply_wire_format_preset(args, raw_width):
    """wire-formatプリセットに応じて args.bits / args.stride を補完・検証する。"""
    wire_format = getattr(args, "wire_format", None)
    if wire_format is None:
        return None

    if wire_format == "12p":
        expected_bits = 12
        expected_min_stride = _align32(min_stride_for_bits(raw_width, 12))
    elif wire_format == "12u":
        expected_bits = 16
        expected_min_stride = _align32(min_stride_for_bits(raw_width, 16))
    elif wire_format == "10u":
        expected_bits = 16
        expected_min_stride = _align32(min_stride_for_bits(raw_width, 16))
    elif wire_format == "16u":
        expected_bits = 16
        expected_min_stride = _align32(min_stride_for_bits(raw_width, 16))
    else:
        raise ValueError(f"未対応の wire-format: {wire_format}")

    if getattr(args, "bits", None) is None:
        args.bits = expected_bits
    elif int(args.bits) != expected_bits:
        raise ValueError(
            f"--wire-format {wire_format} では --bits {expected_bits} が必要です (指定: {args.bits})"
        )

    if getattr(args, "stride", None) is None:
        args.stride = expected_min_stride
    elif int(args.stride) < expected_min_stride:
        raise ValueError(
            f"--wire-format {wire_format} では stride>={expected_min_stride} が必要です (指定: {args.stride})"
        )

    return {
        "wire_format": wire_format,
        "bits": int(args.bits),
        "stride": int(args.stride),
        "effective_bits": 12 if wire_format == "12u" else (10 if wire_format == "10u" else int(args.bits)),
    }


def resolve_effective_bits_and_shift(args, fmt, width, height, first_frame_bytes):
    """wire-format設定から有効ビット深度とシフト量を決定する。"""
    wire_format = getattr(args, "wire_format", None)
    if wire_format == "12u":
        effective_bits = 12
        manual_shift = getattr(args, "u12_shift", "auto")
        manual_choices = ("0", "4")
        low_mask = 0x000F
        high_mask = 0xF000
        auto_tag = "u12"
        msb_shift = 4
    elif wire_format == "10u":
        effective_bits = 10
        manual_shift = getattr(args, "u10_shift", "auto")
        manual_choices = ("0", "6")
        low_mask = 0x003F
        high_mask = 0xFC00
        auto_tag = "u10"
        msb_shift = 6
    else:
        return {
            "effective_bits": int(fmt["bits"]),
            "effective_shift": 0,
            "message": None,
        }

    if manual_shift in manual_choices:
        return {
            "effective_bits": effective_bits,
            "effective_shift": int(manual_shift),
            "message": f"[{auto_tag}] 有効ビット位置を手動指定: >>{int(manual_shift)}",
        }

    sample16 = decode_to_raw16(first_frame_bytes, fmt, width, height)
    low_zero_ratio = float(np.mean((sample16 & low_mask) == 0))
    high_zero_ratio = float(np.mean((sample16 & high_mask) == 0))
    if low_zero_ratio > 0.98 and high_zero_ratio < 0.98:
        effective_shift = msb_shift
    elif high_zero_ratio > 0.98 and low_zero_ratio < 0.98:
        effective_shift = 0
    else:
        effective_shift = msb_shift if low_zero_ratio >= high_zero_ratio else 0

    return {
        "effective_bits": effective_bits,
        "effective_shift": effective_shift,
        "message": (
            f"[{auto_tag}] auto判定: shift=>>{effective_shift} "
            f"(low=0 ratio={low_zero_ratio:.3f}, high=0 ratio={high_zero_ratio:.3f})"
        ),
    }


def normalize_effective_raw16(raw16, fmt_bits, effective_bits, effective_shift):
    """16bitコンテナ入力時に有効ビットへ正規化する。"""
    if int(fmt_bits) == 16 and int(effective_bits) < 16:
        if int(effective_shift) > 0:
            raw16 = raw16 >> int(effective_shift)
        raw16 = raw16 & ((1 << int(effective_bits)) - 1)
    return raw16


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
