#!/usr/bin/env python3
"""
rpicam-raw TCP生ストリーム専用ライブビュー

送信側 (Raspberry Pi):
    rpicam-raw --width 3840 --height 2160 --framerate 0.3 \
               --shutter 600000 --gain 1 -t 0 --listen -o tcp://0.0.0.0:8888

受信側 (Windows PC, IMX678):
    python raw_live_view.py --source tcp://192.168.1.17:8888 \
        --bits 12 --bayer BGGR --raw-width 3856 --raw-height 2180 --stride 5792

操作:
    [q] 終了
    [s] PNG保存 (WB/表示補正前のデベイヤ後8bit)
    [r] NPY保存 (16bit RAW展開後)
    [n] 次のフォーマット候補に切替
    [+/-] skip ±1行
    [a] 自動ストレッチ ON/OFF
    [b] Bayerパターン切替
    [h] ヒストグラム ON/OFF
"""

import argparse
import os
import socket
import sys
import time
import urllib.parse

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "common"))
from hist_overlay import draw_hist_ccdf_overlay


DISPLAY_MAX_W = 1920
DISPLAY_MAX_H = 1080


def fit_to_display(img, max_w=DISPLAY_MAX_W, max_h=DISPLAY_MAX_H):
    """画像をウィンドウ上限サイズに収まるよう縮小する（アスペクト比維持）。"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_info_lines(img, lines, font_scale=0.60, color=(0, 255, 0), thickness=2):
    """複数行のテキストを左上に重ねる（1行24px間隔）。"""
    for i, line in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def _align32(n):
    return (n + 31) & ~31


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
        return raw8[:, :width].astype(np.uint16) << 8
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


def raw16_to_bgr8_stretched(raw16, bayer_code, percentile=99.5, gamma=2.2):
    lo = float(np.percentile(raw16, 0.5))
    hi = float(np.percentile(raw16, percentile))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((raw16.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    gamma_corrected = np.power(normalized, 1.0 / gamma)
    bayer8 = (gamma_corrected * 255.0).astype(np.uint8)
    return cv2.cvtColor(bayer8, bayer_code), lo, hi


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


BAYER_MAP = {
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}


def run_raw_live_view(args):
    host, port = parse_tcp_source(args.source)

    raw_w = args.raw_width if args.raw_width is not None else args.width
    raw_h = args.raw_height if args.raw_height is not None else args.height
    crop = (raw_w != args.width or raw_h != args.height)

    print(f"[RAW] 接続先: {args.source}")
    print(f"[RAW] 表示解像度: {args.width}x{args.height}")
    if crop:
        print(f"[RAW] RAW解像度: {raw_w}x{raw_h} (デコード後クロップ)")
    print(f"[RAW] Bayer: {args.bayer}  ビット: {'自動' if args.bits is None else args.bits}")
    if args.stride:
        print(f"[RAW] stride: {args.stride} bytes/行")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.timeout)
    sock.connect((host, port))
    print("[RAW] 接続成功")

    try:
        if args.stride is not None:
            if args.bits is None:
                print("[error] --stride 指定時は --bits も必須です")
                return
            fmt = {
                "name": f"{args.bits}bit explicit",
                "bits": args.bits,
                "stride": args.stride,
                "frame_size": args.stride * raw_h,
            }
            candidates = [fmt]
            print(f"\n[RAW] frame_size={fmt['frame_size']} bytes  ({args.bits}bit, stride={args.stride})")
            print(f"[RAW] {fmt['frame_size']} bytes 受信中... (タイムアウト={args.timeout}s)")
            t0 = time.time()
            first_frame_bytes = recv_exact(sock, fmt["frame_size"], "初回フレーム ")
            elapsed = time.time() - t0
            print(f"[RAW] 受信完了 ({elapsed:.1f}s)")
            print(f"[RAW] 先頭32bytes (hex): {first_frame_bytes[:32].hex(' ')}")
            fmt_idx = 0
            buf = bytearray()
        else:
            fmt, candidates, first_frame_bytes, buf = probe_format(sock, raw_w, raw_h, args.bits)
            fmt_idx = candidates.index(fmt)

        skip = args.skip
        auto_stretch = not args.no_stretch
        show_hist = True
        bayer_keys = list(BAYER_MAP.keys())
        bayer_idx = bayer_keys.index(args.bayer)
        bayer_code = BAYER_MAP[bayer_keys[bayer_idx]]

        print("\n操作: [q]終了  [s]PNG保存  [r]NPY保存  [n]次候補フォーマット  [+/-]skip±1行  [a]ストレッチ切替  [b]Bayer切替  [h]ヒストグラム切替")

        frame_count = 0
        last_raw16 = None
        last_bgr = None
        frame_bytes = first_frame_bytes

        while True:
            try:
                payload = frame_bytes[skip:skip + fmt["frame_size"]]
                raw16 = decode_to_raw16(payload, fmt, raw_w, raw_h)
                if crop:
                    raw16 = raw16[:args.height, :args.width]

                bgr = raw16_to_bgr8(raw16, fmt["bits"], bayer_code)
                if auto_stretch:
                    bgr_disp, s_lo, s_hi = raw16_to_bgr8_stretched(raw16, bayer_code)
                else:
                    bgr_disp = bgr
                    s_lo, s_hi = 0.0, float((1 << fmt["bits"]) - 1)

                last_raw16 = raw16
                last_bgr = bgr
                frame_count += 1
            except Exception as e:
                print(f"[decode error] {e}")
                bgr_disp = bgr = np.zeros((args.height // 4, args.width // 4, 3), dtype=np.uint8)
                s_lo, s_hi = 0.0, 1.0

            disp = fit_to_display(bgr_disp)
            src_h, src_w = bgr_disp.shape[:2]
            stretch_label = "stretch" if auto_stretch else "raw"
            bayer_name = bayer_keys[bayer_idx]

            lines = [
                f"{args.source}  {src_w}x{src_h}  frame#{frame_count}",
                f"{fmt['name'].strip()} | Bayer:{bayer_name} | {stretch_label}",
            ]
            if last_raw16 is not None:
                vmin = int(last_raw16.min())
                vmax = int(last_raw16.max())
                vmean = float(last_raw16.mean())
                lines.append(f"min={vmin}  max={vmax}  mean={vmean:.1f}  (max={(1 << fmt['bits']) - 1})")
                if auto_stretch:
                    lines.append(f"stretch[{s_lo:.0f} - {s_hi:.0f}]")
                print(f"[frame#{frame_count}] {'  '.join(lines[2:])}")
            draw_info_lines(disp, lines)

            if show_hist:
                disp = draw_hist_ccdf_overlay(
                    disp,
                    disp,
                    brightness_threshold=255,
                    stop_ratio=0.10,
                )

            cv2.imshow("RAW Live View", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s") and last_bgr is not None:
                fname = f"raw_live_view_frame{frame_count}.png"
                cv2.imwrite(fname, last_bgr)
                print(f"[save] PNG: {fname}")
            elif key == ord("r") and last_raw16 is not None:
                fname = f"raw_live_view_frame{frame_count}_raw16.npy"
                np.save(fname, last_raw16)
                print(f"[save] NPY: {fname}  dtype={last_raw16.dtype}  shape={last_raw16.shape}")
            elif key == ord("a"):
                auto_stretch = not auto_stretch
                print(f"[stretch] {'ON' if auto_stretch else 'OFF'}")
            elif key == ord("h"):
                show_hist = not show_hist
                print(f"[hist] ヒストグラム: {'ON' if show_hist else 'OFF'}")
            elif key == ord("b"):
                bayer_idx = (bayer_idx + 1) % len(bayer_keys)
                bayer_code = BAYER_MAP[bayer_keys[bayer_idx]]
                print(f"[bayer] {bayer_keys[bayer_idx]}")
                if last_raw16 is not None:
                    try:
                        bgr = raw16_to_bgr8(last_raw16, fmt["bits"], bayer_code)
                        if auto_stretch:
                            bgr_disp, s_lo, s_hi = raw16_to_bgr8_stretched(last_raw16, bayer_code)
                        else:
                            bgr_disp = bgr
                        last_bgr = bgr
                    except Exception:
                        pass
            elif key == ord("n"):
                fmt_idx = (fmt_idx + 1) % len(candidates)
                fmt = candidates[fmt_idx]
                print(f"[switch] フォーマット: {fmt['name'].strip()}")
                buf.clear()
            elif key == ord("+") or key == ord("="):
                skip += fmt["stride"]
                print(f"[skip] +{fmt['stride']} -> skip={skip}")
                buf.clear()
            elif key == ord("-"):
                skip = max(0, skip - fmt["stride"])
                print(f"[skip] -{fmt['stride']} -> skip={skip}")
                buf.clear()

            total_size = skip + fmt["frame_size"]
            sock.settimeout(args.timeout)
            while len(buf) < total_size:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("接続が切れました")
                buf.extend(chunk)

            frame_bytes = bytes(buf[:total_size])
            buf = buf[total_size:]

    except ConnectionError as e:
        print(f"\n[切断] {e}")
    except KeyboardInterrupt:
        print("\n[終了]")
    finally:
        sock.close()
        cv2.destroyAllWindows()
        print("RAW Live View終了")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="RAW Live View (rpicam-raw TCP receiver)")
    parser.add_argument("--source", required=True, help="接続先 (例: tcp://192.168.1.17:8888)")
    parser.add_argument("--width", type=int, default=3840, help="表示/クロップ幅 (デフォルト: 3840)")
    parser.add_argument("--height", type=int, default=2160, help="表示/クロップ高さ (デフォルト: 2160)")
    parser.add_argument("--raw-width", type=int, default=None, help="実センサーRAW幅 (省略時=--width。例: IMX678=3856)")
    parser.add_argument("--raw-height", type=int, default=None, help="実センサーRAW高さ (省略時=--height。例: IMX678=2180)")
    parser.add_argument("--stride", type=int, default=None, help="1行あたりのバイト数を直接指定 (例: IMX678=5792)")
    parser.add_argument("--bits", type=int, default=None, choices=[8, 10, 12, 16], help="ビット深度を強制指定 (未指定=自動推定)")
    parser.add_argument("--bayer", type=str, default="BGGR", choices=list(BAYER_MAP.keys()), help="Bayerパターン (デフォルト: BGGR / IMX678確認済み)")
    parser.add_argument("--skip", type=int, default=0, help="フレーム先頭の読み飛ばしバイト数 (通常不要)")
    parser.add_argument("--timeout", type=int, default=120, help="受信タイムアウト秒 (デフォルト: 120)")
    parser.add_argument("--no-stretch", action="store_true", help="自動ストレッチを無効化 (デフォルト: ストレッチ有効)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_raw_live_view(args)


if __name__ == "__main__":
    main()
