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
    [s] PNG保存 (WB適用後。ストレッチON時は表示と同じトーン)
    [r] NPY保存 (16bit RAW展開後)
    [n] 次のフォーマット候補に切替
    [+/-] skip ±1行
    [a] 自動ストレッチ ON/OFF
    [b] Bayerパターン切替
    [h] ヒストグラム ON/OFF
    [w] 白点クリックWB ON/OFF
    [W] WBリセット (B/G/R=1.0)
    [g] ガンマ調整モード ON/OFF（左右キーで変更）
    [G] ガンマリセット (0.80)
"""

import argparse
import os
import socket
import sys
import time

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "common"))
from hist_overlay import draw_hist_ccdf_overlay
from display_utils import clamp, draw_info_lines, fit_display_frame, get_screen_size
from raw_utils import (
    BAYER_MAP,
    apply_gamma_correction,
    apply_white_balance,
    calc_candidates,
    compute_wb_gains_from_patch,
    decode_to_raw16,
    min_stride_for_bits,
    parse_tcp_source,
    probe_format,
    raw16_to_bgr8,
    raw16_to_bgr8_stretched,
    recv_exact,
)


PREVIEW_WINDOW_NAME = "RAW Live View"
WB_WINDOW_NAME = "RAW WB"


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

        wb_gains = [1.0, 1.0, 1.0]
        gamma_value = 0.80
        gamma_adjust_mode = False
        click_wb_mode = False
        screen_size = get_screen_size()
        syncing_wb_trackbar = {"active": False}
        mouse_ctx = {
            "base_image": None,
            "disp_shape": (1, 1),
            "pending_msg": None,
        }

        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WB_WINDOW_NAME, cv2.WINDOW_NORMAL)

        def slider_to_gain(v):
            return clamp(v / 100.0, 0.10, 4.00)

        def gain_to_slider(g):
            return int(round(clamp(g, 0.10, 4.00) * 100.0))

        def slider_to_gamma(v):
            return clamp(v / 100.0, 0.10, 4.00)

        def gamma_to_slider(g):
            return int(round(clamp(g, 0.10, 4.00) * 100.0))

        def sync_wb_trackbars():
            syncing_wb_trackbar["active"] = True
            cv2.setTrackbarPos("B x100", WB_WINDOW_NAME, gain_to_slider(wb_gains[0]))
            cv2.setTrackbarPos("G x100", WB_WINDOW_NAME, gain_to_slider(wb_gains[1]))
            cv2.setTrackbarPos("R x100", WB_WINDOW_NAME, gain_to_slider(wb_gains[2]))
            cv2.setTrackbarPos("Gamma x100", WB_WINDOW_NAME, gamma_to_slider(gamma_value))
            syncing_wb_trackbar["active"] = False

        cv2.createTrackbar("B x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("G x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("R x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("Gamma x100", WB_WINDOW_NAME, 80, 400, lambda _v: None)
        sync_wb_trackbars()

        def on_mouse(event, x, y, _flags, _userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if not click_wb_mode:
                return
            base = mouse_ctx["base_image"]
            if base is None:
                return

            disp_h, disp_w = mouse_ctx["disp_shape"]
            src_h, src_w = base.shape[:2]
            sx = clamp(int(x * src_w / max(1, disp_w)), 0, src_w - 1)
            sy = clamp(int(y * src_h / max(1, disp_h)), 0, src_h - 1)

            gains = compute_wb_gains_from_patch(base, sx, sy, radius=12)
            if gains is None:
                return

            wb_gains[0], wb_gains[1], wb_gains[2] = gains
            sync_wb_trackbars()
            mouse_ctx["pending_msg"] = (
                f"[wb] click ({sx},{sy}) -> B={wb_gains[0]:.2f} G={wb_gains[1]:.2f} R={wb_gains[2]:.2f}"
            )

        cv2.setMouseCallback(PREVIEW_WINDOW_NAME, on_mouse)

        print("\n操作: [q]終了  [s]PNG保存  [r]NPY保存  [n]次候補フォーマット  [+/-]skip±1行  [a]ストレッチ切替  [b]Bayer切替  [h]ヒストグラム切替  [w]白点WB  [W]WBリセット  [g]ガンマ調整モード  [G]ガンマリセット")

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

                if not syncing_wb_trackbar["active"]:
                    wb_gains[0] = slider_to_gain(max(10, cv2.getTrackbarPos("B x100", WB_WINDOW_NAME)))
                    wb_gains[1] = slider_to_gain(max(10, cv2.getTrackbarPos("G x100", WB_WINDOW_NAME)))
                    wb_gains[2] = slider_to_gain(max(10, cv2.getTrackbarPos("R x100", WB_WINDOW_NAME)))
                    gamma_value = slider_to_gamma(max(10, cv2.getTrackbarPos("Gamma x100", WB_WINDOW_NAME)))

                # RAWそのものには触らず、デベイヤ後画像にのみWBを適用
                bgr = apply_white_balance(bgr, wb_gains)
                bgr_disp = apply_white_balance(bgr_disp, wb_gains)
                bgr = apply_gamma_correction(bgr, gamma_value)
                bgr_disp = apply_gamma_correction(bgr_disp, gamma_value)

                last_raw16 = raw16
                last_bgr = bgr_disp
                frame_count += 1
            except Exception as e:
                print(f"[decode error] {e}")
                bgr_disp = bgr = np.zeros((args.height // 4, args.width // 4, 3), dtype=np.uint8)
                s_lo, s_hi = 0.0, 1.0

            disp = fit_display_frame(bgr_disp, screen_size=screen_size)
            src_h, src_w = bgr_disp.shape[:2]
            stretch_label = "stretch" if auto_stretch else "raw"
            bayer_name = bayer_keys[bayer_idx]

            lines = [
                f"{args.source}  {src_w}x{src_h}  frame#{frame_count}",
                f"{fmt['name'].strip()} | Bayer:{bayer_name} | {stretch_label}",
                f"WB B:{wb_gains[0]:.2f} G:{wb_gains[1]:.2f} R:{wb_gains[2]:.2f} | Gamma:{gamma_value:.2f} ({'KEY' if gamma_adjust_mode else 'GUI'}) | click:{'ON' if click_wb_mode else 'OFF'}",
            ]
            if last_raw16 is not None:
                vmin = int(last_raw16.min())
                vmax = int(last_raw16.max())
                vmean = float(last_raw16.mean())
                lines.append(f"min={vmin}  max={vmax}  mean={vmean:.1f}  (max={(1 << fmt['bits']) - 1})")
                if auto_stretch:
                    lines.append(f"stretch[{s_lo:.0f} - {s_hi:.0f}]")
                print(f"[frame#{frame_count}] {'  '.join(lines[3:])}")
            draw_info_lines(disp, lines)

            # クリックWB用に、オーバーレイ前の元表示画像を保持
            mouse_ctx["base_image"] = bgr_disp
            mouse_ctx["disp_shape"] = disp.shape[:2]
            if mouse_ctx["pending_msg"]:
                print(mouse_ctx["pending_msg"])
                mouse_ctx["pending_msg"] = None

            if show_hist:
                hist_native = last_raw16[::4, ::4].astype(np.float32) if last_raw16 is not None else None
                disp = draw_hist_ccdf_overlay(
                    disp,
                    disp,
                    brightness_threshold=255,
                    stop_ratio=0.10,
                    bits=fmt["bits"],
                    native_source=hist_native,
                )

            cv2.imshow(PREVIEW_WINDOW_NAME, disp)
            key_code = cv2.waitKeyEx(1)
            key = key_code & 0xFF

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
            elif key == ord("w"):
                click_wb_mode = not click_wb_mode
                print(f"[wb] 白点クリックWB: {'ON' if click_wb_mode else 'OFF'}")
            elif key == ord("W"):
                wb_gains[0], wb_gains[1], wb_gains[2] = 1.0, 1.0, 1.0
                sync_wb_trackbars()
                print("[wb] reset -> B=1.00 G=1.00 R=1.00")
            elif key == ord("g"):
                gamma_adjust_mode = not gamma_adjust_mode
                print(f"[gamma] 調整モード: {'ON' if gamma_adjust_mode else 'OFF'} (左右キーで変更)")
            elif key == ord("G"):
                gamma_value = 0.80
                sync_wb_trackbars()
                print("[gamma] reset -> 0.80")
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
                        bgr = apply_white_balance(bgr, wb_gains)
                        bgr_disp = apply_white_balance(bgr_disp, wb_gains)
                        bgr = apply_gamma_correction(bgr, gamma_value)
                        bgr_disp = apply_gamma_correction(bgr_disp, gamma_value)
                        last_bgr = bgr_disp
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

            if gamma_adjust_mode:
                # OpenCVの矢印キーコード（Windows/Linux）
                if key_code in (2424832, 81):  # Left
                    gamma_value = clamp(gamma_value - 0.05, 0.10, 4.00)
                    sync_wb_trackbars()
                    print(f"[gamma] {gamma_value:.2f}")
                elif key_code in (2555904, 83):  # Right
                    gamma_value = clamp(gamma_value + 0.05, 0.10, 4.00)
                    sync_wb_trackbars()
                    print(f"[gamma] {gamma_value:.2f}")

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
