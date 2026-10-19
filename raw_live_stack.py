#!/usr/bin/env python3
"""
rpicam-raw TCP生ストリーム専用ライブスタック

送信側 (Raspberry Pi):
    rpicam-raw --width 3840 --height 2160 --framerate 0.3 \
               --shutter 600000 --gain 1 -t 0 --listen -o tcp://0.0.0.0:8888

受信側 (Windows PC, IMX678):
    python raw_live_stack.py --source tcp://192.168.1.17:8888 \
        --bits 12 --bayer BGGR --raw-width 3856 --raw-height 2180 --stride 5792

操作:
    [q] 終了
    [m] 設定メニュー
    [i] 情報表示 ON/OFF
    [s] PNG保存 (WB/ガンマ適用後、tEXtメタデータ付き)
    [j] JPEG保存 (WB/ガンマ適用後、EXIF付き)
    [c] 操作メニュー (保存/SER録画/WB・ガンマ・スタックリセット/ズーム/ダーク/JSON保存をメニューから実行)
    [y] 画面録画AVI (info/ヒストグラム/メニュー等のオーバーレイ込み、MJPG)
    [S]/[x] SER保存 (バッファ全フレームをマルチフレームSER)
    [r] NPY保存 (16bit RAW展開後)
    [f] FITS保存
    [n] 次のフォーマット候補に切替
    [+/-] skip ±1行
    [a] 自動ストレッチ ON/OFF
    [A] 位置合わせ ON/OFF
    [E] ダーク減算 ON/OFF
    [space] SER入力: 一時停止/再生
    [,]/[.] SER入力: 1コマ戻る/進む
    [[]/[]] SER入力: 先頭/末尾へジャンプ (Home/Endも可)
    [l] SER入力: ループ ON/OFF
    [b] Bayerパターン切替
    [h] 左右反転 ON/OFF
    [v] 上下反転 ON/OFF
    [H] ヒストグラム ON/OFF
    [t] LiveStack ON/OFF
    [R] LiveStackリセット
    [w] 白点クリックWB ON/OFF
    [W]/[u] WBリセット (B/G/R=1.0)
    [g] ガンマ調整モード ON/OFF（左右キーで変更）
    [G] ガンマリセット (0.80)
    [d] ダークフレーム取得（レンズキャップして押す、複数回で加算平均）
    [D] ダークフレームクリア（終了時に保存されなくなる）
    [C] 設定をJSONに保存（--config 指定時はそのパス、未指定時は config.json）
    ※ダークは終了時に dark.fits へ自動保存、次回起動時に自動読み込み
"""

import argparse
import datetime
import json
import os
import re
import shutil
import socket
import sys
import threading
import time
import traceback

import cv2
import numpy as np
import piexif
from PIL import Image, PngImagePlugin

# pip版OpenCV(Qt)はフォント同梱が無くトラックバーの文字が描画されないため、システムフォントを指す
if sys.platform.startswith("linux"):
    for _font_dir in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype", "/usr/share/fonts"):
        if os.path.isdir(_font_dir):
            os.environ["QT_QPA_FONTDIR"] = _font_dir
            break

sys.path.append(os.path.join(os.path.dirname(__file__), "common"))
from hist_overlay import draw_hist_ccdf_overlay
from display_utils import clamp, decode_key, draw_info_lines, fit_display_frame, get_screen_size
from ser_writer import SerWriter
from ui_menus import ControlMenu, ScreenRecorder, SettingsMenu, route_control_key, toggle_settings_menu
from raw_utils import (
    BAYER_MAP,
    apply_wire_format_preset,
    apply_gamma_correction,
    apply_white_balance,
    calc_candidates,
    compute_wb_gains_from_patch,
    decode_to_raw16,
    estimate_max_frames_limit,
    get_available_memory_bytes,
    min_stride_for_bits,
    normalize_effective_raw16,
    parse_tcp_source,
    probe_format,
    raw16_to_bgr8,
    raw16_to_bgr8_stretched,
    recv_exact,
    resolve_effective_bits_and_shift,
)


PREVIEW_WINDOW_NAME = "RAW Live Stack"
WB_WINDOW_NAME = "RAW Stack WB"
# スタック方式: 内部名 <-> メニュー表示名
STACK_METHOD_LABELS = {"sum": "Sum", "lighten": "Lighten", "darken": "Darken", "median": "Median", "sigma": "Sigma Clip", "sigmasum": "Sigma Sum"}
STACK_LABEL_TO_METHOD = {v: k for k, v in STACK_METHOD_LABELS.items()}
DARK_FLIP_LABELS = {"none": "None", "h": "H", "v": "V", "hv": "HV", "auto": "Auto"}
DARK_FLIP_LABEL_TO_MODE = {v: k for k, v in DARK_FLIP_LABELS.items()}
SER_LOOP_LABELS = {"reset": "ON（先頭でスタックをリセット）", "continuous": "ON（リセットしない・加算を続ける）", "off": "OFF"}
ZOOM_WINDOW_NAME = "RAW Live Stack - Zoom"


# ---------------------------------------------------------------------------
# 表示ユーティリティ（raw_live_stack 固有）
# ---------------------------------------------------------------------------

def draw_ccdf_overlay(target_frame, source_frame, brightness_threshold=255, stop_ratio=0.10, bits=8, native_source=None, stretch_lo=None, stretch_hi=None):
    return draw_hist_ccdf_overlay(
        target_frame,
        source_frame,
        brightness_threshold=brightness_threshold,
        stop_ratio=stop_ratio,
        bits=bits,
        native_source=native_source,
        stretch_lo=stretch_lo,
        stretch_hi=stretch_hi,
    )


def compute_white_ratio_uint8(frame_uint8, threshold):
    """8bitフレームに対して P(X>=threshold) を計算する。"""
    if frame_uint8 is None:
        return 0.0
    if len(frame_uint8.shape) == 3 and frame_uint8.shape[2] >= 3:
        metric = np.max(frame_uint8[:, :, :3], axis=2)
    else:
        metric = frame_uint8
    white_pixels = np.sum(metric >= int(threshold))
    total_pixels = metric.shape[0] * metric.shape[1]
    if total_pixels <= 0:
        return 0.0
    return float(white_pixels) / float(total_pixels)


ROI_MIN_SIZE = 16        # ROIとして受け付ける最小の辺(raw px)
ROI_ALIGN_MIN_SIZE = 256  # 位置合わせ窓の最小の辺(ROIがこれより小さいときはROIの中心で広げる)


def normalize_roi(x0, y0, x1, y1, width, height):
    """ROIをフレーム内に収め、Bayer配列を壊さないよう偶数座標に揃える。小さすぎるときはNone。"""
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0, y0 = max(0, x0) // 2 * 2, max(0, y0) // 2 * 2
    x1, y1 = min(width, x1) // 2 * 2, min(height, y1) // 2 * 2
    if x1 - x0 < ROI_MIN_SIZE or y1 - y0 < ROI_MIN_SIZE:
        return None
    return (x0, y0, x1, y1)


def compute_white_ratio_native(frame_native, threshold_native):
    """ネイティブbit深度配列に対して P(X>=threshold) を計算する。"""
    if frame_native is None:
        return 0.0
    arr = np.asarray(frame_native)
    if arr.size == 0:
        return 0.0
    metric = arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        metric = np.max(arr[:, :, :3], axis=2)
    white_pixels = np.sum(metric >= float(threshold_native))
    total_pixels = metric.shape[0] * metric.shape[1]
    if total_pixels <= 0:
        return 0.0
    return float(white_pixels) / float(total_pixels)


def _now_local_utc_iso():
    """(ローカル時刻 ISO8601 オフセット付き, UTC 時刻 ISO8601 'Z' 付き) を返す。"""
    loc = datetime.datetime.now().astimezone()
    utc = loc.astimezone(datetime.timezone.utc)
    return loc.isoformat(timespec="seconds"), utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_exif_bytes(frame_count, effective_bits, bayer_name, wb_gains, gamma_value,
                     stack_count, stack_enabled, source, flip_h, flip_v, wire_format, w, h,
                     shutter_us=None, gain=None,
                     telescope=None, obs_filter=None, mount=None,
                     object_name=None, observer=None, instrument=None):
    """JPEG用EXIFバイト列を生成する。"""
    now_str = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    _loc_iso, _utc_iso = _now_local_utc_iso()
    fields = [
        f"datetime_local={_loc_iso}",
        f"datetime_utc={_utc_iso}",
        f"bits={effective_bits}",
        f"bayer={bayer_name}",
        f"wb_b={wb_gains[0]:.3f}",
        f"wb_g={wb_gains[1]:.3f}",
        f"wb_r={wb_gains[2]:.3f}",
        f"gamma={gamma_value:.3f}",
        f"stack_count={stack_count}",
        f"stack={'ON' if stack_enabled else 'OFF'}",
        f"frame={frame_count}",
        f"flip_h={flip_h}",
        f"flip_v={flip_v}",
        f"wire={wire_format or 'N/A'}",
    ]
    if shutter_us is not None:
        fields.append(f"shutter={shutter_us}us")
    if gain is not None:
        fields.append(f"gain={gain}")
    if telescope:
        fields.append(f"telescope={telescope}")
    if obs_filter:
        fields.append(f"filter={obs_filter}")
    if mount:
        fields.append(f"mount={mount}")
    if object_name:
        fields.append(f"object={object_name}")
    if observer:
        fields.append(f"observer={observer}")
    if instrument:
        fields.append(f"instrument={instrument}")
    meta = "\n" + "\n".join(fields)
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: b"rpicam-raw",
            piexif.ImageIFD.Software: b"raw_live_stack.py",
            piexif.ImageIFD.DateTime: now_str.encode("ascii"),
            piexif.ImageIFD.ImageDescription: source.encode("ascii", errors="replace"),
            piexif.ImageIFD.ImageWidth: w,
            piexif.ImageIFD.ImageLength: h,
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: now_str.encode("ascii"),
            piexif.ExifIFD.PixelXDimension: w,
            piexif.ExifIFD.PixelYDimension: h,
            piexif.ExifIFD.UserComment: b"ASCII\x00\x00\x00" + meta.encode("ascii", errors="replace"),
        },
    }
    if instrument and instrument.isascii():
        # ModelはEXIF標準のカメラ機種名タグ。ビューアのカメラ情報欄に表示される
        exif_dict["0th"][piexif.ImageIFD.Model] = instrument.encode("ascii")
    if shutter_us is not None:
        exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (int(shutter_us), 1_000_000)
    return piexif.dump(exif_dict)


# ---------------------------------------------------------------------------
# LiveStack クラス
# ---------------------------------------------------------------------------

class LiveStack:
    """LiveStack処理クラス"""

    def __init__(self, max_frames=100, verbose=False, bits=8):
        self.max_frames = max_frames  # max_framesをリングバッファサイズとして明確化
        self.verbose = verbose  # デバッグ出力制御
        self.bits = bits  # 元データのビット深度
        self.overflow_ratio_threshold = 0.10  # 打ち切り比率（例: 0.10 = 10%）
        self.brightness_threshold = (1 << bits) - 1  # 打ち切り輝度しきい値（ネイティブビット深度で表現）
        self.include_overflow_frame = False  # True: しきい値超過フレームも1枚含めて停止
        self.stack_method = "sum"  # sum / lighten / darken / median / sigma / sigmasum（sum以外は最新Max Frames枚を使う）
        self.clip_sigma = 2.5  # sigma方式の棄却しきい値（標準偏差の倍数）
        self.reset()
        self.buffer = [None] * max_frames  # Noneで初期化
        self.buffer_frame_ids = [None] * max_frames  # 各バッファ要素に対応するフレーム番号
        self.buffer_pos = [None] * max_frames  # 各フレームの基準座標系での位置(x,y) raw px・float
        self._ref_img = None  # ズレ検出の基準(キーフレーム)画像
        self._ref_pos = (0.0, 0.0)  # 基準フレームの位置
        self._last_pos = (0.0, 0.0)  # 直近フレームの位置
        self._last_vel = (0.0, 0.0)  # 直近のフレーム間移動量(検出失敗時の外挿用)
        self._reject_count = 0  # 異常値として捨てた連続回数
        self.align_max_jump = 10.0  # 予測位置からこれ以上外れた検出は異常値(raw px)
        self._hann = {}
        self.align_ratio = 0.5  # ずれ検出窓の大きさ（画像幅/高さに対する比率）
        self.roi = None  # (x0,y0,x1,y1) raw px・偶数座標。停止判定と位置合わせの対象領域（None=全体/中央窓）
        self.align_enabled = True  # False: ずれ検出も移動もせず単純加算
        self.align_min_response = 0.05  # phaseCorrelateの応答がこれ未満なら検出失敗
        self.last_align_ms = 0.0  # 直近のずれ検出にかかった時間
        self.buffer_index = 0
        self.dark_frame = None  # d キーで取得するまでは None（サイズを動的に合わせるため）。常に取得時の向きのまま保持する
        self.dark_flip = "none"  # none/h/v/hv: ダークを使うときだけ反転して、反転して録画したSER等の向きに合わせる
        self.dark_dirty = False  # このセッションでダークを取得したか（終了時の自動保存の要否）
        self._dark_eff = None  # 反転済みダークのキャッシュ
        self._dark_eff_src = None  # キャッシュの元になったダーク（同一性で更新を判定する）
        self._dark_eff_flip = None
        self.dark_enabled = True  # False: ダークがあっても減算・ホットピクセル補間をしない
        self.dark_buffer = []  # ダークフレーム用リングバッファ
        self.hot_sigma = 10.0  # ホットピクセル判定のしきい値(σ倍)。0以下で無効
        self._hot_fix = None  # (idx, nb_idx(4,n), nb_valid(4,n)) ホットピクセル補間用の索引
        self._hot_mask = None  # ホットピクセルのboolマスク(ダークと同形状)

    def reset(self):
        """スタック状態をリセット"""
        self.stacked_image = None       # 表示用スタック画像
        self.reference_stack = None     # 位置合わせ用基準スタック（グレースケール）
        self.stack_count = 0
        self.reference_frame = None
        self.reference_template = None  # テンプレートマッチング用
        self.template_rect = None      # テンプレート範囲
        self.fixed_stack_count = None  # 固定スタック数
        self.failed_count = 0          # 連続失敗カウント
        self.stacked_raw16 = None      # raw16空間のfloat32累積スタック（FITS保存用）
        self.last_stop_reason = "reset"
        self.last_overflow_ratio = 0.0
        self.last_processed_frame_id = None

    def add_frame(self, frame):
        """フレームをスタックに追加"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.reference_frame is None:
            # 最初のフレームを基準として設定
            self.reference_frame = gray.copy()
            self.stacked_image = frame.astype(np.float32)
            self.reference_stack = gray.astype(np.float32)
            self.stack_count = 1

            # テンプレート領域を設定（中央80%）
            h, w = gray.shape
            margin_h = int(h * 0.1)  # 上下10%ずつマージン
            margin_w = int(w * 0.1)  # 左右10%ずつマージン
            self.template_rect = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)

            return frame, True

        try:
            # 新仕様: 最新フレーム（gray）をテンプレートとして使用
            # 暗い環境での処理を改善するため前処理を追加
            # コントラスト強化とノイズ軽減
            enhanced_gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=30)
            enhanced_ref_stack = cv2.convertScaleAbs(self.reference_stack / max(1, self.stack_count), alpha=2.0, beta=30)

            # ガウシアンブラーでノイズを軽減
            enhanced_gray = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
            enhanced_ref_stack = cv2.GaussianBlur(enhanced_ref_stack, (3, 3), 0)

            # テンプレートマッチング - 新仕様
            x, y, tw, th = self.template_rect

            # 新しいフレームからテンプレートを抽出（最新フレーム基準）
            current_template = enhanced_gray[y:y+th, x:x+tw]

            # スタック画像内で探索範囲を設定
            search_margin = 50  # 探索範囲のマージン
            search_x = max(0, x - search_margin)
            search_y = max(0, y - search_margin)
            search_w = min(enhanced_ref_stack.shape[1] - search_x, tw + 2*search_margin)
            search_h = min(enhanced_ref_stack.shape[0] - search_y, th + 2*search_margin)

            # スタック画像から探索領域を抽出
            search_area = enhanced_ref_stack[search_y:search_y+search_h, search_x:search_x+search_w]

            # テンプレートマッチング実行（新しいフレーム vs スタック画像）
            result = cv2.matchTemplate(search_area, current_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 適応的な閾値設定（高速シャッター対応）
            # 失敗が続いている場合は閾値を下げる
            base_threshold = 0.3
            adaptive_threshold = base_threshold - (self.failed_count * 0.05)  # 失敗毎に0.05下げる
            adaptive_threshold = max(0.15, adaptive_threshold)  # 最低0.15まで

            if max_val > adaptive_threshold:
                # マッチした位置を計算（スタック画像内での位置）
                match_x = search_x + max_loc[0]
                match_y = search_y + max_loc[1]

                # 新仕様: 移動量を計算（現在フレーム位置 - スタック内位置）
                offset_x = x - match_x  # 符号を逆転
                offset_y = y - match_y  # 符号を逆転

                # デバッグ情報を出力
                print(f"マッチング成功: 相関値={max_val:.3f}, オフセット=({offset_x:.2f}, {offset_y:.2f}), スタック数={self.stack_count}")

                # 大きな移動量は無視（ノイズマッチングを防ぐ）
                if abs(offset_x) > 100 or abs(offset_y) > 100:
                    self.failed_count += 1
                    print(f"位置合わせ失敗 (大きな移動): offset=({offset_x:.2f}, {offset_y:.2f}) 失敗回数:{self.failed_count}/3")
                    return frame, False

                # マッチング成功 - 失敗カウントリセット
                self.failed_count = 0

                # 新仕様: 新しいフレームを基準位置に配置し、スタック画像を調整
                # アフィン変換行列を作成（新しいフレームに合わせてスタック画像を移動）
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

                # スタック画像を位置合わせ（新しいフレームの位置に合わせる）
                h, w = frame.shape[:2]
                aligned_stacked_image = cv2.warpAffine(self.stacked_image, M, (w, h))
                aligned_reference_stack = cv2.warpAffine(self.reference_stack, M, (w, h))

                # 新しいフレームを追加してスタック更新
                self.stacked_image = aligned_stacked_image + frame.astype(np.float32)
                self.reference_stack = aligned_reference_stack + gray.astype(np.float32)

                self.stack_count += 1

                # オーバーフロー検出（設定した比率・輝度しきい値で判定）
                if self.fixed_stack_count is None:
                    # プレビューと同じ √N 正規化後の8bit画像で判定する
                    threshold_8 = int(self.brightness_threshold >> max(0, self.bits - 8))
                    preview_scale = float(np.sqrt(max(1, self.stack_count)))
                    clipped = np.clip(self.stacked_image / preview_scale, 0, 255).astype(np.uint8)
                    metric = np.max(clipped[:, :, :3], axis=2)
                    white_pixels = np.sum(metric >= threshold_8)
                    total_pixels = clipped.shape[0] * clipped.shape[1]
                    overflow_ratio = white_pixels / total_pixels

                    if overflow_ratio >= self.overflow_ratio_threshold:
                        self.fixed_stack_count = self.stack_count
                        print(f"*** オーバーフロー検出: ratio={overflow_ratio:.3f} >= {self.overflow_ratio_threshold:.3f}, threshold={self.brightness_threshold} スタック数固定: {self.fixed_stack_count} ***")

                # 最大スタック数を制限（固定値がある場合はそれを使用）
                max_count = self.fixed_stack_count if self.fixed_stack_count else self.max_frames
                if self.stack_count > max_count:
                    self.stack_count = max_count

                # 固定スタック数に達した場合の処理
                if self.fixed_stack_count and self.stack_count >= self.fixed_stack_count:
                    # 新フレームで既存フレームを置き換え（移動処理）
                    self.stacked_image += aligned_stacked_image + frame.astype(np.float32)
                    self.reference_stack += aligned_reference_stack + gray.astype(np.float32)
                    result_image = np.clip(self.stacked_image, 0, 255).astype(np.uint8)
                    print("移動加算処理")
                else:
                    # 単純加算の場合はクリッピングのみ（正規化なし）
                    result_image = np.clip(self.stacked_image, 0, 255).astype(np.uint8)
                    print(f"加算処理 (フレーム数: {self.stack_count})")

                return result_image, True
            else:
                # マッチング失敗
                self.failed_count += 1
                print(f"テンプレートマッチング失敗 (相関値:{max_val:.3f} < 閾値:{adaptive_threshold:.3f}) 失敗回数:{self.failed_count}/3")

                # 連続失敗が多い場合はリセット
                if self.failed_count >= 3:
                    print("*** 3回連続失敗によりスタックリセット ***")
                    self.reset()
                    return frame, False

            return frame, False

        except Exception as e:
            print(f"スタッキングエラー: {e}")
            return frame, False

    def add_to_buffer(self, frame, frame_id=None):
        """リングバッファにフレームを追加"""
        t0 = time.perf_counter()
        if not self.align_enabled:
            self.buffer[self.buffer_index] = frame
            self.buffer_frame_ids[self.buffer_index] = frame_id
            self.buffer_pos[self.buffer_index] = (0.0, 0.0)
            self.buffer_index = (self.buffer_index + 1) % self.max_frames
            self.last_align_ms = 0.0
            return
        cur = self._prepare_align_image(frame)
        pos = self._last_pos
        if self._ref_img is not None and self._ref_img.shape == cur.shape:
            d = self._detect_shift(self._ref_img, cur)
            pred = (self._last_pos[0] + self._last_vel[0], self._last_pos[1] + self._last_vel[1])
            if d is not None:
                cand = (self._ref_pos[0] + d[0], self._ref_pos[1] + d[1])
                # 予測から大きく外れた検出は誤検出として捨てる（連続したら再ロックとして受け入れる）
                if max(abs(cand[0] - pred[0]), abs(cand[1] - pred[1])) > self.align_max_jump and self._reject_count < 5:
                    d = None
                    self._reject_count += 1
            if d is None:
                pos = pred
                print(f"[align] ずれ検出失敗/異常値 (frame {frame_id}): 前回の移動量で外挿します")
            else:
                relock = self._reject_count >= 5
                self._reject_count = 0
                pos = cand
                self._last_vel = (0.0, 0.0) if relock else (pos[0] - self._last_pos[0], pos[1] - self._last_pos[1])
                # 基準から離れすぎたら基準を付け替える（重なりを確保しつつ誤差の積み重ねをキーフレーム数に抑える）
                if max(abs(d[0]), abs(d[1])) > 0.1 * min(cur.shape) * 2:
                    self._ref_img = cur
                    self._ref_pos = pos
        else:
            pos = (self._last_pos[0] + self._last_vel[0], self._last_pos[1] + self._last_vel[1])
            self._ref_img = cur
            self._ref_pos = pos
        self._last_pos = pos
        self.last_align_ms = (time.perf_counter() - t0) * 1000.0

        self.buffer[self.buffer_index] = frame
        self.buffer_frame_ids[self.buffer_index] = frame_id
        self.buffer_pos[self.buffer_index] = pos
        self.buffer_index = (self.buffer_index + 1) % self.max_frames

    def clear_buffer(self):
        """リングバッファとずれ検出状態を破棄する。"""
        self.buffer = [None] * self.max_frames
        self.buffer_frame_ids = [None] * self.max_frames
        self.buffer_pos = [None] * self.max_frames
        self.buffer_index = 0
        self._ref_img = None
        self._ref_pos = (0.0, 0.0)
        self._last_pos = (0.0, 0.0)
        self._last_vel = (0.0, 0.0)
        self._reject_count = 0
    def _prepare_align_image(self, raw16):
        """中央の窓をダーク減算→2x2ビニング→背景除去して、ずれ検出用画像を作る。"""
        h, w = raw16.shape[:2]
        win = self.align_window(w, h)
        if win is not None:
            x0, y0 = win[0], win[1]
            sw, sh = win[2] - win[0], win[3] - win[1]
        else:
            sh = max(64, int(h * self.align_ratio)) // 2 * 2
            sw = max(64, int(w * self.align_ratio)) // 2 * 2
            sh, sw = min(sh, h // 2 * 2), min(sw, w // 2 * 2)
            y0 = (h - sh) // 2 // 2 * 2
            x0 = (w - sw) // 2 // 2 * 2
        crop = raw16[y0:y0 + sh, x0:x0 + sw].astype(np.float32)
        dark = self.dark_eff
        if dark is not None and self.dark_enabled:
            crop -= dark[y0:y0 + sh, x0:x0 + sw]
        if self.dark_enabled and self._hot_mask is not None and self._hot_mask.shape == raw16.shape[:2]:
            crop[self._hot_mask[y0:y0 + sh, x0:x0 + sw]] = 0
        binned = crop.reshape(sh // 2, 2, sw // 2, 2).sum(axis=(1, 3))
        binned = cv2.GaussianBlur(binned, (0, 0), 1.0)
        return binned - cv2.GaussianBlur(binned, (0, 0), 16)

    def roi_text(self):
        x0, y0, x1, y1 = self.roi
        return f"x={x0} y={y0} {x1 - x0}x{y1 - y0}"

    def align_window(self, w, h):
        """ROIがあるときの位置合わせ窓(x0,y0,x1,y1)。ROIの中心を保ち、辺が小さければ ROI_ALIGN_MIN_SIZE まで広げる。
        フレームからはみ出す分はずらして収める。ROIなし/不整合ならNone（呼び出し側で中央窓を使う）。"""
        roi = self.roi
        if roi is None or roi[2] > w or roi[3] > h:
            return None
        out = []
        for lo, hi, size in ((roi[0], roi[2], w), (roi[1], roi[3], h)):
            length = min(max(hi - lo, ROI_ALIGN_MIN_SIZE), size // 2 * 2)
            start = (lo + hi - length) // 2
            start = max(0, min(start, size - length)) // 2 * 2
            out += [start, start + length]
        return (out[0], out[2], out[1], out[3])

    def roi_view(self, arr):
        """停止判定用: ROIがあればその範囲(ビュー)、なければ全体を返す。"""
        if self.roi is None:
            return arr
        x0, y0, x1, y1 = self.roi
        return arr[y0:y1, x0:x1]

    def _detect_shift(self, prev, cur):
        """prev→curの移動量(dx,dy)をraw pxのfloat(サブピクセル)で返す。失敗時はNone。"""
        hh, ww = cur.shape
        win = self._hann.get((ww, hh))
        if win is None:
            win = self._hann[(ww, hh)] = cv2.createHanningWindow((ww, hh), cv2.CV_32F)
        # phaseCorrelateは入力に窓関数を書き込むため、基準画像を保つにはコピーを渡す
        (dx, dy), response = cv2.phaseCorrelate(prev.copy(), cur.copy(), win)
        if response < self.align_min_response:
            return None
        # ビニング後pxを2倍 = raw px。偶数への丸めは累積後に行う
        return dx * 2.0, dy * 2.0

    def get_latest_frame_id(self):
        """現在バッファにある最新フレームIDを返す。"""
        idx = (self.buffer_index - 1 + self.max_frames) % self.max_frames
        return self.buffer_frame_ids[idx]

    def resize_max_frames(self, new_max_frames):
        """リングバッファ/ダークバッファを保ったまま最大フレーム数を変更する。"""
        new_max_frames = int(max(1, new_max_frames))
        if new_max_frames == self.max_frames:
            return

        old_buffer = self.buffer
        old_ids = self.buffer_frame_ids
        old_deltas = self.buffer_pos
        old_len = len(old_buffer)

        # 旧リングバッファを時系列順（古い→新しい）に展開
        ordered = []
        ordered_ids = []
        ordered_deltas = []
        for i in range(old_len):
            idx = (self.buffer_index + i) % old_len
            f = old_buffer[idx]
            if f is not None:
                ordered.append(f)
                ordered_ids.append(old_ids[idx])
                ordered_deltas.append(old_deltas[idx])

        # 最新フレームを優先して保持
        ordered = ordered[-new_max_frames:]
        ordered_ids = ordered_ids[-new_max_frames:]
        ordered_deltas = ordered_deltas[-new_max_frames:]

        self.max_frames = new_max_frames
        self.buffer = [None] * new_max_frames
        self.buffer_frame_ids = [None] * new_max_frames
        self.buffer_pos = [None] * new_max_frames
        for i, f in enumerate(ordered):
            self.buffer[i] = f
            self.buffer_frame_ids[i] = ordered_ids[i]
            self.buffer_pos[i] = ordered_deltas[i]
        self.buffer_index = len(ordered) % new_max_frames

        if len(self.dark_buffer) > new_max_frames:
            self.dark_buffer = self.dark_buffer[-new_max_frames:]

    def _warp_bayer(self, img, dx, dy):
        """Bayer画像をraw pxで(dx,dy)移動する。4色面ごとにdx/2,dy/2のサブピクセル移動を行う。"""
        h, w = img.shape
        out = np.empty_like(img)
        M = np.float32([[1, 0, dx / 2.0], [0, 1, dy / 2.0]])
        for oy in (0, 1):
            for ox in (0, 1):
                plane = np.ascontiguousarray(img[oy::2, ox::2])
                out[oy::2, ox::2] = cv2.warpAffine(plane, M, (plane.shape[1], plane.shape[0]), flags=cv2.INTER_LINEAR)
        return out

    def process_stack(self, bayer_code, stretch=False, should_continue=None):
        """スタック処理を実行（バッファはraw16 uint16 Bayerで保持）。

        stretch: sum以外の方式で、結果を自動ストレッチして8bit化する（単一フレーム相当の明るさになるため）。
        should_continue: 呼ぶとFalseを返すと処理を静かに中断するコールバック（Stack OFF切り替えをループ中に検知するため）。
        """
        latest_index = (self.buffer_index - 1 + self.max_frames) % self.max_frames
        if self.buffer[latest_index] is None:
            print("リングバッファが空です。スタック処理をスキップします。")
            self.last_stop_reason = "buffer-empty"
            self.last_overflow_ratio = 0.0
            return None
        latest_frame_id = self.buffer_frame_ids[latest_index]

        # 最新フレーム（raw16 Bayer float32）
        latest_f32 = self.buffer[latest_index].astype(np.float32)
        dark = self.dark_eff
        if dark is not None and self.dark_enabled:
            latest_f32 = np.clip(latest_f32 - dark, 0, None)
            self._fix_hot(latest_f32)

        if self.stack_method != "sum":
            return self._process_window(bayer_code, stretch, latest_index, latest_f32, latest_frame_id)

        stacked = latest_f32.copy()  # float32 raw16 Bayerの累積スタック

        h, w = latest_f32.shape[:2]

        # 加算スタック表示用の固定スケール: 12bitなら /16 で8bitに収める
        # N枚加算するとN倍明るくなり、暗い星が徐々に浮かび上がる
        scale = float(1 << max(0, self.bits - 8))
        max_native = float((1 << self.bits) - 1)  # CCDFと同じクリップ上限

        valid_stack_count = 1  # 最新フレームを含む
        stop_reason = None
        last_overflow_ratio = 0.0
        cum_x = cum_y = 0.0  # 最新フレームに対する過去フレームのずれ(float)
        latest_pos = self.buffer_pos[latest_index] or (0.0, 0.0)
        for i in range(self.max_frames - 1):
            if should_continue is not None and not should_continue():
                # Stack OFFへ切り替わった処理途中のスタックはログを出さず静かに破棄する
                stop_reason = "cancelled"
                break
            past_index = (self.buffer_index - 2 - i + self.max_frames) % self.max_frames
            past_raw16 = self.buffer[past_index]
            if past_raw16 is None:
                stop_reason = "buffer-empty"
                break

            past_f32 = past_raw16.astype(np.float32)
            if dark is not None and self.dark_enabled:
                past_f32 = np.clip(past_f32 - dark, 0, None)
                self._fix_hot(past_f32)

            past_pos = self.buffer_pos[past_index] or latest_pos
            cum_x = latest_pos[0] - past_pos[0]
            cum_y = latest_pos[1] - past_pos[1]
            # Bayer色ごとの面を別々にサブピクセルでずらす（色配列を壊さない）
            aligned = self._warp_bayer(past_f32, cum_x, cum_y) if self.align_enabled else past_f32

            test_stack = stacked + aligned

            # オーバーフロー判定: CCDFと同じ計算（累積をネイティブ範囲にクリップして比較）
            overflow_ratio = float(np.mean(
                np.clip(self.roi_view(test_stack), 0.0, max_native) >= float(self.brightness_threshold)
            ))
            last_overflow_ratio = overflow_ratio
            if overflow_ratio >= self.overflow_ratio_threshold:
                if self.include_overflow_frame:
                    stacked = test_stack
                    valid_stack_count += 1
                    print(
                        f"オーバーフロー条件に到達 (ratio={overflow_ratio:.3f} >= {self.overflow_ratio_threshold:.3f})。"
                        "超過フレームを含めて終了します。"
                    )
                    stop_reason = "ratio-stop-included"
                else:
                    print(f"オーバーフロー条件に達しました (ratio={overflow_ratio:.3f} >= {self.overflow_ratio_threshold:.3f})。スタック処理を終了します。")
                    stop_reason = "ratio-stop"
                break

            stacked = test_stack
            valid_stack_count += 1

        if stop_reason is None:
            stop_reason = "max-frames"
        if stop_reason == "cancelled":
            # Stack OFFへ切り替わった処理途中の結果は使わない（stack_count等も更新しない）
            return None
        self.stack_count = valid_stack_count
        self.stacked_raw16 = stacked  # float32 Bayer累積スタック（FITS/NPY保存用）
        self.last_stop_reason = stop_reason
        self.last_overflow_ratio = last_overflow_ratio
        self.last_processed_frame_id = latest_frame_id

        # 表示用: 加算値を固定スケール(bits→8bit)でそのまま変換
        # 1枚: 100ADU → 100/16 = 6  /  10枚加算: 1000ADU → 1000/16 = 62  → ヒストグラムが右シフト
        bayer8 = np.clip(stacked / scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(bayer8, bayer_code)

    def _iter_aligned(self, latest_index):
        """最新フレームを基準に位置合わせした過去フレームを、新しい順に返す（最新フレーム自身は含まない）。"""
        latest_pos = self.buffer_pos[latest_index] or (0.0, 0.0)
        dark = self.dark_eff
        for i in range(self.max_frames - 1):
            past_index = (self.buffer_index - 2 - i + self.max_frames) % self.max_frames
            past_raw16 = self.buffer[past_index]
            if past_raw16 is None:
                return
            past_f32 = past_raw16.astype(np.float32)
            if dark is not None and self.dark_enabled:
                past_f32 = np.clip(past_f32 - dark, 0, None)
                self._fix_hot(past_f32)
            past_pos = self.buffer_pos[past_index] or latest_pos
            dx = latest_pos[0] - past_pos[0]
            dy = latest_pos[1] - past_pos[1]
            yield self._warp_bayer(past_f32, dx, dy) if self.align_enabled else past_f32

    def _reduce_frames(self, frames, method):
        """frames: (n, h, w) float32。画素ごとに中央値またはシグマクリップ平均を求める（行ブロックごとに処理）。

        sigmasum は、棄却後の平均にフレーム数nを掛けて加算と同じスケールにする（棄却した分も補正される）。
        """
        n, h, w = frames.shape
        out = np.empty((h, w), np.float32)
        rows = max(1, 8_000_000 // max(1, w * n))
        for r0 in range(0, h, rows):
            r1 = min(h, r0 + rows)
            block = frames[:, r0:r1]
            if method == "median":
                # 画素ごとに連続配置してから中央値を取る方が速い
                flat = np.ascontiguousarray(block.reshape(n, -1).T)
                out[r0:r1] = np.median(flat, axis=1).reshape(r1 - r0, w)
            else:
                mean = block.mean(axis=0)
                if n >= 3:
                    std = block.std(axis=0)
                    for _ in range(2):
                        keep = np.abs(block - mean) <= self.clip_sigma * std
                        cnt = keep.sum(axis=0)
                        mean = np.where(cnt > 0, (block * keep).sum(axis=0) / np.maximum(cnt, 1), mean)
                        std = np.sqrt((((block - mean) ** 2) * keep).sum(axis=0) / np.maximum(cnt, 1))
                out[r0:r1] = mean * n if method == "sigmasum" else mean
        return out

    def _process_window(self, bayer_code, stretch, latest_index, latest_f32, latest_frame_id):
        """sum以外の方式: 最新Max Frames枚（位置合わせ後）から比較明/比較暗/中央値/シグマクリップ平均/シグマクリップ加算を求める。"""
        method = self.stack_method
        count = 1
        if method in ("lighten", "darken"):
            op = np.maximum if method == "lighten" else np.minimum
            stacked = latest_f32.copy()
            for aligned in self._iter_aligned(latest_index):
                op(stacked, aligned, out=stacked)
                count += 1
        else:
            h, w = latest_f32.shape[:2]
            available = sum(1 for b in self.buffer if b is not None)
            n = min(available, self.max_frames)
            mem = get_available_memory_bytes()
            if mem is not None:
                n_mem = max(1, int(mem * 0.4 // (h * w * 4)))
                if n_mem < n:
                    print(f"[warn] {method}: メモリ不足のため使用枚数を {n} から {n_mem} に制限します")
                    n = n_mem
            frames = np.empty((n, h, w), np.float32)
            frames[0] = latest_f32
            stop_reason = f"window-{method}"
            last_ratio = 0.0
            # sigmasum は加算(sum)と同じ停止条件で枚数を決める（単純加算が飽和判定に達する手前まで）
            acc = latest_f32.copy() if method == "sigmasum" else None
            max_native = float((1 << self.bits) - 1)
            for aligned in self._iter_aligned(latest_index):
                if count >= n:
                    break
                if acc is not None:
                    test = acc + aligned
                    last_ratio = float(np.mean(np.clip(self.roi_view(test), 0.0, max_native) >= float(self.brightness_threshold)))
                    if last_ratio >= self.overflow_ratio_threshold:
                        if self.include_overflow_frame:
                            frames[count] = aligned
                            count += 1
                            stop_reason = "ratio-stop-included"
                        else:
                            stop_reason = "ratio-stop"
                        break
                    acc = test
                frames[count] = aligned
                count += 1
            stacked = self._reduce_frames(frames[:count], method)
            self.last_overflow_ratio = last_ratio
            self.last_stop_reason = stop_reason

        self.stack_count = count
        self.stacked_raw16 = stacked
        if method in ("lighten", "darken"):
            self.last_stop_reason = f"window-{method}"
            self.last_overflow_ratio = 0.0
        self.last_processed_frame_id = latest_frame_id

        if stretch and method != "sigmasum":  # sigmasumは加算と同じスケールなので固定スケールで表示
            return raw16_to_bgr8_stretched(np.clip(stacked, 0, 65535).astype(np.uint16), bayer_code)[0]
        scale = float(1 << max(0, self.bits - 8))
        bayer8 = np.clip(stacked / scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(bayer8, bayer_code)

    @staticmethod
    def flip_array(arr, mode):
        """2次元配列をmode(none/h/v/hv)で反転する（Bayerの色配列も同じ向きに反転する）。"""
        if mode == "h":
            return np.ascontiguousarray(arr[:, ::-1])
        if mode == "v":
            return np.ascontiguousarray(arr[::-1, :])
        if mode == "hv":
            return np.ascontiguousarray(arr[::-1, ::-1])
        return arr

    @property
    def dark_eff(self):
        """引き算・ホットピクセル検出に使うダーク（dark_flip を適用済み）。"""
        d = self.dark_frame
        if d is None or self.dark_flip == "none":
            return d
        if self._dark_eff_src is not d or self._dark_eff_flip != self.dark_flip:
            self._dark_eff = self.flip_array(d, self.dark_flip)
            self._dark_eff_src = d
            self._dark_eff_flip = self.dark_flip
        return self._dark_eff

    def set_dark_flip(self, mode):
        """ダークの反転を切り替え、ホットピクセルの索引を作り直す。"""
        if mode != self.dark_flip:
            self.dark_flip = mode
            self.build_hot_mask()

    @staticmethod
    def _hot_candidates(a, k=8.0):
        """同色面ごとの3x3メディアンとの差がロバストσのk倍を超える画素（ホットピクセル候補）のboolマスク。"""
        out = np.zeros(a.shape, bool)
        for oy in (0, 1):
            for ox in (0, 1):
                plane = np.ascontiguousarray(a[oy::2, ox::2], dtype=np.float32)
                resid = plane - cv2.medianBlur(plane, 3)
                med = float(np.median(resid))
                sigma = max(1.4826 * float(np.median(np.abs(resid - med))), 0.5)
                out[oy::2, ox::2] = resid > med + k * sigma
        return out

    def detect_dark_flip(self, raw16):
        """生フレームとダークのホットピクセルの一致数から、ダークの反転(none/h/v/hv)を判定する。判定できなければNone。

        ヘッダーのBAYER表記は終了時の自動保存などで書き換わることがあり当てにならないため、画素の一致で決める。
        """
        d = self.dark_frame
        if d is None or raw16.shape != d.shape or d.shape[0] % 2 or d.shape[1] % 2:
            return None
        frame_hot = self._hot_candidates(raw16.astype(np.float32))
        scores = {}
        for mode in ("none", "h", "v", "hv"):
            dark_hot = self._hot_candidates(self.flip_array(d, mode))
            scores[mode] = int(np.count_nonzero(frame_hot & dark_hot))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, second = ranked[0], ranked[1]
        print(f"[dark] 向きの判定 (ホットピクセル一致数): {scores}")
        # 偶然の一致より十分多いことを要求する（ダークのホットピクセルが少ないと判定できない）
        if best[1] < 20 or best[1] < 3 * max(second[1], 1):
            return None
        return best[0]

    def build_hot_mask(self):
        """ダークから、同色近傍の中央値より突出した画素を求め、補間用の索引を作る。"""
        self._hot_fix = None
        self._hot_mask = None
        d = self.dark_eff
        if d is None or self.hot_sigma <= 0:
            return
        h, w = d.shape
        if h % 2 or w % 2:
            return
        resid = np.empty_like(d)
        for oy in (0, 1):
            for ox in (0, 1):
                p = np.ascontiguousarray(d[oy::2, ox::2])
                resid[oy::2, ox::2] = p - cv2.medianBlur(p, 3)
        med = float(np.median(resid))
        sigma = max(1.4826 * float(np.median(np.abs(resid - med))), 0.5)
        mask = resid > med + self.hot_sigma * sigma
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            print("[dark] ホットピクセルなし")
            return
        nb = []
        valid = []
        for dy, dx in ((0, -2), (0, 2), (-2, 0), (2, 0)):
            ny = np.clip(ys + dy, 0, h - 1)
            nx = np.clip(xs + dx, 0, w - 1)
            nb.append(ny * w + nx)
            valid.append((~mask[ny, nx]).astype(np.float32))
        self._hot_fix = (ys * w + xs, np.array(nb), np.array(valid))
        self._hot_mask = mask
        print(f"[dark] ホットピクセル {len(ys)}個 ({len(ys) / mask.size * 100:.3f}%) を補間対象にしました (sigma={sigma:.2f}, {self.hot_sigma:g}σ)")

    def _fix_hot(self, f32):
        """ホットピクセルを同色の上下左右(2px離れ)の平均で置き換える。f32は連続配列。"""
        if self._hot_fix is None:
            return
        idx, nb, valid = self._hot_fix
        flat = f32.reshape(-1)
        flat[idx] = (flat[nb] * valid).sum(axis=0) / np.maximum(valid.sum(axis=0), 1.0)

    def set_dark_frame(self, raw16_frame):
        """ダークフレームを加算平均して設定（raw16 uint16 Bayer入力）"""
        if self.dark_flip != "none":
            # 新しいフレームはストリームと同じ向きなので、反転して使っていた既存ダークは反転後の向きに焼き込んでから平均する
            baked = {}
            for b in self.dark_buffer:
                if id(b) not in baked:  # 読み込み済みダークは同じ配列を複数回積んでいるため、1回だけ反転する
                    baked[id(b)] = self.flip_array(b, self.dark_flip)
            self.dark_buffer = [baked[id(b)] for b in self.dark_buffer]
            self.dark_flip = "none"
            print("[dark] 既存ダークに反転を焼き込んで、新しいフレームと平均します (Dark Flip: none)")
        if len(self.dark_buffer) >= self.max_frames:
            self.dark_buffer.pop(0)  # 古いフレームを削除
        self.dark_buffer.append(raw16_frame.astype(np.float32))

        # 加算平均を計算（raw16空間のまま保持）
        dark_sum = np.sum(self.dark_buffer, axis=0)
        self.dark_frame = dark_sum / len(self.dark_buffer)  # float32 raw16 Bayer
        self.dark_dirty = True
        self.build_hot_mask()
        print(f"ダークフレームを更新しました。現在の平均化フレーム数: {len(self.dark_buffer)}")


# ---------------------------------------------------------------------------
# 保存ユーティリティ
# ---------------------------------------------------------------------------

def flip_bayer_key(bayer_key, flip_h, flip_v, width, height):
    """反転後のBayerパターン名を返す（偶数サイズのときのみパターンが入れ替わる）"""
    if len(bayer_key) != 4:
        return bayer_key
    top, bottom = bayer_key[0:2], bayer_key[2:4]
    if flip_h and width % 2 == 0:
        top, bottom = top[::-1], bottom[::-1]
    if flip_v and height % 2 == 0:
        top, bottom = bottom, top
    return top + bottom


class SerReader:
    """--source に .ser を指定したときのファイル入力。本アプリの SER 保存形式(16bit Bayer, リトルエンディアン)を読む。"""

    HEADER_SIZE = 178
    COLOR_ID_TO_BAYER = {11: "RGGB", 10: "GRBG", 9: "GBRG", 8: "BGGR"}

    def __init__(self, path):
        import struct
        self.path = path
        self._f = open(path, "rb")
        try:
            head = self._f.read(self.HEADER_SIZE)
            if len(head) < self.HEADER_SIZE or head[:14] != b"LUCAM-RECORDER":
                raise ValueError("SERヘッダーではありません")
            color_id, _little, w, h, depth, n = struct.unpack_from("<6I", head, 18)
            if depth != 16:
                raise ValueError(f"未対応の PixelDepth: {depth} (16bitのみ対応)")
            self.width, self.height = w, h
            self.bayer = self.COLOR_ID_TO_BAYER.get(color_id)
            self.frame_size = w * h * 2
            # ヘッダーのフレーム数が0(録画中断)やファイルより多い場合はファイルサイズから求める
            avail = (os.fstat(self._f.fileno()).st_size - self.HEADER_SIZE) // self.frame_size
            self.frame_count = min(n, avail) if n > 0 else avail
            if self.frame_count <= 0:
                raise ValueError("フレームがありません")
        except Exception:
            self._f.close()
            raise
        self._idx = 0

    @property
    def position(self):
        """直近に返したフレームの番号(1始まり)。"""
        return self._idx

    def read_at(self, index):
        """index(0始まり、範囲外は循環)のフレームを返す。"""
        self._idx = index % self.frame_count
        return self.next_frame()

    def next_frame(self):
        """次のフレームのバイト列を返す。最後まで読んだら先頭に戻る。"""
        if self._idx >= self.frame_count:
            self._idx = 0
        self._f.seek(self.HEADER_SIZE + self._idx * self.frame_size)
        data = self._f.read(self.frame_size)
        self._idx += 1
        return data

    def close(self):
        self._f.close()


def save_ser(frames, filename, bayer_key, bits, width, height, lsb=False):
    """SER形式でraw16 Bayerフレームを保存（天体撮影用マルチフレームフォーマット）

    SER は外部ライブラリ不要のシンプルなバイナリフォーマット。
    AutoStakkert! / PIPP / Registax 等で直接読み込んで再スタック可能。
    """
    import struct

    # SER ColorID マッピング（SER Player互換: RGGB↔BGGR, GRBG↔GBRG を反転）
    color_id_map = {"RGGB": 11, "GRBG": 10, "GBRG": 9, "BGGR": 8}
    color_id = color_id_map.get(bayer_key, 0)  # 不明なら MONO(0)

    # タイムスタンプ: .NET DateTime.Ticks (100ns 単位, 0001-01-01 起点)
    now_utc = datetime.datetime.utcnow()
    ticks = int((now_utc - datetime.datetime(1, 1, 1)).total_seconds() * 10_000_000)
    ticks_local = ticks + int(datetime.datetime.now().astimezone().utcoffset().total_seconds() * 10_000_000)

    # SER ヘッダー (178 bytes 固定)
    header = (
        b"LUCAM-RECORDER"            # 14 bytes: マジック
        + struct.pack("<I", 0)       #  4 bytes: LuID (0 = 未使用)
        + struct.pack("<I", color_id)#  4 bytes: ColorID
        + struct.pack("<I", 0)       #  4 bytes: LittleEndian: 0=SER Player互換 (フラグ意味が逆のため)
        + struct.pack("<I", width)   #  4 bytes: ImageWidth
        + struct.pack("<I", height)  #  4 bytes: ImageHeight
        + struct.pack("<I", 16)          #  4 bytes: PixelDepthPerPlane: 16bit (C実装に合わせ固定)
        + struct.pack("<I", len(frames))  # 4 bytes: FrameCount
        + b"LiveStack".ljust(40, b"\x00")[:40]   # 40 bytes: Observer
        + b"rpicam-raw".ljust(40, b"\x00")[:40]  # 40 bytes: Instrument
        + b"".ljust(40, b"\x00")     # 40 bytes: Telescope
        + struct.pack("<q", ticks_local)   #  8 bytes: DateTime (local)
        + struct.pack("<q", ticks)   #  8 bytes: DateTimeUTC
    )
    assert len(header) == 178, f"SER header size mismatch: {len(header)}"

    with open(filename, "wb") as f:
        f.write(header)
        shift = 0 if lsb else (16 - bits)  # LSB詰め or MSB詰め(上位詰め)
        for frame in frames:
            f.write((frame.astype(np.float32) * (1 << shift)).clip(0, 65535).astype(np.uint16).tobytes())

    print(f"[save] SER: {filename}  frames={len(frames)}  {width}x{height} {bits}bit {bayer_key}")


def _ascii_safe_header_value(key, value):
    """FITSヘッダーは標準印字可能ASCIIのみ許容されるため、非ASCII文字列は警告を出して除外する。"""
    if isinstance(value, str) and not value.isascii():
        print(f"[warn] FITSヘッダー {key} は非ASCII文字を含むため書き込みをスキップします: {value!r}")
        return None
    return value


def save_fits(image, filename, metadata):
    """FITS形式でRGB画像を保存"""
    try:
        from astropy.io import fits
    except Exception as e:
        print(f"FITS保存に失敗: astropy が利用できません ({e})")
        return

    # BGRからRGBに変換
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.moveaxis(image, -1, 0)  # 軸を変更して形状を (3, height, width) に

    hdu = fits.PrimaryHDU(image)
    header = hdu.header

    # メタデータをヘッダーに追加
    for key, value in metadata.items():
        safe_value = _ascii_safe_header_value(key, value)
        if safe_value is not None:
            header[key] = safe_value

    # NAXIS3を色数として設定
    if len(image.shape) == 3:
        header['NAXIS3'] = image.shape[0]

    hdu.writeto(filename, overwrite=True)
    print(f"FITSファイル保存: {filename}")


def save_fits_bayer_f32(image, filename, metadata):
    """スタック結果をfloat32のBayer単面（クリップ・丸めなし、無反転）でFITS保存する。後処理(HDR合成・ストレッチ)用。"""
    try:
        from astropy.io import fits
    except Exception as e:
        print(f"FITS保存に失敗: astropy が利用できません ({e})")
        return
    hdu = fits.PrimaryHDU(np.asarray(image, dtype=np.float32))
    for key, value in metadata.items():
        if key == "BITPIX":  # BITPIXはFITSの予約キーのため使わない
            continue
        safe_value = _ascii_safe_header_value(key, value)
        if safe_value is not None:
            hdu.header[key] = safe_value
    hdu.writeto(filename, overwrite=True)
    print(f"FITSファイル保存(float32 Bayer): {filename}")


def _save_dark_fits(path, dark_frame, bayer_name, dark_count):
    """ダークフレームをFITS(float32 Bayer単面)で保存する。"""
    try:
        from astropy.io import fits
    except Exception as e:
        print(f"[dark] FITS保存失敗: {e}")
        return
    hdu = fits.PrimaryHDU(dark_frame.astype(np.float32))
    hdu.header["DATE"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hdu.header["DARK_CNT"] = dark_count
    hdu.header["BAYER"] = bayer_name
    hdu.writeto(path, overwrite=True)
    print(f"[dark] 保存: {path}  ({dark_count}枚平均  shape={dark_frame.shape})")


def _load_dark_fits(path, expected_shape):
    """FITSからダークフレームを読み込む。形状不一致時は(None, 0)を返す。"""
    try:
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
    except Exception as e:
        print(f"[dark] 読み込み失敗: {path}  ({e})")
        return None, 0
    if data.shape != expected_shape:
        print(f"[dark] 形状不一致のためスキップ: ファイル={data.shape}  期待={expected_shape}")
        return None, 0
    date = hdr.get("DATE", "不明")
    cnt = int(hdr.get("DARK_CNT", 1))
    bayer = hdr.get("BAYER", "?")
    print(f"[dark] 読み込み: {path}  ({cnt}枚平均  {bayer}  取得={date})")
    return data, cnt


def _parse_rpicam_param(cmd, key):
    """rpicam-rawコマンド文字列から --key value を取り出す。見つからなければNone。"""
    m = re.search(rf'--{key}\s+(\S+)', cmd)
    return m.group(1) if m else None


def _ssh_start_rpicam(ssh_host, ssh_user, ssh_key, rpicam_cmd, port):
    """SSH鍵認証でPiに接続しrpicam-rawをバックグラウンド起動する。(client, pid)を返す。"""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_path = os.path.expanduser(ssh_key)
    # look_for_keys=Trueのままだと指定鍵が拒否された際に~/.ssh/id_rsa等を自動探索し、
    # 暗号化鍵に当たってPasswordRequiredExceptionになるため無効化する
    client.connect(ssh_host, username=ssh_user, key_filename=key_path, look_for_keys=False)

    cmd = rpicam_cmd.strip()
    if "--listen" not in cmd:
        cmd += " --listen"
    if "-o tcp://" not in cmd:
        cmd += f" -o tcp://0.0.0.0:{port}"
    if " -t " not in cmd:
        cmd += " -t 0"

    remote_cmd = f"nohup {cmd} > /tmp/rpicam_raw.log 2>&1 & echo $!"
    _, stdout, _ = client.exec_command(remote_cmd)
    pid_line = stdout.readline().strip()
    try:
        pid = int(pid_line)
    except ValueError:
        pid = None

    print(f"[SSH] {ssh_host}: {cmd}")
    print(f"[SSH] rpicam-raw 起動 PID={pid}")
    return client, pid


def _ssh_kill_rpicam(client, pid, ssh_host):
    """SSH経由でrpicam-rawプロセスを停止し接続を閉じる。"""
    if client is None:
        return
    try:
        kill_cmd = f"kill {pid}" if pid else "pkill -f rpicam-raw"
        client.exec_command(f"{kill_cmd} 2>/dev/null; true")
        print(f"[SSH] {ssh_host}: rpicam-raw 停止 (PID={pid})")
    except Exception as e:
        print(f"[SSH] プロセス停止エラー: {e}")
    finally:
        try:
            client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def run_raw_live_stack(args):
    ser_reader = None
    if args.source.lower().endswith(".ser"):
        try:
            ser_reader = SerReader(args.source)
        except (OSError, ValueError) as e:
            print(f"[error] SER読み込み失敗: {args.source}  ({e})")
            return
        # ヘッダーが宸法を持つので、ソケット用の設定やクロップは無効化する
        args.width, args.height = ser_reader.width, ser_reader.height
        args.raw_width = args.raw_height = None
        args.crop_x = args.crop_y = 0
        args.wire_format = None
        args.stride = None
        args.ssh_host = None
        if ser_reader.bayer:
            args.bayer = ser_reader.bayer
        if args.bits is None:
            args.bits = 12  # SERは16bit容器で有効ビット深度を持たない
        host = port = None
        print(f"[SER] 入力ファイル: {args.source}  {ser_reader.width}x{ser_reader.height}  {ser_reader.frame_count}フレーム (ループ再生)  bayer={args.bayer} bits={args.bits} {'LSB詰め' if args.ser_lsb else 'MSB詰め'}")
    else:
        host, port = parse_tcp_source(args.source)

    raw_w = args.raw_width if args.raw_width is not None else args.width
    raw_h = args.raw_height if args.raw_height is not None else args.height
    crop_x = args.crop_x
    crop_y = args.crop_y
    crop = (raw_w != args.width or raw_h != args.height or crop_x != 0 or crop_y != 0)

    preset_info = apply_wire_format_preset(args, raw_w)
    if preset_info is not None:
        if args.wire_format == "12p":
            print(f"[preset] wire-format=12p -> bits={preset_info['bits']} stride={preset_info['stride']}")
        else:
            print(
                f"[preset] wire-format={args.wire_format} -> bits={preset_info['bits']} "
                f"stride={preset_info['stride']} (effective_bits={preset_info['effective_bits']})"
            )

    print(f"[RAW] 接続先: {args.source}")
    print(f"[RAW] 表示解像度: {args.width}x{args.height}")
    if crop:
        print(f"[RAW] RAW解像度: {raw_w}x{raw_h} (デコード後クロップ: x={crop_x} y={crop_y} {args.width}x{args.height})")
    print(f"[RAW] Bayer: {args.bayer}  ビット: {'自動' if args.bits is None else args.bits}")
    if args.wire_format in ("12u", "10u", "16u"):
        effective_text = {"12u": "12", "10u": "10", "16u": "16"}[args.wire_format]
        print(f"[RAW] wire-format={args.wire_format} を有効化: 受信コンテナ16bit / 有効ビット{effective_text}bit として処理します")
    if args.stride:
        print(f"[RAW] stride: {args.stride} bytes/行")

    # ダークフレームを接続前に事前読み込み（args.width/heightは接続前から判明する）
    _preloaded_dark = None
    _preloaded_dark_count = 0
    if os.path.exists(args.dark_file):
        _preloaded_dark, _preloaded_dark_count = _load_dark_fits(args.dark_file, (args.height, args.width))

    # SSH経由でrpicam-rawを起動する（--ssh-host指定時のみ）
    _ssh_client = None
    _rpicam_pid = None
    if args.ssh_host:
        _ssh_client, _rpicam_pid = _ssh_start_rpicam(
            args.ssh_host, args.ssh_user, args.ssh_key, args.rpicam_cmd, port
        )
        print("[SSH] rpicam-raw 起動待ち (2秒)...")
        time.sleep(2)

    sock = None
    if ser_reader is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(args.timeout)
        sock.connect((host, port))
        print("[RAW] 接続成功")

    try:
        if ser_reader is not None:
            fmt = {
                "name": "SER 16bit",
                "bits": 16,
                "stride": ser_reader.width * 2,
                "frame_size": ser_reader.frame_size,
            }
            candidates = [fmt]
            fmt_idx = 0
            buf = bytearray()
            first_frame_bytes = ser_reader.next_frame()
        elif args.stride is not None:
            if args.bits is None:
                print("[error] --stride 指定時は --bits も必須です")
                return
            min_stride = min_stride_for_bits(raw_w, args.bits)
            if args.stride < min_stride:
                print(f"[error] bits={args.bits} には stride>={min_stride} が必要ですが、指定は {args.stride} です")
                print("[hint] 送信側が RGGB_PISP_COMP1 (例: stride 3904) の場合、この受信器の 10/12bit CSI2P デコーダとは形式が一致しません")
                print("[hint] 次のいずれかで合わせてください:")
                print("  1) 送信側を 12bit CSI2P 出力へ変更し、受信側を --bits 12 --stride 5792 にする")
                print("  2) 受信側を --bits 8 --stride 3904 にして暫定表示する（PISP_COMP1を8bit相当として扱う）")
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

        if ser_reader is not None:
            effective_info = {
                "effective_bits": int(args.bits),
                "effective_shift": 0 if args.ser_lsb else 16 - int(args.bits),
                "message": None,
            }
        else:
            effective_info = resolve_effective_bits_and_shift(args, fmt, raw_w, raw_h, first_frame_bytes)
        effective_bits = effective_info["effective_bits"]
        effective_shift = effective_info["effective_shift"]
        if effective_info["message"]:
            print(effective_info["message"])
        fmt["effective_bits"] = effective_bits
        fmt["effective_shift"] = effective_shift

        skip = args.skip
        auto_stretch = not args.no_stretch
        show_hist = True
        info_display = True
        flip_h = args.flip_h
        flip_v = args.flip_v
        bayer_keys = list(BAYER_MAP.keys())
        bayer_idx = bayer_keys.index(args.bayer)
        bayer_code = BAYER_MAP[bayer_keys[bayer_idx]]

        stack_enabled = False
        # 利用可能メモリと解像度からmax_framesの安全上限を見積もり、--max-framesをクランプする
        max_frames_limit = estimate_max_frames_limit(raw_w, raw_h)
        if args.max_frames <= 0:
            max_frames = max_frames_limit
            print(f"[info] --max-frames max: メモリ見積もり上限 {max_frames} フレームを使用します")
        else:
            max_frames = min(args.max_frames, max_frames_limit)
            if max_frames < args.max_frames:
                print(f"[warn] --max-frames {args.max_frames} はメモリ見積もり上限を超えるため {max_frames} にクランプしました")
        live_stack = LiveStack(max_frames=max_frames, verbose=False, bits=effective_bits)
        live_stack.hot_sigma = args.hot_sigma
        # auto は最初のフレームでホットピクセルの一致から向きを判定する（それまでは none）
        _dark_flip_auto = [args.dark_flip == "auto"]
        live_stack.dark_flip = args.dark_flip if args.dark_flip in ("h", "v", "hv") else "none"
        if _preloaded_dark is not None:
            live_stack.dark_frame = _preloaded_dark
            # dark_bufferにダミーを積んでDARK_CNTを保持する（終了時の再保存で枚数が失われないよう）
            live_stack.dark_buffer = [_preloaded_dark] * _preloaded_dark_count
            live_stack.build_hot_mask()
        if args.stop_threshold is not None:
            live_stack.brightness_threshold = max(1, min((1 << effective_bits) - 1, args.stop_threshold))
        if args.stop_ratio is not None:
            live_stack.overflow_ratio_threshold = max(0.01, min(0.50, args.stop_ratio / 100.0))
        live_stack.align_ratio = max(0.1, min(1.0, args.align_size / 100.0))
        live_stack.align_enabled = not args.no_align
        live_stack.dark_enabled = not args.no_dark_sub
        live_stack.stack_method = args.stack_method
        live_stack.clip_sigma = args.clip_sigma

        settings_menu = SettingsMenu()
        control_menu = ControlMenu([
            ("Save PNG", "s", None),
            ("Save JPEG", "j", None),
            ("Save FITS", "f", None),
            ("Save NPY", "r", None),
            ("SER Rec", "S", "rec"),
            ("Screen Rec (AVI)", "y", "scr"),
            ("Zoom", "z", "zoom"),
            ("ROI (select/clear)", "k", "roi"),
            ("Click WB", "w", "click_wb"),
            ("WB Reset", "W", None),
            ("Gamma Reset", "G", None),
            ("Stack Reset", "R", None),
            ("Dark Capture", "d", None),
            ("Dark Clear", "D", None),
            ("Save Config JSON", "C", None),
        ])
        stream_menu_names = {"Max Frames", "Stack Mode", "Info Display", "Histogram", "Flip H", "Flip V", "SER Flip", "Rec Raw", "Rec Stack", "Rec Folder", "Dark Flip", "Stack Method", "Stop Threshold", "Stop Ratio(%)", "Align Size(%)", "Align", "Dark Sub"}
        # Stop Threshold の範囲をビット深度に合わせて更新
        max_val = (1 << effective_bits) - 1
        for _s in settings_menu.settings:
            if _s["name"] == "Stop Threshold":
                _s["max"] = max_val
                _s["value"] = max_val
                _s["step"] = max(1, max_val // 100)
                break
        # Max Frames の上限もメモリ見積もりに合わせて更新（メニュー操作でのクランプと矛盾させない）
        for _s in settings_menu.settings:
            if _s["name"] == "Max Frames":
                _s["max"] = max_frames_limit
                break
        settings_menu.settings = [s for s in settings_menu.settings if s["name"] in stream_menu_names]
        settings_menu.set_current_values(
            camera=0,
            gain=1.0,
            exposure=16667,
            max_frames=max_frames,
            stack_mode=False,
            stack_method=STACK_METHOD_LABELS[live_stack.stack_method],
            info_display=True,
            hist_display=show_hist,
            flip_h=args.flip_h,
            flip_v=args.flip_v,
            ser_flip=args.ser_flip,
            rec_raw=not args.no_rec_raw,
            rec_stack=not args.no_rec_stack,
            rec_folder=not args.no_rec_folder,
            size_label="N/A",
            stop_ratio_percent=int(live_stack.overflow_ratio_threshold * 100),
            stop_threshold=live_stack.brightness_threshold,
            align_size_percent=int(round(live_stack.align_ratio * 100)),
            align_enabled=live_stack.align_enabled,
            dark_enabled=live_stack.dark_enabled,
            dark_flip=DARK_FLIP_LABELS["auto" if _dark_flip_auto[0] else live_stack.dark_flip],
        )

        screen_size = get_screen_size()
        # コマンドライン引数の解像度とシステム画面から表示サイズを起動時に一度だけ計算
        _dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        _sized = fit_display_frame(_dummy, screen_size=screen_size)
        disp_w, disp_h = _sized.shape[1], _sized.shape[0]
        print(f"[view] 表示サイズ: {disp_w}x{disp_h}  (入力: {args.width}x{args.height}, 画面: {screen_size})")

        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, disp_w, disp_h)
        # 既定の配置はウィンドウの左上が画面中央になり全体が見えないので、画面内に収まる位置へ移動する
        if screen_size is not None:
            cv2.moveWindow(PREVIEW_WINDOW_NAME, max(0, (screen_size[0] - disp_w) // 2), 40)

        # --- WB / ガンマ 初期化 ---
        wb_gains = [1.0, 1.0, 1.0]
        gamma_value = 0.80
        gamma_adjust_mode = False
        click_wb_mode = False
        zoom_mode = False
        zoom_rect = [None]       # (px1, py1, px2, py2) プレビュー座標
        _zoom_from_drag = [False]   # True=ドラッグ選択、False=クリック選択
        _zoom_drag = {"start": None, "dragging": False, "current": None}
        # ROI: 停止判定・位置合わせ・ヒストグラムの対象領域（Zoomとは別。座標はraw px・反転前）
        roi_ui = {"select": False, "start": None, "current": None, "dragging": False}
        if args.roi:
            live_stack.roi = normalize_roi(args.roi[0], args.roi[1], args.roi[0] + args.roi[2], args.roi[1] + args.roi[3], args.width, args.height)
            if live_stack.roi is None:
                print(f"[roi] 設定のROI {args.roi} は無効(範囲外または小さすぎる)のため無視します")
            else:
                print(f"[roi] 読み込み: {live_stack.roi_text()}")

        def slider_to_gain(v):
            return v / 100.0

        def gain_to_slider(g):
            return int(g * 100)

        def slider_to_gamma(v):
            return v / 100.0

        def gamma_to_slider(g):
            return int(g * 100)

        # Qt版OpenCVはツールバー/ステータスバーが付いてレイアウトが崩れるため GUI_NORMAL で省略
        cv2.namedWindow(WB_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.createTrackbar("B x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("G x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("R x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("Gamma x100", WB_WINDOW_NAME, 80, 400, lambda _v: None)
        cv2.resizeWindow(WB_WINDOW_NAME, 420, 120)
        if screen_size is not None:
            cv2.moveWindow(WB_WINDOW_NAME, 0, max(0, screen_size[1] - 160))
        if args.wb_b is not None:
            cv2.setTrackbarPos("B x100", WB_WINDOW_NAME, gain_to_slider(args.wb_b))
        if args.wb_g is not None:
            cv2.setTrackbarPos("G x100", WB_WINDOW_NAME, gain_to_slider(args.wb_g))
        if args.wb_r is not None:
            cv2.setTrackbarPos("R x100", WB_WINDOW_NAME, gain_to_slider(args.wb_r))
        if args.gamma is not None:
            cv2.setTrackbarPos("Gamma x100", WB_WINDOW_NAME, gamma_to_slider(args.gamma))

        def _preview_to_raw(px, py):
            pw, ph = preview_size_ref[0]
            dw, dh = display_size_ref[0]
            dx, dy = px * dw / max(1, pw), py * dh / max(1, ph)
            return (dw - dx if flip_h else dx), (dh - dy if flip_v else dy)

        def _apply_roi(roi):
            live_stack.roi = roi
            do_stack_reset()
            if roi is None:
                print("[roi] クリア（全体で判定・中央窓で位置合わせ）")
                _hud_msgs.append(("[roi] クリア", time.time() + 4))
            else:
                print(f"[roi] 設定: {live_stack.roi_text()}")
                _hud_msgs.append((f"[roi] {live_stack.roi_text()}", time.time() + 4))
                if min(roi[2] - roi[0], roi[3] - roi[1]) < ROI_ALIGN_MIN_SIZE:
                    print(f"[roi] 辺が{ROI_ALIGN_MIN_SIZE}px未満のため、位置合わせ窓はROIの中心で{ROI_ALIGN_MIN_SIZE}pxまで広げます（停止判定・ヒストグラムはROIそのもの）")

        def on_mouse(event, x, y, flags, param):
            if roi_ui["select"]:
                if event == cv2.EVENT_LBUTTONDOWN:
                    roi_ui.update(start=(x, y), current=(x, y), dragging=True)
                elif event == cv2.EVENT_MOUSEMOVE and roi_ui["dragging"]:
                    roi_ui["current"] = (x, y)
                elif event == cv2.EVENT_LBUTTONUP and roi_ui["dragging"]:
                    x0, y0 = roi_ui["start"]
                    roi_ui.update(dragging=False, select=False, start=None, current=None)
                    if abs(x - x0) < 8 and abs(y - y0) < 8:
                        print("[roi] 選択を取消（ドラッグで範囲を指定してください）")
                        return
                    ax, ay = _preview_to_raw(x0, y0)
                    bx, by = _preview_to_raw(x, y)
                    dw, dh = display_size_ref[0]
                    roi = normalize_roi(ax, ay, bx, by, dw, dh)
                    if roi is None:
                        print(f"[roi] 小さすぎます（{ROI_MIN_SIZE}px以上）。もう一度 [k] で選択してください")
                    else:
                        _apply_roi(roi)
                return
            if zoom_mode:
                if event == cv2.EVENT_LBUTTONDOWN:
                    _zoom_drag["start"] = (x, y)
                    _zoom_drag["dragging"] = True
                    _zoom_drag["current"] = (x, y)
                elif event == cv2.EVENT_MOUSEMOVE and _zoom_drag["dragging"]:
                    _zoom_drag["current"] = (x, y)
                elif event == cv2.EVENT_LBUTTONUP and _zoom_drag["dragging"]:
                    _zoom_drag["dragging"] = False
                    x0, y0 = _zoom_drag["start"]
                    x1, y1 = x, y
                    if abs(x1 - x0) < 8 and abs(y1 - y0) < 8:
                        # クリック: zoom_click_size 原解像度画素を中心から切り出す
                        half = args.zoom_click_size // 2
                        zoom_rect[0] = (x0 - half, y0 - half, x0 + half, y0 + half)
                        _zoom_from_drag[0] = False
                    else:
                        zoom_rect[0] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
                        _zoom_from_drag[0] = True
            elif event == cv2.EVENT_LBUTTONDOWN and click_wb_mode:
                base = display_frame_ref[0]
                if base is None:
                    return
                sw, sh = base.shape[1], base.shape[0]
                pw, ph = preview_size_ref[0]
                sx = int(x * sw / max(1, pw))
                sy = int(y * sh / max(1, ph))
                gains = compute_wb_gains_from_patch(base, sx, sy, radius=12)
                if gains is None:
                    return
                wb_gains[0], wb_gains[1], wb_gains[2] = gains
                cv2.setTrackbarPos("B x100", WB_WINDOW_NAME, gain_to_slider(wb_gains[0]))
                cv2.setTrackbarPos("G x100", WB_WINDOW_NAME, gain_to_slider(wb_gains[1]))
                cv2.setTrackbarPos("R x100", WB_WINDOW_NAME, gain_to_slider(wb_gains[2]))
                print(f"[wb] click ({sx},{sy}) -> B={wb_gains[0]:.2f} G={wb_gains[1]:.2f} R={wb_gains[2]:.2f}")

        display_frame_ref = [None]   # クリックWBのためにdisplay_frame参照を保持
        display_size_ref = [(args.width, args.height)]  # 表示フレーム(=raw px)のサイズ。ROIの座標変換用
        preview_size_ref = [(disp_w, disp_h)]
        cv2.setMouseCallback(PREVIEW_WINDOW_NAME, on_mouse)

        print("\n操作: [q]終了  [m]設定メニュー  [i]情報表示  [s]PNG保存  [r]NPY保存  [f]FITS保存")
        print("       [n]次候補フォーマット  [+/-]skip±1行  [a]ストレッチ切替  [A]位置合わせ切替  [E]ダーク減算切替  [b]Bayer切替")
        print("       [o]停止判定モード切替（超過フレームを含む/含まない）  [k]ROI指定/クリア（停止判定・位置合わせ・ヒストグラムの対象領域）")
        print("       [h]左右反転  [v]上下反転  [H]ヒストグラム切替  [t]LiveStack ON/OFF  [R]LiveStackリセット")
        print("       [w]白点クリックWB  [W]/[u]WBリセット(Qt版OpenCVはShift不可のため[u])  [g]ガンマ調整モード  [G]ガンマリセット")
        print("       [d]ダークフレーム取得（レンズキャップして押す）  [D]ダーククリア")
        print("       [c]操作メニュー（保存・SER録画・リセット・ズーム等をメニューから実行）")
        print("       [y]画面録画AVI開始/停止（info/ヒストグラム/メニュー込み。SNS用デモ向け）")
        print("       [S]/[x]SER録画開始/停止トグル（Linux/Qt版OpenCVではShiftが効かないため[x]を使用）")
        if ser_reader is not None:
            print("       [space]一時停止/再生  [,][.]コマ戻し/送り  [[][]]先頭/末尾へ  [l]ループ切替(リセット→継続→なし)")

        frame_count = 0
        last_raw16 = None      # 受信した生フレーム（ダーク取得・NPY・SER録画に使う。ダーク減算しない）
        last_view16 = None     # 表示・統計用（スタックOFFかつDark Sub ONならダーク減算後）
        save_frame = None
        prev_frame_t = time.perf_counter()

        perf_input_fps = [0.0]
        perf_stack_ms = [0.0]
        perf_stack_hz = [0.0]

        def ema(prev, value, alpha=0.25):
            if prev <= 0.0:
                return value
            return (1.0 - alpha) * prev + alpha * value

        # --- SERトグル録画状態 ---
        # 画面録画（info/ヒストグラム/メニュー等のオーバーレイ込みのプレビューをAVIに保存）
        screen_rec = ScreenRecorder(args.screen_fps, args.rec_dir or ".")
        _ser_recording = False
        _ser_raw = None        # カメラRAWのSER（後処理用。反転しない）
        _ser_stack = None      # スタック済みフレームのSER（公開用。スタックON中のみ書く）
        _ser_base = ""         # 出力ファイル名の前半（フォルダ有効時はフォルダ内のパス）
        _ser_flip_h = False    # 録画開始時点のフリップ状態を固定（スタックSERのみ）
        _ser_flip_v = False
        _ser_size = (0, 0)
        _ser_want_stack = False
        # 画面上に一時表示する通知メッセージ: [(text, expire_time), ...]
        _hud_msgs = []
        if live_stack.dark_frame is not None:
            _hud_msgs.append((f"[dark] {args.dark_file} 読み込み済み", time.time() + 6))
        frame_bytes = first_frame_bytes
        _ser_interval = 1.0 / max(args.ser_fps, 0.01)
        _ser_next_t = time.perf_counter()
        ser_paused = False
        ser_loop_mode = args.ser_loop_mode  # reset: 先頭に戻るときスタックをリセット / continuous: リセットせず続ける / off: 末尾で停止
        _frame_fresh = True  # False: 同じフレームの再表示(一時停止中)

        def _open_ser(suffix, bayer_key):
            w_r, h_r = _ser_size
            return SerWriter(
                f"{_ser_base}_{suffix}.ser", w_r, h_r, bayer_key,
                observer=getattr(args, "observer", None) or f"WB B:{wb_gains[0]:.2f} G:{wb_gains[1]:.2f} R:{wb_gains[2]:.2f} g:{gamma_value:.2f}",
                instrument=getattr(args, "instrument", None) or f"rpicam-raw {args.wire_format or ''}",
                telescope=getattr(args, "telescope", None) or f"stk={live_stack.stack_count} {effective_bits}bit {bayer_key}",
            )

        def _stack_ser_bayer_key():
            return flip_bayer_key(bayer_keys[bayer_idx], _ser_flip_h, _ser_flip_v, *_ser_size)

        def _copy_rec_extras(folder):
            """後処理で再現できるよう、使ったダークと設定JSONを録画フォルダに置く。"""
            if live_stack.dark_frame is not None and live_stack.dark_enabled:
                dst = os.path.join(folder, os.path.basename(args.dark_file))
                if not live_stack.dark_dirty and os.path.isfile(args.dark_file):
                    shutil.copy2(args.dark_file, dst)
                    print(f"[rec] ダークをコピー: {dst}")
                else:
                    _save_dark_fits(dst, live_stack.dark_frame, bayer_keys[bayer_idx], len(live_stack.dark_buffer))
            used = [p for p in (args.config, getattr(args, "meta", None)) if p]
            cfg_dir = os.path.dirname(os.path.abspath(args.config)) if args.config else ""
            meta_copied = False
            for p in used:
                src = p if os.path.isfile(p) else os.path.join(cfg_dir, p)
                if os.path.isfile(src):
                    shutil.copy2(src, os.path.join(folder, os.path.basename(src)))
                    print(f"[rec] 設定をコピー: {os.path.join(folder, os.path.basename(src))}")
                    meta_copied = meta_copied or p == getattr(args, "meta", None)
                elif p == getattr(args, "meta", None):
                    print(f"[rec] metaファイルが見つかりません: {p}")
            if not args.config:
                _save_config(os.path.join(folder, "config.json"), args)
            if not meta_copied:
                # metaファイルが無い/コピーできない場合でも、読み込み済みの観測情報を残す
                meta_vals = {k: getattr(args, k) for k in _OBS_META_KEYS if getattr(args, k, None)}
                if meta_vals:
                    meta_name = os.path.basename(getattr(args, "meta", None) or "meta.json")
                    with open(os.path.join(folder, meta_name), "w", encoding="utf-8") as f:
                        json.dump(meta_vals, f, ensure_ascii=False, indent=2)
                    print(f"[rec] 観測メタ情報を保存: {os.path.join(folder, meta_name)}")

        def _close_ser_files(reason):
            nonlocal _ser_raw, _ser_stack
            for label, w in (("RAW", _ser_raw), ("STACK", _ser_stack)):
                if w is not None:
                    w.close()
                    print(f"[SER] {reason}: {w.path}  {w.frame_count}フレーム ({label})")
            _ser_raw = _ser_stack = None

        def _save_stem():
            # 静止画(PNG/JPEG/FITS/NPY)は --rec-dir 直下に日時名で保存する（1ファイルなので日時フォルダは作らない）
            stem = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
            if ser_reader is not None:
                stem = f"{stem}_f{ser_reader.position:04d}"
            rec_root = args.rec_dir or "."
            os.makedirs(rec_root, exist_ok=True)
            return os.path.join(rec_root, stem)

        # --- バックグラウンドスタックスレッド ---
        _stack_event = threading.Event()   # 新フレーム追加を通知
        _stack_stop  = threading.Event()   # スレッド停止フラグ
        _stacked_bgr = [None]              # 最新スタック結果 (BGRフレーム)
        _last_worker_frame_id = [None]     # 同一フレーム再計算を避けるための最後に処理したID

        def do_stack_reset():
            """live_stack.reset() + リングバッファ + 表示バッファのクリアを一括実行"""
            live_stack.reset()
            live_stack.clear_buffer()
            _stacked_bgr[0] = None
            _last_worker_frame_id[0] = None

        def _stack_worker():
            while not _stack_stop.is_set():
                triggered = _stack_event.wait(timeout=0.5)
                _stack_event.clear()
                if _stack_stop.is_set():
                    break
                if not triggered or not stack_enabled:
                    continue

                latest_id = live_stack.get_latest_frame_id()
                if latest_id is None:
                    continue
                if _last_worker_frame_id[0] == latest_id:
                    # 新しいフレームが来ていない間は再計算しない
                    continue

                _t0 = time.perf_counter()
                try:
                    result = live_stack.process_stack(
                        bayer_code, stretch=auto_stretch, should_continue=lambda: stack_enabled
                    )
                except Exception as e:
                    print(f"[stack worker error] {e}")
                    result = None
                _t1 = time.perf_counter()
                elapsed_ms = (_t1 - _t0) * 1000.0
                perf_stack_ms[0] = ema(perf_stack_ms[0], elapsed_ms)
                if elapsed_ms > 0.0:
                    perf_stack_hz[0] = ema(perf_stack_hz[0], 1000.0 / elapsed_ms)
                print(f"[perf] process_stack: {(_t1 - _t0) * 1000:.0f}ms  align: {live_stack.last_align_ms:.0f}ms  frames={live_stack.stack_count}")
                if result is not None:
                    _stacked_bgr[0] = result
                    _last_worker_frame_id[0] = latest_id

        _stack_thread = threading.Thread(
            target=_stack_worker, daemon=True, name="stack-worker"
        )
        _stack_thread.start()

        while True:
            # --- デコード ---
            try:
                payload = frame_bytes[skip:skip + fmt["frame_size"]]
                raw16 = decode_to_raw16(payload, fmt, raw_w, raw_h)
                if crop:
                    raw16 = raw16[crop_y:crop_y + args.height, crop_x:crop_x + args.width]
                raw16 = normalize_effective_raw16(raw16, fmt["bits"], effective_bits, effective_shift)

                # Dark Flip: auto は最初のフレームで、ダークと生フレームのホットピクセルの一致から向きを決める
                if _dark_flip_auto[0] and live_stack.dark_frame is not None:
                    _dark_flip_auto[0] = False
                    print("[dark] ダークの向きを判定中...")
                    found = live_stack.detect_dark_flip(raw16)
                    if found is None:
                        print("[dark] ダークの向きを判定できませんでした（一致が不十分）。Dark Flip を手動で指定してください")
                        found = "none"
                    live_stack.set_dark_flip(found)
                    do_stack_reset()
                    settings_menu.set_value("Dark Flip", DARK_FLIP_LABELS[found])
                    print(f"[dark] Dark Flip: {DARK_FLIP_LABELS[found]}")

                # スタックOFFでもDark Sub ONならダーク減算後の映像を表示する（スタックONはスタック側で減算する）
                view16 = raw16
                dark = live_stack.dark_eff
                if not stack_enabled and dark is not None and live_stack.dark_enabled and dark.shape == raw16.shape:
                    sub = np.clip(raw16.astype(np.float32) - dark, 0, None)
                    live_stack._fix_hot(sub)
                    view16 = np.clip(sub, 0, 65535).astype(np.uint16)

                # スタック用: ストレッチなし・生のデベイヤ画像
                bgr = raw16_to_bgr8(view16, effective_bits, bayer_code)

                # 表示用: ストレッチあり or なし
                if auto_stretch:
                    bgr_disp, s_lo, s_hi = raw16_to_bgr8_stretched(view16, bayer_code)
                else:
                    bgr_disp = bgr.copy()
                    s_lo, s_hi = 0.0, float((1 << effective_bits) - 1)

                last_raw16 = raw16
                last_view16 = view16
                if _frame_fresh:
                    frame_count += 1
                    now_t = time.perf_counter()
                    dt = now_t - prev_frame_t
                    prev_frame_t = now_t
                    if dt > 0.0:
                        perf_input_fps[0] = ema(perf_input_fps[0], 1.0 / dt)
            except Exception as e:
                print(f"[decode error] {e}")
                bgr = bgr_disp = np.zeros((args.height // 4, args.width // 4, 3), dtype=np.uint8)
                s_lo, s_hi = 0.0, 1.0

            # --- LiveStack ---
            # raw16 Bayerをそのままバッファへ（12bit情報を消さずに保持）
            if _frame_fresh:
                live_stack.add_to_buffer(raw16, frame_count)

            # --- SER リアルタイム追記 ---
            # RAW SER: 受信フレームをそのまま / スタックSER: スタックON中のスタック加算値（反転はスタックSERのみ）
            if _ser_recording and _frame_fresh:
                shift = 0 if args.ser_lsb else (16 - effective_bits)  # LSB詰め or MSB詰め(上位詰め)
                if _ser_raw is not None and raw16 is not None:
                    _ser_raw.write_frame((raw16.astype(np.float32) * (1 << shift)).clip(0, 65535).astype(np.uint16))
                if _ser_want_stack and stack_enabled and live_stack.stacked_raw16 is not None:
                    if _ser_stack is None:
                        _ser_stack = _open_ser("stack", _stack_ser_bayer_key())
                        print(f"[SER] スタックSER開始: {_ser_stack.path}")
                    ser_u16 = (live_stack.stacked_raw16.astype(np.float32) * (1 << shift)).clip(0, 65535).astype(np.uint16)
                    if _ser_flip_h and _ser_flip_v:
                        ser_u16 = cv2.flip(ser_u16, -1)
                    elif _ser_flip_h:
                        ser_u16 = cv2.flip(ser_u16, 1)
                    elif _ser_flip_v:
                        ser_u16 = cv2.flip(ser_u16, 0)
                    _ser_stack.write_frame(ser_u16)

            _frame_fresh = False

            if stack_enabled:
                _stack_event.set()           # バックグラウンドスタックを起動
                stacked_result = _stacked_bgr[0]  # 最後に完成したスタック結果を使用
                if stacked_result is None:
                    stacked_result = bgr_disp
                save_frame = stacked_result.copy()
                display_frame = stacked_result.copy()
            else:
                save_frame = bgr_disp.copy()
                display_frame = bgr_disp.copy()

            if flip_h and flip_v:
                save_frame = cv2.flip(save_frame, -1)
                display_frame = cv2.flip(display_frame, -1)
            elif flip_h:
                save_frame = cv2.flip(save_frame, 1)
                display_frame = cv2.flip(display_frame, 1)
            elif flip_v:
                save_frame = cv2.flip(save_frame, 0)
                display_frame = cv2.flip(display_frame, 0)

            # Now表示とCCDF描画で同じ母集団を使うため、ここでネイティブ統計配列を1回だけ確定する。
            def _stats_view(arr):
                """統計用に間引いた配列。ROIがあればその範囲だけ（小さいROIは間引きを弱める）。"""
                if arr is None:
                    return None
                step = 4
                if live_stack.roi is not None:
                    arr = live_stack.roi_view(arr)
                    step = max(1, min(4, int(np.sqrt(arr.shape[0] * arr.shape[1]) // 128)))
                return arr[::step, ::step]

            native_stats = None
            if stack_enabled and live_stack.stacked_raw16 is not None:
                max_native = float((1 << live_stack.bits) - 1)
                native_stats = np.clip(_stats_view(live_stack.stacked_raw16), 0.0, max_native)

            display_frame_ref[0] = save_frame  # クリックWB用に WB適用前を保持
            display_size_ref[0] = (save_frame.shape[1], save_frame.shape[0])

            # --- WBウィンドウに現在値を表示（グレー領域を埋める）---
            wb_gains[0] = slider_to_gain(max(10, cv2.getTrackbarPos("B x100", WB_WINDOW_NAME)))
            wb_gains[1] = slider_to_gain(max(10, cv2.getTrackbarPos("G x100", WB_WINDOW_NAME)))
            wb_gains[2] = slider_to_gain(max(10, cv2.getTrackbarPos("R x100", WB_WINDOW_NAME)))
            gamma_value = slider_to_gamma(max(10, cv2.getTrackbarPos("Gamma x100", WB_WINDOW_NAME)))
            wb_info = np.zeros((28, 420, 3), dtype=np.uint8)
            cv2.putText(wb_info,
                f"B:{wb_gains[0]:.2f}  G:{wb_gains[1]:.2f}  R:{wb_gains[2]:.2f}  Gamma:{gamma_value:.2f}  click:{'ON' if click_wb_mode else 'OFF'}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow(WB_WINDOW_NAME, wb_info)

            # --- 表示（先にリサイズしてからWB/ガンマ適用 → フルサイズ処理を回避）---
            preview_frame = fit_display_frame(display_frame, screen_size=screen_size, ratio=0.85, fallback_height=600)
            preview_size_ref[0] = (preview_frame.shape[1], preview_frame.shape[0])
            wb_applied = apply_white_balance(preview_frame, wb_gains)
            preview_frame = apply_gamma_correction(wb_applied, gamma_value)

            # --- 情報オーバーレイ ---
            if info_display:
                src_h, src_w = display_frame.shape[:2]
                stretch_label = "stretch" if auto_stretch else "raw"
                bayer_name = bayer_keys[bayer_idx]
                lines = [
                    f"{args.source}",
                    f"{src_w}x{src_h}  frame#{frame_count}"
                    + (f"  SER {ser_reader.position}/{ser_reader.frame_count}{'  PAUSE' if ser_paused else ''}{'  loop:continuous' if ser_loop_mode == 'continuous' else ('' if ser_loop_mode == 'reset' else '  no-loop')}" if ser_reader is not None else ""),
                    f"{fmt['name'].strip()} | Bayer:{bayer_name} | {stretch_label} | Stack:{STACK_METHOD_LABELS[live_stack.stack_method] if stack_enabled else 'OFF'}",
                ]
                if args.wire_format in ("12u", "10u", "16u"):
                    lines.append(
                        f"{args.wire_format} decode: effective={effective_bits}bit (>>{effective_shift})"
                    )
                lines.append(
                    f"WB B:{wb_gains[0]:.2f} G:{wb_gains[1]:.2f} R:{wb_gains[2]:.2f} | Gamma:{gamma_value:.2f} | click:{'ON' if click_wb_mode else 'OFF'}"
                )
                if live_stack.roi is not None:
                    lines.append(f"ROI: {live_stack.roi_text()}")
                if live_stack.dark_frame is not None:
                    lines.append(f"Dark: {'ON' if live_stack.dark_enabled else 'OFF'} ({len(live_stack.dark_buffer) or '読込'}枚)  flip:{DARK_FLIP_LABELS[live_stack.dark_flip]}")
                if last_view16 is not None:
                    vmin = int(last_view16.min())
                    vmax = int(last_view16.max())
                    vmean = float(last_view16.mean())
                    lines.append(f"min={vmin}  max={vmax}  mean={vmean:.1f}  (max={(1 << effective_bits) - 1})")
                    if auto_stretch:
                        lines.append(f"stretch[{s_lo:.0f} - {s_hi:.0f}]")
                    print(f"[frame#{frame_count}] {'  '.join(lines[2:])}")
                if stack_enabled:
                    # Now比率を計算（CCDFと同じネイティブbit深度データ基準）
                    if native_stats is not None:
                        current_ratio = compute_white_ratio_native(
                            native_stats,
                            live_stack.brightness_threshold,
                        )
                    elif last_view16 is not None:
                        current_ratio = compute_white_ratio_native(
                            _stats_view(last_view16),
                            live_stack.brightness_threshold,
                        )
                    else:
                        current_ratio = 0.0
                    lines.append("---")
                    lines.append(f"Frames: {live_stack.stack_count} / {live_stack.max_frames}")
                    lines.append(
                        f"Now: {current_ratio * 100:.2f}% / Stop@>= {live_stack.overflow_ratio_threshold * 100:.2f}%  Thresh:{live_stack.brightness_threshold}"
                    )
                    lines.append(
                        f"StopReason: {live_stack.last_stop_reason}  lastRatio:{live_stack.last_overflow_ratio * 100:.2f}%"
                    )
                    lines.append(
                        f"StopMode: {'include-crossing-frame' if live_stack.include_overflow_frame else 'exclude-crossing-frame'}"
                    )
                    if perf_input_fps[0] > 0.0 or perf_stack_hz[0] > 0.0:
                        lag_frames = 0
                        if live_stack.last_processed_frame_id is not None:
                            lag_frames = max(0, frame_count - int(live_stack.last_processed_frame_id))
                        util = 0.0
                        if perf_input_fps[0] > 1e-6:
                            util = perf_stack_hz[0] / perf_input_fps[0]
                        lines.append(
                            f"Perf InFPS:{perf_input_fps[0]:.2f}  StackHz:{perf_stack_hz[0]:.2f}  StackMs:{perf_stack_ms[0]:.0f}  Util:{util:.2f}x  Lag:{lag_frames}frm"
                        )
                draw_info_lines(preview_frame, lines)

            if info_display and show_hist:
                if stack_enabled and native_stats is not None:
                    # 加算累積を max_native でクリップ → 溢れたピクセルが最大ビンに積まれ
                    # CCDFが閾値/比率の交点を正しく通過する
                    preview_frame = draw_ccdf_overlay(
                        preview_frame,
                        save_frame,
                        brightness_threshold=live_stack.brightness_threshold,
                        stop_ratio=live_stack.overflow_ratio_threshold,
                        bits=live_stack.bits,
                        native_source=native_stats,
                        stretch_lo=None,
                        stretch_hi=None,
                    )
                else:
                    # Stack OFF: 単フレームraw16で真の12bitヒストグラム＋stretchマーカー表示
                    # 1/4サブサンプリング（統計的な精度は十分）
                    hist_native = _stats_view(last_view16).astype(np.float32) if last_view16 is not None else None
                    preview_frame = draw_ccdf_overlay(
                        preview_frame,
                        save_frame,
                        brightness_threshold=live_stack.brightness_threshold,
                        stop_ratio=live_stack.overflow_ratio_threshold,
                        bits=live_stack.bits,
                        native_source=hist_native,
                        stretch_lo=s_lo,
                        stretch_hi=s_hi,
                    )

            if settings_menu.menu_active:
                preview_frame = settings_menu.draw_menu(preview_frame)
            if control_menu.menu_active:
                preview_frame = control_menu.draw_menu(preview_frame, {
                    "rec": _ser_recording, "scr": screen_rec.active, "zoom": zoom_mode, "click_wb": click_wb_mode,
                    "roi": live_stack.roi is not None or roi_ui["select"],
                })

            # --- REC インジケータ ---
            if _ser_recording:
                cv2.circle(preview_frame, (preview_frame.shape[1] - 18, 18), 8, (0, 0, 220), -1)
                cv2.putText(preview_frame, f"REC {(_ser_raw or _ser_stack).frame_count if (_ser_raw or _ser_stack) else 0}f",
                    (preview_frame.shape[1] - 100, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)

            # --- HUD通知（起動時ダーク読み込み等、1本6秒）---
            now_t = time.time()
            _hud_msgs = [(msg, exp) for msg, exp in _hud_msgs if exp > now_t]
            for i, (msg, _) in enumerate(_hud_msgs):
                cv2.putText(preview_frame, msg,
                    (10, preview_frame.shape[0] - 14 - i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1, cv2.LINE_AA)

            # --- ズーム PiP 描画 ---
            if zoom_mode:
                try:
                    if _zoom_drag["dragging"] and _zoom_drag["start"] and _zoom_drag["current"]:
                        cv2.rectangle(preview_frame,
                                      _zoom_drag["start"], _zoom_drag["current"],
                                      (0, 255, 255), 1)
                    if zoom_rect[0] is not None:
                        ph, pw = preview_frame.shape[:2]
                        dh, dw = display_frame.shape[:2]

                        if _zoom_from_drag[0]:
                            # ドラッグ: プレビュー座標を元解像度座標に直接マップ
                            dx1 = max(0, round(zoom_rect[0][0] * dw / pw))
                            dy1 = max(0, round(zoom_rect[0][1] * dh / ph))
                            dx2 = min(dw, round(zoom_rect[0][2] * dw / pw))
                            dy2 = min(dh, round(zoom_rect[0][3] * dh / ph))
                        else:
                            # クリック: zoom_click_size 原解像度画素を中心から切り出す
                            cx_p = (zoom_rect[0][0] + zoom_rect[0][2]) // 2
                            cy_p = (zoom_rect[0][1] + zoom_rect[0][3]) // 2
                            cx_d = int(cx_p * dw / pw)
                            cy_d = int(cy_p * dh / ph)
                            half = args.zoom_click_size // 2
                            dx1 = max(0, min(cx_d - half, dw - args.zoom_click_size))
                            dy1 = max(0, min(cy_d - half, dh - args.zoom_click_size))
                            dx2 = min(dw, dx1 + args.zoom_click_size)
                            dy2 = min(dh, dy1 + args.zoom_click_size)

                        # プレビュー上に選択枠を表示（元座標を逆変換）
                        rx1 = max(0, round(dx1 * pw / dw))
                        ry1 = max(0, round(dy1 * ph / dh))
                        rx2 = min(pw, round(dx2 * pw / dw))
                        ry2 = min(ph, round(dy2 * ph / dh))
                        cv2.rectangle(preview_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)

                        if dx2 > dx1 and dy2 > dy1:
                            # 元解像度クロップに WB+gamma を適用（小領域なので低コスト）
                            pip_crop = display_frame[dy1:dy2, dx1:dx2]
                            pip_crop = apply_white_balance(pip_crop, wb_gains)
                            pip_crop = apply_gamma_correction(pip_crop, gamma_value)

                            # pip_size に拡大（常に 1倍以上）
                            pip_s = args.zoom_pip_size
                            scale_f = max(1.0, min(8.0,
                                pip_s / max(1, max(pip_crop.shape[0], pip_crop.shape[1]))))
                            new_w = max(1, round(pip_crop.shape[1] * scale_f))
                            new_h = max(1, round(pip_crop.shape[0] * scale_f))
                            zoomed = cv2.resize(pip_crop, (new_w, new_h),
                                                interpolation=cv2.INTER_NEAREST)
                            zh, zw = zoomed.shape[:2]
                            margin = 8
                            corner = args.zoom_corner
                            ox = margin if corner in ("tl", "bl") else pw - zw - margin
                            oy = margin if corner in ("tl", "tr") else ph - zh - margin
                            if ox >= 0 and oy >= 0:
                                cv2.rectangle(preview_frame, (ox - 1, oy - 1), (ox + zw, oy + zh), (0, 255, 255), 1)
                                preview_frame[oy:oy + zh, ox:ox + zw] = zoomed
                except Exception as _e:
                    print(f"[zoom error] {_e}")
                    traceback.print_exc()

            # --- ROI 枠 ---
            ROI_COLOR = (255, 0, 255)  # Zoom(黄)と区別する
            ph_, pw_ = preview_frame.shape[:2]
            dw_, dh_ = display_size_ref[0]
            if roi_ui["dragging"] and roi_ui["start"] and roi_ui["current"]:
                cv2.rectangle(preview_frame, roi_ui["start"], roi_ui["current"], ROI_COLOR, 1)
            if live_stack.roi is not None and (info_display or roi_ui["select"]):
                rx0, ry0, rx1, ry1 = live_stack.roi
                if flip_h:
                    rx0, rx1 = dw_ - rx1, dw_ - rx0
                if flip_v:
                    ry0, ry1 = dh_ - ry1, dh_ - ry0
                p0 = (round(rx0 * pw_ / dw_), round(ry0 * ph_ / dh_))
                p1 = (round(rx1 * pw_ / dw_), round(ry1 * ph_ / dh_))
                cv2.rectangle(preview_frame, p0, p1, ROI_COLOR, 1)
                cv2.putText(preview_frame, "ROI", (p0[0] + 4, max(14, p0[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 1, cv2.LINE_AA)
                if live_stack.align_enabled:
                    # 位置合わせ窓（ROIより広い場合だけ薄く表示）
                    aw = live_stack.align_window(dw_, dh_)
                    if aw is not None and aw != live_stack.roi:
                        ax0, ay0, ax1, ay1 = aw
                        if flip_h:
                            ax0, ax1 = dw_ - ax1, dw_ - ax0
                        if flip_v:
                            ay0, ay1 = dh_ - ay1, dh_ - ay0
                        cv2.rectangle(preview_frame, (round(ax0 * pw_ / dw_), round(ay0 * ph_ / dh_)),
                                      (round(ax1 * pw_ / dw_), round(ay1 * ph_ / dh_)), (140, 0, 140), 1)
            if roi_ui["select"]:
                cv2.putText(preview_frame, "ROI: drag to select  ([k] cancel)", (10, preview_frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROI_COLOR, 1, cv2.LINE_AA)

            # --- 画面録画 ---
            screen_rec.write(preview_frame)
            screen_rec.draw_indicator(preview_frame)  # 録画フレームに写らないよう書き込み後に描画

            cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)

            # --- キー入力 ---
            if ser_reader is not None:
                # 次フレームの時刻までキー入力を待つ（常に --ser-fps で再生速度を制限）
                key_raw = cv2.waitKeyEx(max(1, int((_ser_next_t - time.perf_counter()) * 1000)))
                _ser_next_t = max(_ser_next_t + _ser_interval, time.perf_counter())
            else:
                key_raw = cv2.waitKeyEx(1)
            key, nav = decode_key(key_raw)

            # 操作メニュー（[c]で開閉、Enterで対応するキー操作を実行）
            consumed, action_key = route_control_key(key, key_raw, settings_menu, control_menu)
            if action_key is not None:
                key = action_key
            elif consumed:
                continue

            # 設定メニュー優先
            if settings_menu.handle_key(key_raw):
                if settings_menu.apply_requested:
                    values = settings_menu.get_current_values()
                    if values["max_frames"] != live_stack.max_frames:
                        live_stack.resize_max_frames(values["max_frames"])
                        print(f"Max Frames設定: {live_stack.max_frames}")
                    new_method = STACK_LABEL_TO_METHOD.get(values["stack_method"], "sum")
                    if new_method != live_stack.stack_method:
                        live_stack.stack_method = new_method
                        _last_worker_frame_id[0] = None  # バッファは残したまま、次の更新で再計算する
                        print(f"[stack] 方式: {STACK_METHOD_LABELS[new_method]}")
                    if values["stack_mode"] != stack_enabled:
                        stack_enabled = values["stack_mode"]
                        if stack_enabled:
                            do_stack_reset()
                            print("LiveStack 有効")
                        else:
                            print("LiveStack 無効")
                    if values["info_display"] != info_display:
                        info_display = values["info_display"]
                        print(f"情報表示: {'ON' if info_display else 'OFF'}")
                    if values["hist_display"] != show_hist:
                        show_hist = values["hist_display"]
                        print(f"[hist] ヒストグラム: {'ON' if show_hist else 'OFF'}")
                    if values["flip_h"] != flip_h:
                        flip_h = values["flip_h"]
                        print(f"左右反転: {'ON' if flip_h else 'OFF'}")
                    if values["flip_v"] != flip_v:
                        flip_v = values["flip_v"]
                        print(f"上下反転: {'ON' if flip_v else 'OFF'}")
                    if values["ser_flip"] != bool(args.ser_flip):
                        args.ser_flip = values["ser_flip"]
                        print(f"[SER] 反転をスタックSERに反映: {'ON' if args.ser_flip else 'OFF'} (次回の録画開始から有効)")
                    for _key, _attr, _label in (("rec_raw", "no_rec_raw", "RAW SER"), ("rec_stack", "no_rec_stack", "スタックSER"), ("rec_folder", "no_rec_folder", "日時フォルダ")):
                        if values[_key] == getattr(args, _attr):
                            setattr(args, _attr, not values[_key])
                            print(f"[SER] {_label}: {'ON' if values[_key] else 'OFF'} (次回の録画開始から有効)")
                    live_stack.overflow_ratio_threshold = max(0.01, min(0.50, values["stop_ratio_percent"] / 100.0))
                    live_stack.brightness_threshold = max(1, min((1 << live_stack.bits) - 1, values["stop_threshold"]))
                    live_stack.align_ratio = max(0.1, min(1.0, values["align_size_percent"] / 100.0))
                    if values["align_enabled"] != live_stack.align_enabled:
                        live_stack.align_enabled = values["align_enabled"]
                        do_stack_reset()
                        print(f"[align] {'ON' if live_stack.align_enabled else 'OFF'}")
                    if values["dark_enabled"] != live_stack.dark_enabled:
                        live_stack.dark_enabled = values["dark_enabled"]
                        do_stack_reset()
                        print(f"[dark] 減算: {'ON' if live_stack.dark_enabled else 'OFF'}")
                    new_flip = DARK_FLIP_LABEL_TO_MODE.get(values["dark_flip"], "none")
                    if new_flip == "auto":
                        if not _dark_flip_auto[0]:
                            _dark_flip_auto[0] = True  # 次のフレームで判定する
                            print("[dark] Dark Flip: 次のフレームで向きを自動判定します")
                    elif _dark_flip_auto[0] or new_flip != live_stack.dark_flip:
                        _dark_flip_auto[0] = False
                        live_stack.set_dark_flip(new_flip)
                        do_stack_reset()
                        print(f"[dark] Dark Flip: {DARK_FLIP_LABELS[new_flip]}")
                continue

            if key == ord("q"):
                break
            elif key == ord("z"):
                zoom_mode = not zoom_mode
                if not zoom_mode:
                    zoom_rect[0] = None
                    _zoom_from_drag[0] = False
                    _zoom_drag["start"] = None
                    _zoom_drag["dragging"] = False
                print(f"[zoom] {'ON' if zoom_mode else 'OFF'}")
            elif key == ord("k"):
                if roi_ui["select"]:
                    roi_ui.update(select=False, dragging=False, start=None, current=None)
                    print("[roi] 選択を取消")
                elif live_stack.roi is not None:
                    _apply_roi(None)
                else:
                    roi_ui["select"] = True
                    print("[roi] プレビュー上でドラッグして範囲を指定してください（[k]で取消）")
            elif key == ord("w"):
                click_wb_mode = not click_wb_mode
                print(f"[wb] 白点クリックWB: {'ON' if click_wb_mode else 'OFF'}")
            elif key in (ord("W"), ord("u")):
                wb_gains[:] = [1.0, 1.0, 1.0]
                cv2.setTrackbarPos("B x100", WB_WINDOW_NAME, 100)
                cv2.setTrackbarPos("G x100", WB_WINDOW_NAME, 100)
                cv2.setTrackbarPos("R x100", WB_WINDOW_NAME, 100)
                print("[wb] リセット")
            elif key == ord("g"):
                gamma_adjust_mode = not gamma_adjust_mode
                print(f"[gamma] 調整モード: {'ON' if gamma_adjust_mode else 'OFF'}")
            elif key == ord("G"):
                gamma_value = 0.80
                cv2.setTrackbarPos("Gamma x100", WB_WINDOW_NAME, gamma_to_slider(gamma_value))
                print("[gamma] リセット (0.80)")
            elif gamma_adjust_mode and nav == "left":  # 左キー
                gamma_value = max(0.10, round(gamma_value - 0.05, 2))
                cv2.setTrackbarPos("Gamma x100", WB_WINDOW_NAME, gamma_to_slider(gamma_value))
                print(f"[gamma] {gamma_value:.2f}")
            elif gamma_adjust_mode and nav == "right":  # 右キー
                gamma_value = min(4.00, round(gamma_value + 0.05, 2))
                cv2.setTrackbarPos("Gamma x100", WB_WINDOW_NAME, gamma_to_slider(gamma_value))
                print(f"[gamma] {gamma_value:.2f}")
            elif key == ord("m"):
                if toggle_settings_menu(settings_menu, control_menu):
                    settings_menu.set_current_values(
                        camera=0,
                        gain=1.0,
                        exposure=16667,
                        max_frames=live_stack.max_frames,
                        stack_mode=stack_enabled,
                        stack_method=STACK_METHOD_LABELS[live_stack.stack_method],
                        info_display=info_display,
                        hist_display=show_hist,
                        flip_h=flip_h,
                        flip_v=flip_v,
                        ser_flip=args.ser_flip,
                        rec_raw=not args.no_rec_raw,
                        rec_stack=not args.no_rec_stack,
                        rec_folder=not args.no_rec_folder,
                        size_label="N/A",
                        stop_ratio_percent=int(live_stack.overflow_ratio_threshold * 100),
                        stop_threshold=live_stack.brightness_threshold,
                        align_size_percent=int(round(live_stack.align_ratio * 100)),
                        align_enabled=live_stack.align_enabled,
                        dark_enabled=live_stack.dark_enabled,
                        dark_flip=DARK_FLIP_LABELS["auto" if _dark_flip_auto[0] else live_stack.dark_flip],
                    )
                    print("設定メニューを開きました")
                else:
                    print("設定メニューを閉じました")
            elif key == ord("i"):
                info_display = not info_display
                for s in settings_menu.settings:
                    if s["name"] == "Info Display":
                        s["value"] = info_display
                        break
                print(f"情報表示: {'ON' if info_display else 'OFF'}")
            elif key == ord("s") and display_frame is not None:
                fname = f"{_save_stem()}.png"
                wb_full = apply_white_balance(display_frame, wb_gains)
                png_frame = apply_gamma_correction(wb_full, gamma_value)
                h_s, w_s = png_frame.shape[:2]
                png_info = PngImagePlugin.PngInfo()
                _loc_iso, _utc_iso = _now_local_utc_iso()
                png_info.add_text("DateTime", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                png_info.add_text("DateTimeLocal", _loc_iso)
                png_info.add_text("DateTimeUTC", _utc_iso)
                png_info.add_text("Source", args.source)
                png_info.add_text("Bits", str(effective_bits))
                png_info.add_text("Bayer", bayer_keys[bayer_idx])
                png_info.add_text("WB_B", f"{wb_gains[0]:.3f}")
                png_info.add_text("WB_G", f"{wb_gains[1]:.3f}")
                png_info.add_text("WB_R", f"{wb_gains[2]:.3f}")
                png_info.add_text("Gamma", f"{gamma_value:.3f}")
                png_info.add_text("StackCount", str(live_stack.stack_count))
                png_info.add_text("StackMode", "ON" if stack_enabled else "OFF")
                png_info.add_text("Frame", str(frame_count))
                png_info.add_text("FlipH", str(flip_h))
                png_info.add_text("FlipV", str(flip_v))
                if args.wire_format:
                    png_info.add_text("WireFormat", args.wire_format)
                for _png_key, _attr in [("Telescope", "telescope"), ("Filter", "filter"),
                                        ("Mount", "mount"), ("Object", "object"),
                                        ("Observer", "observer"), ("Instrument", "instrument")]:
                    _v = getattr(args, _attr, None)
                    if _v:
                        png_info.add_text(_png_key, str(_v))
                pil_img = Image.fromarray(cv2.cvtColor(png_frame, cv2.COLOR_BGR2RGB))
                pil_img.save(fname, "PNG", pnginfo=png_info)
                print(f"[save] PNG: {fname}  (WB/Gamma適用後, フルサイズ, tEXt付き)")
            elif key == ord("j") and save_frame is not None:
                fname = f"{_save_stem()}.jpg"
                wb_full = apply_white_balance(save_frame, wb_gains)
                jpg_frame = apply_gamma_correction(wb_full, gamma_value)
                h_j, w_j = jpg_frame.shape[:2]
                exif_bytes = _make_exif_bytes(
                    frame_count, effective_bits, bayer_keys[bayer_idx],
                    wb_gains, gamma_value, live_stack.stack_count,
                    stack_enabled, args.source, flip_h, flip_v,
                    args.wire_format, w_j, h_j,
                    shutter_us=_parse_rpicam_param(args.rpicam_cmd, "shutter") if args.rpicam_cmd else None,
                    gain=_parse_rpicam_param(args.rpicam_cmd, "gain") if args.rpicam_cmd else None,
                    telescope=getattr(args, "telescope", None),
                    obs_filter=getattr(args, "filter", None),
                    mount=getattr(args, "mount", None),
                    object_name=getattr(args, "object", None),
                    observer=getattr(args, "observer", None),
                    instrument=getattr(args, "instrument", None),
                )
                pil_img = Image.fromarray(cv2.cvtColor(jpg_frame, cv2.COLOR_BGR2RGB))
                pil_img.save(fname, "JPEG", quality=95, exif=exif_bytes)
                print(f"[save] JPEG: {fname}  (WB/Gamma適用後, フルサイズ, EXIF付き)")
            elif key in (ord("S"), ord("x")):
                if not _ser_recording:
                    # --- 録画開始 ---
                    want_raw = not args.no_rec_raw
                    want_stack = not args.no_rec_stack
                    if not (want_raw or want_stack):
                        print("[SER] Rec Raw / Rec Stack がどちらも OFF のため録画しません")
                        _hud_msgs.append(("[SER] Rec Raw/Stack が両方OFF", time.time() + 5))
                    else:
                        h_r = last_raw16.shape[0] if last_raw16 is not None else args.height
                        w_r = last_raw16.shape[1] if last_raw16 is not None else args.width
                        _ser_size = (w_r, h_r)
                        _ser_flip_h = bool(args.ser_flip and flip_h)
                        _ser_flip_v = bool(args.ser_flip and flip_v)
                        _ser_want_stack = want_stack
                        stem = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
                        rec_root = args.rec_dir or "."
                        if args.no_rec_folder:
                            os.makedirs(rec_root, exist_ok=True)
                            _ser_base = os.path.join(rec_root, stem)
                        else:
                            rec_folder_path = os.path.join(rec_root, stem)
                            os.makedirs(rec_folder_path, exist_ok=True)
                            _ser_base = os.path.join(rec_folder_path, stem)
                            _copy_rec_extras(rec_folder_path)
                        if want_raw:
                            _ser_raw = _open_ser("raw", bayer_keys[bayer_idx])
                        _ser_stack = None  # スタックONになったフレームで開く
                        _ser_recording = True
                        print(f"[SER] 録画開始: {_ser_base}_*.ser  {w_r}x{h_r} effective={effective_bits}bit (container={fmt['bits']}bit) RAW={'ON' if want_raw else 'OFF'} STACK={'ON' if want_stack else 'OFF'} (stack flip_h={_ser_flip_h} flip_v={_ser_flip_v})")
                        if want_stack and not stack_enabled:
                            print("[SER] スタックSERはスタックON(Stack Mode)の間だけ記録されます")
                else:
                    # --- 録画停止: トレーラー書き込み + フレーム数パッチ ---
                    _close_ser_files("録画停止")
                    _ser_recording = False
            elif key == ord("y"):
                screen_rec.toggle(preview_frame)
            elif key == ord("r") and last_raw16 is not None:
                fname = f"{_save_stem()}.npy"
                np.save(fname, last_raw16)
                print(f"[save] NPY: {fname}  dtype={last_raw16.dtype}  shape={last_raw16.shape}")
            elif key == ord("f") and save_frame is not None:
                _loc_iso, _utc_iso = _now_local_utc_iso()
                metadata = {
                    "STACKCNT": live_stack.stack_count,
                    "MODE": "LiveStack" if stack_enabled else "LiveView",
                    "DATE": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "DATE-OBS": _utc_iso.rstrip("Z"),
                    "TIMESYS": "UTC",
                    "DATE-LOC": _loc_iso,
                    "BITPIX": effective_bits,
                    "BAYER": bayer_keys[bayer_idx],
                    "WB_B": round(float(wb_gains[0]), 4),
                    "WB_G": round(float(wb_gains[1]), 4),
                    "WB_R": round(float(wb_gains[2]), 4),
                    "GAMMA": round(float(gamma_value), 4),
                    "FRAME": frame_count,
                    "FLIP_H": flip_h,
                    "FLIP_V": flip_v,
                    "SOURCE": args.source[:68],
                }
                if args.wire_format:
                    metadata["WIRE_FMT"] = args.wire_format
                for _fits_key, _attr in [("TELESCOP", "telescope"), ("FILTER", "filter"),
                                         ("MOUNT", "mount"), ("OBJECT", "object"),
                                         ("OBSERVER", "observer"), ("INSTRUME", "instrument")]:
                    _v = getattr(args, _attr, None)
                    if _v:
                        metadata[_fits_key] = _v
                fname = f"{_save_stem()}.fits"
                if stack_enabled and live_stack.stacked_raw16 is not None:
                    # 累積スタックをuint16 BGRにデベイヤしてフリップを適用
                    stk_u16 = np.clip(live_stack.stacked_raw16, 0, 65535).astype(np.uint16)
                    bgr16 = cv2.cvtColor(stk_u16, bayer_code)
                    if flip_h and flip_v:
                        bgr16 = cv2.flip(bgr16, -1)
                    elif flip_h:
                        bgr16 = cv2.flip(bgr16, 1)
                    elif flip_v:
                        bgr16 = cv2.flip(bgr16, 0)
                    save_fits(bgr16, fname, metadata)
                    print(f"[save] FITS (uint16 BGR stacked {live_stack.stack_method}, {live_stack.stack_count}frm): {fname}")
                    # クリップ・丸めのない線形データ（Sumの65535超も保持）。反転は掛けず、向きはFLIP_H/FLIP_Vに記録
                    f32_meta = dict(metadata, NFRAMES=live_stack.stack_count, METHOD=live_stack.stack_method,
                                    EFFBITS=effective_bits, BAYER=bayer_keys[bayer_idx],
                                    DARKSUB=bool(live_stack.dark_enabled and live_stack.dark_eff is not None))
                    f32_meta.pop("MODE", None)
                    save_fits_bayer_f32(live_stack.stacked_raw16, f"{os.path.splitext(fname)[0]}_bayer_f32.fits", f32_meta)
                else:
                    save_fits(save_frame, fname, metadata)
            elif key == ord("a"):
                auto_stretch = not auto_stretch
                print(f"[stretch] {'ON' if auto_stretch else 'OFF'}")
            elif key == ord("A"):
                live_stack.align_enabled = not live_stack.align_enabled
                for s in settings_menu.settings:
                    if s["name"] == "Align":
                        s["value"] = live_stack.align_enabled
                        break
                do_stack_reset()  # 位置が混在しないようバッファを捨てる
                print(f"[align] {'ON' if live_stack.align_enabled else 'OFF'}")
            elif key == ord("E"):
                live_stack.dark_enabled = not live_stack.dark_enabled
                for s in settings_menu.settings:
                    if s["name"] == "Dark Sub":
                        s["value"] = live_stack.dark_enabled
                        break
                do_stack_reset()
                print(f"[dark] 減算: {'ON' if live_stack.dark_enabled else 'OFF'}")
            elif key == ord("h"):
                flip_h = not flip_h
                for s_item in settings_menu.settings:
                    if s_item["name"] == "Flip H":
                        s_item["value"] = flip_h
                        break
                print(f"左右反転: {'ON' if flip_h else 'OFF'}")
            elif key == ord("v"):
                flip_v = not flip_v
                for s_item in settings_menu.settings:
                    if s_item["name"] == "Flip V":
                        s_item["value"] = flip_v
                        break
                print(f"上下反転: {'ON' if flip_v else 'OFF'}")
            elif key == ord("H"):
                show_hist = not show_hist
                for s in settings_menu.settings:
                    if s["name"] == "Histogram":
                        s["value"] = show_hist
                        break
                print(f"[hist] ヒストグラム: {'ON' if show_hist else 'OFF'}")
            elif key == ord("t"):
                stack_enabled = not stack_enabled
                for s in settings_menu.settings:
                    if s["name"] == "Stack Mode":
                        s["value"] = stack_enabled
                        break
                if stack_enabled:
                    do_stack_reset()
                print(f"[stack] {'ON' if stack_enabled else 'OFF'}")
            elif key == ord("R"):
                do_stack_reset()
                print("[stack] reset")
            elif key == ord("o"):
                live_stack.include_overflow_frame = not live_stack.include_overflow_frame
                do_stack_reset()
                print(
                    f"[stack] 停止判定モード: {'超過フレームを含める' if live_stack.include_overflow_frame else '超過フレームを含めない'}"
                )
            elif key == ord("d") and last_raw16 is not None:
                live_stack.set_dark_frame(last_raw16)
                _dark_flip_auto[0] = False
                settings_menu.set_value("Dark Flip", DARK_FLIP_LABELS["none"])
                n = len(live_stack.dark_buffer)
                print(f"[dark] ダークフレーム取得: {n}枚平均")
            elif key == ord("D"):
                live_stack.dark_frame = None
                live_stack.dark_buffer = []
                live_stack.dark_flip = "none"
                _dark_flip_auto[0] = False
                settings_menu.set_value("Dark Flip", DARK_FLIP_LABELS["none"])
                live_stack.build_hot_mask()
                print("[dark] ダークフレームクリア")
            elif key == ord("C"):
                config_path = args.config if args.config else "config.json"
                # メニューで変更された可能性がある値をargsに反映してから保存
                args.stop_threshold = live_stack.brightness_threshold
                args.stop_ratio = round(live_stack.overflow_ratio_threshold * 100.0, 1)
                args.align_size = int(round(live_stack.align_ratio * 100))
                args.no_align = not live_stack.align_enabled
                args.stack_method = live_stack.stack_method
                args.no_dark_sub = not live_stack.dark_enabled
                args.roi = None if live_stack.roi is None else [live_stack.roi[0], live_stack.roi[1], live_stack.roi[2] - live_stack.roi[0], live_stack.roi[3] - live_stack.roi[1]]
                args.dark_flip = "auto" if _dark_flip_auto[0] else live_stack.dark_flip
                args.wb_b = round(float(wb_gains[0]), 3)
                args.wb_g = round(float(wb_gains[1]), 3)
                args.wb_r = round(float(wb_gains[2]), 3)
                args.gamma = round(float(gamma_value), 3)
                _save_config(config_path, args)
                _hud_msgs.append((f"[config] 保存: {config_path}", time.time() + 5))
            elif key == ord("b"):
                bayer_idx = (bayer_idx + 1) % len(bayer_keys)
                bayer_code = BAYER_MAP[bayer_keys[bayer_idx]]
                do_stack_reset()
                print(f"[bayer] {bayer_keys[bayer_idx]}")
            elif key == ord("n"):
                fmt_idx = (fmt_idx + 1) % len(candidates)
                fmt = candidates[fmt_idx]
                do_stack_reset()
                print(f"[switch] フォーマット: {fmt['name'].strip()}")
                buf.clear()
            elif (key == ord("+") or key == ord("=")) and ser_reader is None:
                skip += fmt["stride"]
                do_stack_reset()
                print(f"[skip] +{fmt['stride']} -> skip={skip}")
                buf.clear()
            elif key == ord("-") and ser_reader is None:
                skip = max(0, skip - fmt["stride"])
                do_stack_reset()
                print(f"[skip] -{fmt['stride']} -> skip={skip}")
                buf.clear()
            elif ser_reader is not None and key == ord(" "):
                ser_paused = not ser_paused
                if not ser_paused and ser_loop_mode == 'off' and ser_reader.position >= ser_reader.frame_count:
                    frame_bytes = ser_reader.read_at(0)
                    do_stack_reset()
                    _frame_fresh = True
                print(f"[SER] {'一時停止' if ser_paused else '再生'}")
            elif ser_reader is not None and key in (ord("."), ord(",")):
                ser_paused = True
                if key == ord("."):
                    if ser_reader.position >= ser_reader.frame_count and ser_loop_mode != "continuous":
                        do_stack_reset()  # 末尾→先頭の連続しないフレーム
                    frame_bytes = ser_reader.next_frame()
                else:
                    frame_bytes = ser_reader.read_at(ser_reader.position - 2)
                    do_stack_reset()
                _frame_fresh = True
            elif ser_reader is not None and (key in (ord("["), ord("]")) or nav in ("home", "end")):  # Home/End
                ser_paused = True
                to_first = key == ord("[") or nav == "home"
                frame_bytes = ser_reader.read_at(0 if to_first else ser_reader.frame_count - 1)
                do_stack_reset()
                _frame_fresh = True
            elif ser_reader is not None and key == ord("l"):
                ser_loop_mode = {"reset": "continuous", "continuous": "off", "off": "reset"}[ser_loop_mode]
                print(f"[SER] ループ: {SER_LOOP_LABELS[ser_loop_mode]}")

            # --- 次フレーム受信 ---
            if ser_reader is not None:
                if not ser_paused and ser_reader.position >= ser_reader.frame_count:
                    if ser_loop_mode != "off":
                        if ser_loop_mode == "reset":
                            do_stack_reset()  # 先頭に戻ると時系列が途切れる
                        # continuous: リセットせず再生を続け、同じ映像を加算し続けてオーバーフローまで進める
                    else:
                        ser_paused = True
                        print("[SER] 末尾で停止 ([l]ループ切替, [space]で先頭から再生)")
                if not ser_paused:
                    frame_bytes = ser_reader.next_frame()
                    _frame_fresh = True
                continue
            total_size = skip + fmt["frame_size"]
            sock.settimeout(args.timeout)
            while len(buf) < total_size:
                chunk = sock.recv(65536)
                if not chunk:
                    raise ConnectionError("接続が切れました")
                buf.extend(chunk)

            frame_bytes = bytes(buf[:total_size])
            buf = buf[total_size:]
            _frame_fresh = True

    except ConnectionError as e:
        print(f"\n[切断] {e}")
    except KeyboardInterrupt:
        print("\n[終了]")
    finally:
        # 初回フレーム受信で切断された場合など、以下の変数が未作成のまま到達することがある
        _started = "_stack_thread" in locals()
        if "screen_rec" in locals():
            screen_rec.stop("終了につき保存")
        if "_ser_recording" in locals() and _ser_recording:
            _close_ser_files("終了につき録画保存")
        if _started:
            _stack_stop.set()
            _stack_event.set()   # wait() をすぐに解除してスレッドを終了させる
            _stack_thread.join(timeout=5.0)
        # ダークフレームの自動保存（このセッションで取得・更新したときだけ。読み込んだだけのダークは書き換えない。
        # [D]クリア後は dark_frame=None のため保存しない）
        try:
            if live_stack.dark_frame is not None and live_stack.dark_dirty:
                _save_dark_fits(args.dark_file, live_stack.dark_frame,
                                bayer_keys[bayer_idx], len(live_stack.dark_buffer))
        except Exception:
            pass
        if sock is not None:
            sock.close()
        if ser_reader is not None:
            ser_reader.close()
        cv2.destroyAllWindows()
        _ssh_kill_rpicam(_ssh_client, _rpicam_pid, args.ssh_host or "")
        print("RAW Live Stack終了")


def _max_frames_arg(value):
    """--max-frames の値: 整数、または "max"/0 でメモリ見積もり上限（内部では0）"""
    if isinstance(value, str) and value.strip().lower() == "max":
        return 0
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = -1
    if n < 0:
        raise argparse.ArgumentTypeError("--max-frames は 0 以上の整数か 'max' を指定してください")
    return n


def build_arg_parser():
    parser = argparse.ArgumentParser(description="RAW Live Stack (rpicam-raw TCP receiver + LiveStack)")
    parser.add_argument("--source", required=True, help="接続先 (例: tcp://192.168.1.17:8888)。.ser を指定するとファイルをループ再生するテストモード")
    parser.add_argument("--width", type=int, default=3840, help="表示/クロップ幅 (デフォルト: 3840)")
    parser.add_argument("--height", type=int, default=2160, help="表示/クロップ高さ (デフォルト: 2160)")
    parser.add_argument("--raw-width", type=int, default=None, help="実センサーRAW幅 (省略時=--width。例: IMX678=3856)")
    parser.add_argument("--raw-height", type=int, default=None, help="実センサーRAW高さ (省略時=--height。例: IMX678=2180)")
    parser.add_argument("--crop-x", type=int, default=0, help="クロップ開始X座標 (デフォルト: 0)")
    parser.add_argument("--crop-y", type=int, default=0, help="クロップ開始Y座標 (デフォルト: 0)")
    parser.add_argument("--stride", type=int, default=None, help="1行あたりのバイト数を直接指定 (例: IMX678=5792)")
    parser.add_argument("--wire-format", type=str, default=None, choices=["12p", "12u", "10u", "16u"], help="送信側ワイヤ形式プリセット。12p=12bit packed, 12u/10u=unpacked(16bit容器), 16u=16bit真値")
    parser.add_argument("--u12-shift", type=str, default="auto", choices=["auto", "0", "4"], help="--wire-format 12u 時の有効12bit位置。auto=自動判定, 0=LSB詰め, 4=MSB詰め(>>4)")
    parser.add_argument("--u10-shift", type=str, default="auto", choices=["auto", "0", "6"], help="--wire-format 10u 時の有効10bit位置。auto=自動判定, 0=LSB詰め, 6=MSB詰め(>>6)")
    parser.add_argument("--bits", type=int, default=None, choices=[8, 10, 12, 16], help="ビット深度を強制指定 (未指定=自動推定)。迷う場合は --wire-format を優先")
    parser.add_argument("--bayer", type=str, default="BGGR", choices=list(BAYER_MAP.keys()), help="Bayerパターン (デフォルト: BGGR / IMX678確認済み)")
    parser.add_argument("--skip", type=int, default=0, help="フレーム先頭の読み飛ばしバイト数 (通常不要)")
    parser.add_argument("--timeout", type=int, default=120, help="受信タイムアウト秒 (デフォルト: 120)")
    parser.add_argument("--no-stretch", action="store_true", help="自動ストレッチを無効化 (デフォルト: ストレッチ有効)")
    parser.add_argument("--max-frames", type=_max_frames_arg, default=100, help="LiveStack最大フレーム数 (デフォルト: 100)。'max' または 0 でメモリ見積もり上限まで使う")
    parser.add_argument("--ser-fps", type=float, default=5.0, help="SERファイル入力時の再生フレームレート (デフォルト: 5)")
    parser.add_argument("--screen-fps", type=float, default=15.0, help="画面録画([y]キー/操作メニュー)のフレームレート (デフォルト: 15)")
    parser.add_argument("--ser-loop-mode", type=str, default="reset", choices=list(SER_LOOP_LABELS.keys()),
                        help="SERファイル入力の末尾での動作 (デフォルト: reset)。reset=先頭に戻りスタックをリセット / continuous=リセットせず加算を続ける(フレーム不足でオーバーフローまで加算したいとき) / off=末尾で停止。[l]キーで切替")
    parser.add_argument("--ser-lsb", action="store_true", help="SER書き出し/読み込みを下位詰め(LSB-aligned)にする (デフォルト: 上位詰めMSB-aligned)")
    parser.add_argument("--ser-flip", action="store_true", help="スタックSERに上下左右反転を反映する (録画開始時の状態で固定、ColorIDも反転後のBayerに更新)。RAW SERは常に無反転")
    parser.add_argument("--no-rec-raw", action="store_true", help="[X]録画でカメラRAWのSER(*_raw.ser)を保存しない (メニューの Rec Raw でも切替可)")
    parser.add_argument("--no-rec-stack", action="store_true", help="[X]録画でスタック済みフレームのSER(*_stack.ser)を保存しない (スタックON時のみ保存。メニューの Rec Stack でも切替可)")
    parser.add_argument("--rec-dir", type=str, default="recordings", help="[X]録画と静止画(PNG/JPEG/FITS/NPY)の出力先ルートフォルダ (デフォルト: recordings。.gitignore しやすいよう1段まとめる。\".\" でカレント)")
    parser.add_argument("--no-rec-folder", action="store_true", help="[X]録画ごとの日時フォルダを作らず --rec-dir 直下にSERだけ保存する (日時フォルダにはSER・使用したダーク・設定JSONが入る。メニューの Rec Folder でも切替可)")
    parser.add_argument("--flip-h", action="store_true", help="左右反転して起動")
    parser.add_argument("--flip-v", action="store_true", help="上下反転して起動")
    parser.add_argument("--dark-file", type=str, default="dark.fits",
                        help="ダークフレームの保存/読み込みパス (デフォルト: dark.fits)")
    parser.add_argument("--hot-sigma", type=float, default=10.0,
                        help="ダークからのホットピクセル判定しきい値 (σ倍、0=無効、デフォルト: 10)")
    parser.add_argument("--stop-threshold", type=int, default=None,
                        help="スタック打ち切り輝度しきい値 (ネイティブbit深度単位、未指定=bit深度の最大値)")
    parser.add_argument("--stop-ratio", type=float, default=None,
                        help="スタック打ち切り比率%% (1−50、未指定=10.0)")
    parser.add_argument("--align-size", type=int, default=50,
                        help="位置合わせ検出窓の大きさ 画像幅/高さに対する%% (10−100、デフォルト: 50)")
    parser.add_argument("--stack-method", type=str, default="sum", choices=list(STACK_METHOD_LABELS.keys()),
                        help="スタック方式 (デフォルト: sum)。sum=加算(飽和で停止)。lighten=比較明 / darken=比較暗 / median=中央値 / sigma=シグマクリップ平均 / sigmasum=シグマクリップ加算(平均×枚数) は最新 --max-frames 枚を使う。メニューの Stack Method でも切替可")
    parser.add_argument("--clip-sigma", type=float, default=2.5, help="--stack-method sigma の棄却しきい値(標準偏差の倍数, デフォルト: 2.5)")
    parser.add_argument("--dark-flip", type=str, default="none", choices=list(DARK_FLIP_LABELS.keys()),
                        help="ダークを使うときの反転 (デフォルト: none)。反転して録画したSER(--ser-flip)など、ダークと映像の向きが違うときに h/v/hv を指定する。auto は最初のフレームでホットピクセルの一致から向きを判定する。メニューの Dark Flip でも切替可")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                        help="停止判定(Stop Threshold/Ratio)・ヒストグラム・位置合わせの対象領域 (raw px。偶数に丸められる)。未指定=全体判定・中央窓で位置合わせ。位置合わせ窓はROIの中心で256pxまで広げる。[k]キーでもドラッグ指定でき、[C]でJSONに保存される")
    parser.add_argument("--no-align", action="store_true", help="位置合わせを無効化して単純加算 ([A]キー/メニューで切替可)")
    parser.add_argument("--no-dark-sub", action="store_true", help="ダーク減算とホットピクセル補間を無効化 ([E]キー/メニューで切替可)")
    parser.add_argument("--wb-b", type=float, default=None, help="初期WBゲイン B (デフォルト: 1.0)")
    parser.add_argument("--wb-g", type=float, default=None, help="初期WBゲイン G (デフォルト: 1.0)")
    parser.add_argument("--wb-r", type=float, default=None, help="初期WBゲイン R (デフォルト: 1.0)")
    parser.add_argument("--gamma", type=float, default=None, help="初期ガンマ値 (デフォルト: 0.80)")
    parser.add_argument("--zoom-pip-size", type=int, default=640,
                        help="ズームPiP表示サイズ（px、デフォルト: 640）")
    parser.add_argument("--zoom-click-size", type=int, default=128,
                        help="ズームクリック選択サイズ（px、デフォルト: 128）")
    parser.add_argument("--zoom-corner", default="br", choices=["tl", "tr", "bl", "br"],
                        help="PiP配置隅: tl=左上 / tr=右上 / bl=左下 / br=右下 (デフォルト: br)")
    parser.add_argument("--ssh-host", type=str, default=None,
                        help="SSH接続先ホスト。指定時にPi側でrpicam-rawを自動起動する")
    parser.add_argument("--ssh-user", type=str, default="pi",
                        help="SSHユーザー名 (デフォルト: pi)")
    parser.add_argument("--ssh-key", type=str, default="~/.ssh/id_rsa",
                        help="SSH秘密鍵パス (デフォルト: ~/.ssh/id_rsa)")
    parser.add_argument("--rpicam-cmd", type=str, default=None,
                        help="Pi側で実行するrpicam-rawコマンド。--listen/-oは自動補完される")
    parser.add_argument("--no-ssh", action="store_true",
                        help="SSH自動起動を無効化。設定ファイルのssi_hostの指定を上書きする")
    parser.add_argument("--telescope", type=str, default=None, help="観測メタ: 鏡筒名 (metaファイルの値を上書き)")
    parser.add_argument("--filter", type=str, default=None, help="観測メタ: フィルター名 (metaファイルの値を上書き)")
    parser.add_argument("--mount", type=str, default=None, help="観測メタ: 赤道儀名 (metaファイルの値を上書き)")
    parser.add_argument("--object", type=str, default=None, help="観測メタ: 観測対象名 (例: M42。metaファイルを編集せず都度指定できる)")
    parser.add_argument("--observer", type=str, default=None, help="観測メタ: 観測者名 (metaファイルの値を上書き)")
    parser.add_argument("--instrument", type=str, default=None, help="観測メタ: カメラ/センサー名 (metaファイルの値を上書き)")
    return parser


_OBS_META_KEYS = ("telescope", "filter", "mount", "object", "observer", "instrument")


def _load_config(path):
    """設定JSONを再帰的に読み込んでdictを返す。
    meta=観測機材メタ情報、host=ラズパイ個体依存(IP/SSH)、sensor=センサー依存(RAW形式等)を
    子ファイルとして持てる(観測用.json -> host.json -> sensor.json)。
    子ファイルは継承元(優先度低)、自ファイルの明示値が優先度高。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[config] 読み込み失敗: {path}  ({e})")
        return {}
    base_dir = os.path.dirname(os.path.abspath(path))
    merged = {}
    for sub_key, label in (("sensor", "sensor"), ("host", "host"), ("meta", "meta")):
        sub_path = cfg.get(sub_key)
        if not sub_path:
            continue
        sub_abs = os.path.join(base_dir, sub_path)
        sub_cfg = _load_config(sub_abs)
        if sub_cfg:
            merged.update({k: v for k, v in sub_cfg.items() if not k.startswith("_")})
            print(f"[config] {label}読み込み: {sub_abs}")
    merged.update(cfg)
    return merged


def _save_config(path, args):
    """実効引数をJSON設定ファイルに保存する。"""
    exclude = {"config", "save_config", "_comment"} | set(_OBS_META_KEYS)
    d = {k: v for k, v in vars(args).items() if k not in exclude}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"[config] 保存: {path}")


def main():
    # --config / --save-config / --meta だけ先に取り出してJSONを読む
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", nargs="?", const="config.json", default=None)
    pre.add_argument("--save-config", default=None)
    pre.add_argument("--meta", default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_arg_parser()
    parser.add_argument("--config", type=str, nargs="?", const="config.json", default=None,
                        help="JSON設定ファイルのパス。値省略時は config.json を使用")
    parser.add_argument("--save-config", type=str, default=None,
                        help="実効設定をJSONに保存して終了 (例: imx678.json)")
    parser.add_argument("--meta", type=str, default=None,
                        help="観測メタ情報JSONファイルのパス (telescope/filter/mount等)")

    if pre_args.config:
        cfg = _load_config(pre_args.config)
        if cfg:
            # JSONに source があれば --source の required を外す
            if "source" in cfg:
                for action in parser._actions:
                    if action.dest == "source":
                        action.required = False
            parser.set_defaults(**cfg)
            print(f"[config] 読み込み: {pre_args.config}")
            if "source" not in cfg and "ssh_host" not in cfg:
                print(f"[config] 警告: {pre_args.config} に source/ssh_host がありません。"
                      f"sensor専用など下位階層のファイルを直接指定していませんか？ "
                      f"観測プラン用のトップファイル(host参照を含む)を指定してください")

    # --meta はコマンドライン指定が config の "meta" キーより優先
    if pre_args.meta:
        meta_cfg = _load_config(pre_args.meta)
        if meta_cfg:
            parser.set_defaults(**{k: v for k, v in meta_cfg.items() if not k.startswith("_")})
            print(f"[config] meta読み込み(CLI): {pre_args.meta}")

    args = parser.parse_args()

    if args.no_ssh or (args.source and args.source.lower().endswith(".ser")):
        args.ssh_host = None

    if args.source is None:
        # --ssh-hostがあれば source を自動導出（ポートはデフォルト8888）
        if args.ssh_host:
            args.source = f"tcp://{args.ssh_host}:8888"
            print(f"[SSH] --source 未指定のため自動設定: {args.source}")
        else:
            parser.error("--source または --config に source、もしくは --ssh-host を指定してください")

    if args.ssh_host and not args.rpicam_cmd:
        parser.error("--ssh-host 指定時は --rpicam-cmd も必須です")

    if args.save_config:
        _save_config(args.save_config, args)
        return

    run_raw_live_stack(args)


if __name__ == "__main__":
    main()
