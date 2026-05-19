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
    [s] PNG保存
    [S] SER保存 (バッファ全フレームをマルチフレームSER)
    [r] NPY保存 (16bit RAW展開後)
    [f] FITS保存
    [n] 次のフォーマット候補に切替
    [+/-] skip ±1行
    [a] 自動ストレッチ ON/OFF
    [b] Bayerパターン切替
    [h] ヒストグラム ON/OFF
    [t] LiveStack ON/OFF
    [R] LiveStackリセット
    [w] 白点クリックWB ON/OFF
    [W] WBリセット (B/G/R=1.0)
    [g] ガンマ調整モード ON/OFF（左右キーで変更）
    [G] ガンマリセット (0.80)
    [d] ダークフレーム取得（レンズキャップして押す）
    [D] ダークフレームクリア
"""

import argparse
import os
import socket
import sys
import threading
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


PREVIEW_WINDOW_NAME = "RAW Live Stack"
WB_WINDOW_NAME = "RAW Stack WB"


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
        self.reset()
        self.buffer = [None] * max_frames  # Noneで初期化
        self.buffer_index = 0
        self.dark_frame = None  # d キーで取得するまでは None（サイズを動的に合わせるため）
        self.dark_buffer = []  # ダークフレーム用リングバッファ

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

    def add_to_buffer(self, frame):
        """リングバッファにフレームを追加"""
        self.buffer[self.buffer_index] = frame
        self.buffer_index = (self.buffer_index + 1) % self.max_frames

    def process_stack(self, bayer_code):
        """スタック処理を実行（バッファはraw16 uint16 Bayerで保持）"""
        latest_index = (self.buffer_index - 1 + self.max_frames) % self.max_frames
        if self.buffer[latest_index] is None:
            print("リングバッファが空です。スタック処理をスキップします。")
            return None

        # 最新フレーム（raw16 Bayer float32）
        latest_f32 = self.buffer[latest_index].astype(np.float32)
        if self.dark_frame is not None:
            latest_f32 = np.clip(latest_f32 - self.dark_frame, 0, None)

        stacked = latest_f32.copy()  # float32 raw16 Bayerの累積スタック

        # テンプレートマッチング用: raw16を正規化8bitに変換（暗い星も検出可能にする）
        def to_match8(f32):
            mn, mx = float(f32.min()), float(f32.max())
            if mx <= mn:
                return np.zeros(f32.shape, dtype=np.uint8)
            return ((f32 - mn) / (mx - mn) * 255).astype(np.uint8)

        h, w = latest_f32.shape[:2]
        latest_match8 = cv2.GaussianBlur(to_match8(latest_f32), (3, 3), 0)

        # 加算スタック表示用の固定スケール: 12bitなら /16 で8bitに収める
        # N枚加算するとN倍明るくなり、暗い星が徐々に浮かび上がる
        scale = float(1 << max(0, self.bits - 8))
        threshold_8 = int(self.brightness_threshold >> max(0, self.bits - 8))

        valid_stack_count = 1  # 最新フレームを含む
        for i in range(self.max_frames - 1):
            past_index = (self.buffer_index - 2 - i + self.max_frames) % self.max_frames
            past_raw16 = self.buffer[past_index]
            if past_raw16 is None:
                break

            past_f32 = past_raw16.astype(np.float32)
            if self.dark_frame is not None:
                past_f32 = np.clip(past_f32 - self.dark_frame, 0, None)

            past_match8 = cv2.GaussianBlur(to_match8(past_f32), (3, 3), 0)

            # テンプレートマッチングで位置ずれを検出
            result = cv2.matchTemplate(past_match8, latest_match8, cv2.TM_CCOEFF_NORMED)
            _, _mv, _, max_loc = cv2.minMaxLoc(result)
            offset_x, offset_y = max_loc
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            aligned = cv2.warpAffine(past_f32, M, (w, h))

            test_stack = stacked + aligned

            # オーバーフロー判定: 加算値を固定スケールで8bit換算して閾値と比較
            display_test = np.clip(test_stack / scale, 0, 255)
            overflow_ratio = float(np.sum(display_test >= threshold_8)) / display_test.size
            if overflow_ratio >= self.overflow_ratio_threshold:
                print(f"オーバーフロー条件に達しました (ratio={overflow_ratio:.3f} >= {self.overflow_ratio_threshold:.3f})。スタック処理を終了します。")
                break

            stacked = test_stack
            valid_stack_count += 1

        self.stack_count = valid_stack_count
        self.stacked_raw16 = stacked  # float32 Bayer累積スタック（FITS/NPY保存用）

        # 表示用: 加算値を固定スケール(bits→8bit)でそのまま変換
        # 1枚: 100ADU → 100/16 = 6  /  10枚加算: 1000ADU → 1000/16 = 62  → ヒストグラムが右シフト
        bayer8 = np.clip(stacked / scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(bayer8, bayer_code)

    def set_dark_frame(self, raw16_frame):
        """ダークフレームを加算平均して設定（raw16 uint16 Bayer入力）"""
        if len(self.dark_buffer) >= self.max_frames:
            self.dark_buffer.pop(0)  # 古いフレームを削除
        self.dark_buffer.append(raw16_frame.astype(np.float32))

        # 加算平均を計算（raw16空間のまま保持）
        dark_sum = np.sum(self.dark_buffer, axis=0)
        self.dark_frame = dark_sum / len(self.dark_buffer)  # float32 raw16 Bayer
        print(f"ダークフレームを更新しました。現在の平均化フレーム数: {len(self.dark_buffer)}")


# ---------------------------------------------------------------------------
# SettingsMenu クラス
# ---------------------------------------------------------------------------

class SettingsMenu:
    """設定メニュークラス"""

    def __init__(self):
        self.apply_requested = False
        self.settings = [
            {
                "name": "Camera",
                "value": 0,
                "values": [0, 1]
            },
            {
                "name": "Size",
                "value": "N/A",
                "values": ["N/A"]
            },
            {
                "name": "Gain",
                "value": 2.0,
                "min": 1.0,
                "max": 8.0,
                "step": 0.5
            },
            {
                "name": "Exposure",
                "value": 16667,  # 1/60秒
                "values": [
                    10000000,  # 10秒
                    5000000,   # 5秒
                    2000000,   # 2秒
                    1000000,   # 1秒
                    500000,    # 1/2秒
                    250000,    # 1/4秒
                    125000,    # 1/8秒
                    62500,     # 1/16秒
                    33333,     # 1/30秒
                    16667,     # 1/60秒
                    8000,      # 1/125秒
                    4000,      # 1/250秒
                    2000,      # 1/500秒
                    1000,      # 1/1000秒
                    500        # 1/2000秒
                ]
            },
            {
                "name": "Max Frames",
                "value": 100,
                "min": 1,
                "max": 100,
                "step": 1
            },
            {
                "name": "Stack Mode",
                "value": False
            },
            {
                "name": "Info Display",
                "value": True
            },
            {
                "name": "Stop Threshold",
                "value": 255,
                "min": 5,
                "max": 255,
                "step": 5
            },
            {
                "name": "Stop Ratio(%)",
                "value": 10,
                "min": 1,
                "max": 50,
                "step": 1
            }
        ]
        self.selected_item = 0
        self.menu_active = False

    def handle_key(self, key_raw):
        """キー入力処理"""
        if not self.menu_active:
            return False

        self.apply_requested = False

        # キーの下位8bit（通常キー）
        key = key_raw & 0xFF

        # waitKeyExの矢印キー値（Windows）
        if key_raw == 2490368:  # Up
            self.selected_item = (self.selected_item - 1) % len(self.settings)
            return True
        elif key_raw == 2621440:  # Down
            self.selected_item = (self.selected_item + 1) % len(self.settings)
            return True
        elif key_raw == 2424832:  # Left
            self.change_value(-1)
            return True
        elif key_raw == 2555904:  # Right
            self.change_value(1)
            return True
        # 互換用（環境差異）
        elif key in [82, 65]:  # 上
            self.selected_item = (self.selected_item - 1) % len(self.settings)
            return True
        elif key in [84, 66]:  # 下
            self.selected_item = (self.selected_item + 1) % len(self.settings)
            return True
        elif key in [81, 68]:  # 左
            self.change_value(-1)
            return True
        elif key in [83, 67]:  # 右
            self.change_value(1)
            return True
        elif key in [10, 13]:  # Enter(LF/CR) - 設定適用（メニューは閉じない）
            self.apply_requested = True
            return True
        elif key == 27:  # ESC - キャンセル
            self.menu_active = False
            return True

        return False

    def change_value(self, direction):
        """設定値を変更"""
        setting = self.settings[self.selected_item]

        if setting["name"] == "Camera":
            values = setting["values"]
            try:
                current_index = values.index(setting["value"])
            except ValueError:
                current_index = 0
            new_index = (current_index + direction) % len(values)
            setting["value"] = values[new_index]

        elif setting["name"] == "Gain":
            new_value = setting["value"] + (direction * setting["step"])
            setting["value"] = max(setting["min"], min(setting["max"], new_value))

        elif setting["name"] == "Exposure":
            values = setting["values"]
            try:
                current_index = values.index(setting["value"])
            except ValueError:
                current_index = min(range(len(values)), key=lambda i: abs(values[i] - setting["value"]))
            new_index = max(0, min(len(values)-1, current_index + direction))
            setting["value"] = values[new_index]

        elif setting["name"] == "Size":
            values = setting.get("values", [])
            if not values:
                return
            try:
                current_index = values.index(setting["value"])
            except ValueError:
                current_index = 0
            new_index = (current_index + direction) % len(values)
            setting["value"] = values[new_index]

        elif setting["name"] == "Max Frames":
            new_value = setting["value"] + (direction * setting["step"])
            setting["value"] = max(setting["min"], min(setting["max"], new_value))

        elif setting["name"] == "Stack Mode":
            setting["value"] = not setting["value"]

        elif setting["name"] == "Info Display":
            setting["value"] = not setting["value"]

        elif setting["name"] == "Stop Ratio(%)":
            new_value = setting["value"] + (direction * setting["step"])
            setting["value"] = max(setting["min"], min(setting["max"], new_value))

        elif setting["name"] == "Stop Threshold":
            new_value = setting["value"] + (direction * setting["step"])
            setting["value"] = max(setting["min"], min(setting["max"], new_value))

    def get_exposure_text(self, exposure_us):
        """露出時間をわかりやすいテキストに変換"""
        if exposure_us >= 1000000:  # 1秒以上
            seconds = exposure_us / 1000000
            if seconds == int(seconds):
                return f"{int(seconds)}s"
            else:
                return f"{seconds:.1f}s"
        else:  # 1秒未満
            denominator = int(1000000 / exposure_us)
            return f"1/{denominator}"

    def draw_menu(self, frame):
        """設定メニューを描画"""
        if not self.menu_active:
            return frame

        # 半透明の背景（設定項目数に応じて高さを調整）
        menu_height = 170 + len(self.settings) * 35 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (50, 50), (600, menu_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # タイトル
        cv2.putText(frame, "Settings Menu", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Up/Down: Select Item  Left/Right: Change Value",
                   (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Enter: Apply  ESC: Cancel",
                   (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 設定項目を表示
        for i, setting in enumerate(self.settings):
            y_pos = 170 + i * 35
            color = (0, 255, 0) if i == self.selected_item else (255, 255, 255)

            # 値のテキスト生成
            if setting["name"] == "Camera":
                value_text = f"{setting['value']}"
            elif setting["name"] == "Exposure":
                value_text = self.get_exposure_text(setting["value"])
            elif setting["name"] == "Size":
                value_text = setting["value"]
            elif setting["name"] == "Stack Mode":
                value_text = "ON" if setting["value"] else "OFF"
            elif setting["name"] == "Info Display":
                value_text = "ON" if setting["value"] else "OFF"
            elif setting["name"] == "Stop Ratio(%)":
                value_text = f"{int(setting['value'])}%"
            elif setting["name"] == "Stop Threshold":
                value_text = f"{int(setting['value'])}"
            elif setting["name"] == "Gain":
                value_text = f"{setting['value']:.1f}"
            else:
                value_text = str(setting["value"])

            text = f"{setting['name']}: {value_text}"
            cv2.putText(frame, text, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 選択中の項目にカーソル表示
            if i == self.selected_item:
                cv2.putText(frame, ">", (55, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def get_current_values(self):
        """現在の設定値を辞書で返す"""
        values_by_name = {s.get("name"): s.get("value") for s in self.settings}
        return {
            "camera": values_by_name.get("Camera", 0),
            "size_label": values_by_name.get("Size", "N/A"),
            "gain": values_by_name.get("Gain", 2.0),
            "exposure": values_by_name.get("Exposure", 16667),
            "max_frames": int(values_by_name.get("Max Frames", 100)),
            "stack_mode": bool(values_by_name.get("Stack Mode", False)),
            "info_display": bool(values_by_name.get("Info Display", True)),
            "stop_threshold": int(values_by_name.get("Stop Threshold", 255)),
            "stop_ratio_percent": int(values_by_name.get("Stop Ratio(%)", 10)),
        }

    def set_current_values(self, camera, gain, exposure, max_frames, stack_mode, info_display=True, size_label="N/A", stop_ratio_percent=10, stop_threshold=255):
        """現在の設定値を更新"""
        values = {
            "Camera": camera,
            "Size": size_label,
            "Gain": gain,
            "Exposure": exposure,
            "Max Frames": max_frames,
            "Stack Mode": stack_mode,
            "Info Display": info_display,
            "Stop Threshold": int(stop_threshold),
            "Stop Ratio(%)": int(stop_ratio_percent),
        }

        for setting in self.settings:
            name = setting.get("name")
            if name not in values:
                continue

            value = values[name]
            if name in ["Stop Threshold", "Stop Ratio(%)"] and "min" in setting and "max" in setting:
                value = max(setting["min"], min(setting["max"], int(value)))

            setting["value"] = value

    def set_size_choices(self, size_labels, current_label=None):
        """Size項目の選択肢を更新"""
        if not size_labels:
            self.settings[1]["values"] = ["N/A"]
            self.settings[1]["value"] = "N/A"
            return

        self.settings[1]["values"] = size_labels
        if current_label in size_labels:
            self.settings[1]["value"] = current_label
        else:
            self.settings[1]["value"] = size_labels[0]


# ---------------------------------------------------------------------------
# 保存ユーティリティ
# ---------------------------------------------------------------------------

def save_ser(frames, filename, bayer_key, bits, width, height):
    """SER形式でraw16 Bayerフレームを保存（天体撮影用マルチフレームフォーマット）

    SER は外部ライブラリ不要のシンプルなバイナリフォーマット。
    AutoStakkert! / PIPP / Registax 等で直接読み込んで再スタック可能。
    """
    import struct
    import datetime

    # SER ColorID マッピング（SER Player互換: RGGB↔BGGR, GRBG↔GBRG を反転）
    color_id_map = {"RGGB": 11, "GRBG": 10, "GBRG": 9, "BGGR": 8}
    color_id = color_id_map.get(bayer_key, 0)  # 不明なら MONO(0)

    # タイムスタンプ: .NET DateTime.Ticks (100ns 単位, 0001-01-01 起点)
    now_utc = datetime.datetime.utcnow()
    ticks = int((now_utc - datetime.datetime(1, 1, 1)).total_seconds() * 10_000_000)

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
        + struct.pack("<q", ticks)   #  8 bytes: DateTime (local)
        + struct.pack("<q", ticks)   #  8 bytes: DateTimeUTC
    )
    assert len(header) == 178, f"SER header size mismatch: {len(header)}"

    with open(filename, "wb") as f:
        f.write(header)
        for frame in frames:
            f.write(frame.astype(np.uint16).tobytes())

    print(f"[save] SER: {filename}  frames={len(frames)}  {width}x{height} {bits}bit {bayer_key}")


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
        header[key] = value

    # NAXIS3を色数として設定
    if len(image.shape) == 3:
        header['NAXIS3'] = image.shape[0]

    hdu.writeto(filename, overwrite=True)
    print(f"FITSファイル保存: {filename}")


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

def run_raw_live_stack(args):
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

        skip = args.skip
        auto_stretch = not args.no_stretch
        show_hist = True
        info_display = True
        bayer_keys = list(BAYER_MAP.keys())
        bayer_idx = bayer_keys.index(args.bayer)
        bayer_code = BAYER_MAP[bayer_keys[bayer_idx]]

        stack_enabled = False
        live_stack = LiveStack(max_frames=args.max_frames, verbose=False, bits=fmt["bits"])

        settings_menu = SettingsMenu()
        stream_menu_names = {"Max Frames", "Stack Mode", "Info Display", "Stop Threshold", "Stop Ratio(%)"}        
        # Stop Threshold の範囲をビット深度に合わせて更新
        max_val = (1 << fmt["bits"]) - 1
        for _s in settings_menu.settings:
            if _s["name"] == "Stop Threshold":
                _s["max"] = max_val
                _s["value"] = max_val
                _s["step"] = max(1, max_val // 100)
                break
        settings_menu.settings = [s for s in settings_menu.settings if s["name"] in stream_menu_names]
        settings_menu.set_current_values(
            camera=0,
            gain=1.0,
            exposure=16667,
            max_frames=args.max_frames,
            stack_mode=False,
            info_display=True,
            size_label="N/A",
            stop_ratio_percent=int(live_stack.overflow_ratio_threshold * 100),
            stop_threshold=live_stack.brightness_threshold,
        )

        screen_size = get_screen_size()
        # コマンドライン引数の解像度とシステム画面から表示サイズを起動時に一度だけ計算
        _dummy = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        _sized = fit_display_frame(_dummy, screen_size=screen_size)
        disp_w, disp_h = _sized.shape[1], _sized.shape[0]
        print(f"[view] 表示サイズ: {disp_w}x{disp_h}  (入力: {args.width}x{args.height}, 画面: {screen_size})")

        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, disp_w, disp_h)

        # --- WB / ガンマ 初期化 ---
        wb_gains = [1.0, 1.0, 1.0]
        gamma_value = 0.80
        gamma_adjust_mode = False
        click_wb_mode = False

        def slider_to_gain(v):
            return v / 100.0

        def gain_to_slider(g):
            return int(g * 100)

        def slider_to_gamma(v):
            return v / 100.0

        def gamma_to_slider(g):
            return int(g * 100)

        cv2.namedWindow(WB_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("B x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("G x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("R x100", WB_WINDOW_NAME, 100, 400, lambda _v: None)
        cv2.createTrackbar("Gamma x100", WB_WINDOW_NAME, 80, 400, lambda _v: None)

        def on_wb_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and click_wb_mode:
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
        preview_size_ref = [(disp_w, disp_h)]
        cv2.setMouseCallback(PREVIEW_WINDOW_NAME, on_wb_click)

        print("\n操作: [q]終了  [m]設定メニュー  [i]情報表示  [s]PNG保存  [r]NPY保存  [f]FITS保存")
        print("       [n]次候補フォーマット  [+/-]skip±1行  [a]ストレッチ切替  [b]Bayer切替")
        print("       [h]ヒストグラム切替  [t]LiveStack ON/OFF  [R]LiveStackリセット")
        print("       [w]白点クリックWB  [W]WBリセット  [g]ガンマ調整モード  [G]ガンマリセット")
        print("       [d]ダークフレーム取得（レンズキャップして押す）  [D]ダーククリア")
        print("       [S]SER録画開始/停止トグル")

        frame_count = 0
        last_raw16 = None
        save_frame = None

        # --- SERトグル録画状態 ---
        _ser_recording = False
        _ser_file = None       # 開くと BytesIO ではなく直接 open()
        _ser_frame_count = 0
        _ser_fname = ""
        _ser_timestamps = []   # フレームごとの受信時刻 Ticks（トレーラー用）
        frame_bytes = first_frame_bytes

        # --- バックグラウンドスタックスレッド ---
        _stack_event = threading.Event()   # 新フレーム追加を通知
        _stack_stop  = threading.Event()   # スレッド停止フラグ
        _stacked_bgr = [None]              # 最新スタック結果 (BGRフレーム)

        def do_stack_reset():
            """live_stack.reset() + 表示バッファのクリアを一括実行"""
            live_stack.reset()
            _stacked_bgr[0] = None

        def _stack_worker():
            while not _stack_stop.is_set():
                triggered = _stack_event.wait(timeout=0.5)
                _stack_event.clear()
                if _stack_stop.is_set():
                    break
                if not triggered or not stack_enabled:
                    continue
                _t0 = time.perf_counter()
                result = live_stack.process_stack(bayer_code)
                _t1 = time.perf_counter()
                print(f"[perf] process_stack: {(_t1 - _t0) * 1000:.0f}ms  frames={live_stack.stack_count}")
                if result is not None:
                    _stacked_bgr[0] = result

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
                    raw16 = raw16[:args.height, :args.width]

                # スタック用: ストレッチなし・生のデベイヤ画像
                bgr = raw16_to_bgr8(raw16, fmt["bits"], bayer_code)

                # 表示用: ストレッチあり or なし
                if auto_stretch:
                    bgr_disp, s_lo, s_hi = raw16_to_bgr8_stretched(raw16, bayer_code)
                else:
                    bgr_disp = bgr.copy()
                    s_lo, s_hi = 0.0, float((1 << fmt["bits"]) - 1)

                last_raw16 = raw16
                frame_count += 1
            except Exception as e:
                print(f"[decode error] {e}")
                bgr = bgr_disp = np.zeros((args.height // 4, args.width // 4, 3), dtype=np.uint8)
                s_lo, s_hi = 0.0, 1.0

            # --- LiveStack ---
            # raw16 Bayerをそのままバッファへ（12bit情報を消さずに保持）
            live_stack.add_to_buffer(raw16)

            # --- SER リアルタイム追記 ---
            if _ser_recording and _ser_file is not None and raw16 is not None:
                import datetime
                frame_f32 = raw16.astype(np.float32)
                if live_stack.dark_frame is not None:
                    frame_f32 = np.clip(frame_f32 - live_stack.dark_frame, 0, None)
                _ser_file.write(frame_f32.clip(0, 65535).astype(np.uint16).tobytes())
                # 受信時刻を .NET DateTime.Ticks (100ns単位) で記録
                now_utc = datetime.datetime.utcnow()
                ticks = int((now_utc - datetime.datetime(1, 1, 1)).total_seconds() * 10_000_000)
                _ser_timestamps.append(ticks)
                _ser_frame_count += 1

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

            display_frame_ref[0] = save_frame  # クリックWB用に WB適用前を保持

            # --- WBウィンドウに現在値を表示（グレー領域を埋める）---
            wb_gains[0] = slider_to_gain(max(10, cv2.getTrackbarPos("B x100", WB_WINDOW_NAME)))
            wb_gains[1] = slider_to_gain(max(10, cv2.getTrackbarPos("G x100", WB_WINDOW_NAME)))
            wb_gains[2] = slider_to_gain(max(10, cv2.getTrackbarPos("R x100", WB_WINDOW_NAME)))
            gamma_value = slider_to_gamma(max(10, cv2.getTrackbarPos("Gamma x100", WB_WINDOW_NAME)))
            wb_info = np.zeros((28, 400, 3), dtype=np.uint8)
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
                    f"{args.source}  {src_w}x{src_h}  frame#{frame_count}",
                    f"{fmt['name'].strip()} | Bayer:{bayer_name} | {stretch_label} | Stack:{'ON' if stack_enabled else 'OFF'}",
                ]
                if stack_enabled:
                    lines.append(f"Frames: {live_stack.stack_count} / {live_stack.max_frames}")
                lines.append(
                    f"WB B:{wb_gains[0]:.2f} G:{wb_gains[1]:.2f} R:{wb_gains[2]:.2f} | Gamma:{gamma_value:.2f} | click:{'ON' if click_wb_mode else 'OFF'}"
                )
                if last_raw16 is not None:
                    vmin = int(last_raw16.min())
                    vmax = int(last_raw16.max())
                    vmean = float(last_raw16.mean())
                    lines.append(f"min={vmin}  max={vmax}  mean={vmean:.1f}  (max={(1 << fmt['bits']) - 1})")
                    if auto_stretch:
                        lines.append(f"stretch[{s_lo:.0f} - {s_hi:.0f}]")
                    print(f"[frame#{frame_count}] {'  '.join(lines[2:])}")
                if stack_enabled:
                    threshold_8 = int(live_stack.brightness_threshold >> max(0, live_stack.bits - 8))
                    current_ratio = compute_white_ratio_uint8(save_frame, threshold_8)
                    lines.append(f"Now: {current_ratio * 100:.2f}% / Limit: {live_stack.overflow_ratio_threshold * 100:.2f}%  Thresh:{live_stack.brightness_threshold}")
                draw_info_lines(preview_frame, lines)

            if info_display and show_hist:
                if stack_enabled and live_stack.stacked_raw16 is not None:
                    # 加算累積を max_native でクリップ → 溢れたピクセルが最大ビンに積まれ
                    # CCDFが閾値/比率の交点を正しく通過する
                    max_native = float((1 << live_stack.bits) - 1)
                    # 1/4サブサンプリングで計算量を1/16に削減（統計的な精度は十分）
                    hist_native = np.clip(live_stack.stacked_raw16[::4, ::4], 0.0, max_native)
                    preview_frame = draw_ccdf_overlay(
                        preview_frame,
                        save_frame,
                        brightness_threshold=live_stack.brightness_threshold,
                        stop_ratio=live_stack.overflow_ratio_threshold,
                        bits=live_stack.bits,
                        native_source=hist_native,
                        stretch_lo=None,
                        stretch_hi=None,
                    )
                else:
                    # Stack OFF: 単フレームraw16で真の12bitヒストグラム＋stretchマーカー表示
                    # 1/4サブサンプリング（統計的な精度は十分）
                    hist_native = last_raw16[::4, ::4].astype(np.float32) if last_raw16 is not None else None
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

            # --- REC インジケータ ---
            if _ser_recording:
                cv2.circle(preview_frame, (preview_frame.shape[1] - 18, 18), 8, (0, 0, 220), -1)
                cv2.putText(preview_frame, f"REC {_ser_frame_count}f",
                    (preview_frame.shape[1] - 100, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1, cv2.LINE_AA)

            cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)

            # --- キー入力 ---
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF

            # 設定メニュー優先
            if settings_menu.handle_key(key_raw):
                if settings_menu.apply_requested:
                    values = settings_menu.get_current_values()
                    if values["max_frames"] != live_stack.max_frames:
                        live_stack.max_frames = values["max_frames"]
                        print(f"Max Frames設定: {live_stack.max_frames}")
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
                    live_stack.overflow_ratio_threshold = max(0.01, min(0.50, values["stop_ratio_percent"] / 100.0))
                    live_stack.brightness_threshold = max(1, min((1 << live_stack.bits) - 1, values["stop_threshold"]))
                continue

            if key == ord("q"):
                break
            elif key == ord("w"):
                click_wb_mode = not click_wb_mode
                print(f"[wb] 白点クリックWB: {'ON' if click_wb_mode else 'OFF'}")
            elif key == ord("W"):
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
            elif gamma_adjust_mode and key_raw in (81, 2424832):  # 左キー
                gamma_value = max(0.10, round(gamma_value - 0.05, 2))
                cv2.setTrackbarPos("Gamma x100", WB_WINDOW_NAME, gamma_to_slider(gamma_value))
                print(f"[gamma] {gamma_value:.2f}")
            elif gamma_adjust_mode and key_raw in (83, 2555904):  # 右キー
                gamma_value = min(4.00, round(gamma_value + 0.05, 2))
                cv2.setTrackbarPos("Gamma x100", WB_WINDOW_NAME, gamma_to_slider(gamma_value))
                print(f"[gamma] {gamma_value:.2f}")
            elif key == ord("m"):
                settings_menu.menu_active = not settings_menu.menu_active
                if settings_menu.menu_active:
                    settings_menu.set_current_values(
                        camera=0,
                        gain=1.0,
                        exposure=16667,
                        max_frames=live_stack.max_frames,
                        stack_mode=stack_enabled,
                        info_display=info_display,
                        size_label="N/A",
                        stop_ratio_percent=int(live_stack.overflow_ratio_threshold * 100),
                        stop_threshold=live_stack.brightness_threshold,
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
                fname = f"raw_live_stack_frame{frame_count}.png"
                wb_full = apply_white_balance(display_frame, wb_gains)
                png_frame = apply_gamma_correction(wb_full, gamma_value)
                cv2.imwrite(fname, png_frame)
                print(f"[save] PNG: {fname}  (WB/Gamma適用後, フルサイズ)")
            elif key == ord("S"):
                if not _ser_recording:
                    # --- 録画開始 ---
                    import struct, datetime
                    # ColorID=BayerパターンID: SER Player互換（RGGB↔BGGR, GRBG↔GBRG を反転）
                    color_id_map = {"RGGB": 11, "GRBG": 10, "GBRG": 9, "BGGR": 8}
                    color_id = color_id_map.get(bayer_keys[bayer_idx], 0)
                    h_r = last_raw16.shape[0] if last_raw16 is not None else args.height
                    w_r = last_raw16.shape[1] if last_raw16 is not None else args.width
                    now_utc = datetime.datetime.utcnow()
                    ticks = int((now_utc - datetime.datetime(1, 1, 1)).total_seconds() * 10_000_000)
                    header = (
                        b"LUCAM-RECORDER"
                        + struct.pack("<I", 0)
                        + struct.pack("<I", color_id)    # ColorID: Bayerパターン (8-11)
                        + struct.pack("<I", 0)            # LittleEndian: 0=SER Player互換 (フラグ意味が逆のため)
                        + struct.pack("<I", w_r)
                        + struct.pack("<I", h_r)
                        + struct.pack("<I", 16)          # PixelDepthPerPlane: 16bit (C実装に合わせ固定)
                        + struct.pack("<I", 0)   # フレーム数: 停止時にパッチ
                        + b"LiveStack".ljust(40, b"\x00")[:40]
                        + b"rpicam-raw".ljust(40, b"\x00")[:40]
                        + b"".ljust(40, b"\x00")
                        + struct.pack("<q", ticks)
                        + struct.pack("<q", ticks)
                    )
                    _ser_fname = f"raw_live_stack_frame{frame_count}.ser"
                    _ser_file = open(_ser_fname, "wb")
                    _ser_file.write(header)
                    _ser_frame_count = 0
                    _ser_timestamps.clear()
                    _ser_recording = True
                    print(f"[SER] 録画開始: {_ser_fname}  {w_r}x{h_r} {fmt['bits']}bit {bayer_keys[bayer_idx]} (ColorID={color_id})")
                else:
                    # --- 録画停止: トレーラー書き込み + フレーム数パッチ ---
                    import struct
                    for ts in _ser_timestamps:
                        _ser_file.write(struct.pack("<q", ts))
                    _ser_file.seek(38)
                    _ser_file.write(struct.pack("<I", _ser_frame_count))
                    _ser_file.close()
                    _ser_file = None
                    _ser_recording = False
                    _ser_timestamps.clear()
                    print(f"[SER] 録画停止: {_ser_fname}  {_ser_frame_count}フレーム")
            elif key == ord("r") and last_raw16 is not None:
                fname = f"raw_live_stack_frame{frame_count}_raw16.npy"
                np.save(fname, last_raw16)
                print(f"[save] NPY: {fname}  dtype={last_raw16.dtype}  shape={last_raw16.shape}")
            elif key == ord("f") and save_frame is not None:
                metadata = {
                    "STACKCNT": live_stack.stack_count,
                    "MODE": "LiveStack" if stack_enabled else "LiveView",
                    "DATE": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "BITPIX": fmt["bits"],
                }
                fname = f"raw_live_stack_frame{frame_count}.fits"
                if stack_enabled and live_stack.stacked_raw16 is not None:
                    # raw16空間で加算したスタックをN枚平均してuint16 RGBで保存
                    n = max(1, live_stack.stack_count)
                    max_native = (1 << fmt["bits"]) - 1
                    avg16 = np.clip(live_stack.stacked_raw16 / n, 0, max_native).astype(np.uint16)
                    bgr16 = cv2.cvtColor(avg16, bayer_code)  # uint16 BGR
                    save_fits(bgr16, fname, metadata)
                    print(f"[save] FITS (uint16 stacked): {fname}")
                else:
                    save_fits(save_frame, fname, metadata)
            elif key == ord("a"):
                auto_stretch = not auto_stretch
                print(f"[stretch] {'ON' if auto_stretch else 'OFF'}")
            elif key == ord("h"):
                show_hist = not show_hist
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
            elif key == ord("d") and last_raw16 is not None:
                live_stack.set_dark_frame(last_raw16)
                n = len(live_stack.dark_buffer)
                print(f"[dark] ダークフレーム取得: {n}枚平均")
            elif key == ord("D"):
                live_stack.dark_frame = None
                live_stack.dark_buffer = []
                print("[dark] ダークフレームクリア")
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
            elif key == ord("+") or key == ord("="):
                skip += fmt["stride"]
                do_stack_reset()
                print(f"[skip] +{fmt['stride']} -> skip={skip}")
                buf.clear()
            elif key == ord("-"):
                skip = max(0, skip - fmt["stride"])
                do_stack_reset()
                print(f"[skip] -{fmt['stride']} -> skip={skip}")
                buf.clear()

            # --- 次フレーム受信 ---
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
        if _ser_recording and _ser_file is not None:
            import struct
            for ts in _ser_timestamps:
                _ser_file.write(struct.pack("<q", ts))
            _ser_file.seek(38)
            _ser_file.write(struct.pack("<I", _ser_frame_count))
            _ser_file.close()
            print(f"[SER] 終了につき録画保存: {_ser_fname}  {_ser_frame_count}フレーム")
        _stack_stop.set()
        _stack_event.set()   # wait() をすぐに解除してスレッドを終了させる
        _stack_thread.join(timeout=5.0)
        sock.close()
        cv2.destroyAllWindows()
        print("RAW Live Stack終了")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="RAW Live Stack (rpicam-raw TCP receiver + LiveStack)")
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
    parser.add_argument("--max-frames", type=int, default=100, help="LiveStack最大フレーム数 (デフォルト: 100)")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_raw_live_stack(args)


if __name__ == "__main__":
    main()
