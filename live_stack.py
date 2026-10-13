#!/usr/bin/env python3
"""
LiveStack機能付きカメラプレビューアプリケーション
- リアルタイムカメラプレビュー
- 位置合わせによるフレームスタッキング
- ノイズ軽減と画質向上
"""
import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

from hist_overlay import draw_hist_ccdf_overlay
from display_utils import clamp, decode_key, draw_info_lines, fit_display_frame, get_screen_size
from ui_menus import ControlMenu, ScreenRecorder, SettingsMenu, route_control_key, toggle_settings_menu
import cv2
import numpy as np
import time


def draw_ccdf_overlay(target_frame, source_frame, brightness_threshold=255, stop_ratio=0.10):
    return draw_hist_ccdf_overlay(
        target_frame,
        source_frame,
        brightness_threshold=brightness_threshold,
        stop_ratio=stop_ratio,
    )


def normalize_frame_to_uint8(frame):
    """入力フレームのビット幅を推定し、内部処理用に8bitへ正規化する。"""
    if frame is None:
        return frame, 8

    if frame.dtype == np.uint8:
        return frame, 8

    if np.issubdtype(frame.dtype, np.integer):
        observed_max = int(np.max(frame))
        if observed_max <= 0:
            # 画素が全て0のときはdtype幅を採用
            bit_depth = int(np.iinfo(frame.dtype).bits)
        else:
            bit_depth = max(8, int(np.ceil(np.log2(observed_max + 1))))

        max_value = float((1 << bit_depth) - 1)
        normalized = np.clip(frame.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)
        return normalized, bit_depth

    # floatなどは0..255レンジを想定してクリップ
    normalized = np.clip(frame, 0, 255).astype(np.uint8)
    return normalized, 8


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


def open_capture(source):
    """cv2.VideoCaptureを開く（ローカルカメラ時は低遅延化）"""
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_source_arg(source_arg):
    """source引数をint(カメラ番号)またはstr(URL)に変換"""
    if source_arg is None:
        return None
    stripped = source_arg.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped

class LiveStack:
    """LiveStack処理クラス"""
    
    def __init__(self, max_frames=100, verbose=False):  # max_framesを100に変更
        self.max_frames = max_frames  # max_framesをリングバッファサイズとして明確化
        self.verbose = verbose  # デバッグ出力制御
        self.overflow_ratio_threshold = 0.10  # 打ち切り比率（例: 0.10 = 10%）
        self.brightness_threshold = 255  # 打ち切り輝度しきい値（各チャンネル）
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
                    # ヒストグラム表示と同じ8bit空間で判定する
                    clipped = np.clip(self.stacked_image, 0, 255).astype(np.uint8)
                    metric = np.max(clipped[:, :, :3], axis=2)
                    white_pixels = np.sum(metric >= self.brightness_threshold)
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

    def process_stack(self):
        """スタック処理を実行"""
        latest_index = (self.buffer_index - 1 + self.max_frames) % self.max_frames  # 負の値を防ぐ
        if self.buffer[latest_index] is None:
            print("リングバッファが空です。スタック処理をスキップします。")
            return None

        # 最新のフレームを基準に設定
        latest_frame = self.buffer[latest_index]
        if self.dark_frame is not None:
            latest_frame = cv2.subtract(latest_frame, self.dark_frame)  # ダークフレームを引き算
            latest_frame = np.clip(latest_frame, 0, 255)  # 負の値をクリッピング
        display_frame = latest_frame.astype(np.float32)

        # 過去のフレームを遡りながら処理
        valid_stack_count = 1  # 最新フレームを含む
        for i in range(self.max_frames - 1):
            past_index = (self.buffer_index - 2 - i + self.max_frames) % self.max_frames  # 負の値を防ぐ
            past_frame = self.buffer[past_index]
            if past_frame is None:
                # ここより古いスロットも未投入なので探索終了
                break

            if self.dark_frame is not None:
                past_frame = cv2.subtract(past_frame, self.dark_frame)  # ダークフレームを引き算
                past_frame = np.clip(past_frame, 0, 255)  # 負の値をクリッピング

            # 位置ずれをmatchTemplate関数で求める
            gray_latest = cv2.cvtColor(latest_frame, cv2.COLOR_BGR2GRAY)
            gray_past = cv2.cvtColor(past_frame, cv2.COLOR_BGR2GRAY)

            # テンプレートと比較対象をワーク領域にコピー
            work_latest = cv2.convertScaleAbs(gray_latest, alpha=1.5, beta=20)  # 明るさ調整
            work_past = cv2.convertScaleAbs(gray_past, alpha=1.5, beta=20)  # 明るさ調整

            # 位置ずれをmatchTemplate関数で調べる
            result = cv2.matchTemplate(work_past, work_latest, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # 位置情報を使用して表示用フレームに加算
            offset_x, offset_y = max_loc
            M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
            aligned_past_frame = cv2.warpAffine(past_frame.astype(np.float32), M, (latest_frame.shape[1], latest_frame.shape[0]))

            # まず「このフレームを加算した結果」を評価し、
            # 閾値を超えるならこのフレームは加算しない
            test_frame = display_frame + aligned_past_frame

            # オーバーフロー条件のチェック（設定した比率・輝度しきい値）
            # ヒストグラム表示と同じ8bit空間で判定する
            clipped = np.clip(test_frame, 0, 255).astype(np.uint8)
            metric = np.max(clipped[:, :, :3], axis=2)
            white_pixels = np.sum(metric >= self.brightness_threshold)
            total_pixels = clipped.shape[0] * clipped.shape[1]
            overflow_ratio = white_pixels / total_pixels
            if overflow_ratio >= self.overflow_ratio_threshold:
                print(f"オーバーフロー条件に達しました (ratio={overflow_ratio:.3f} >= {self.overflow_ratio_threshold:.3f}, threshold={self.brightness_threshold})。スタック処理を終了します。")
                break

            # 閾値未満なら加算を採用
            display_frame = test_frame
            valid_stack_count += 1

        self.stack_count = valid_stack_count  # 有効なスタック数を更新

        # 加算値をそのまま8bit化すると櫛形が強く出るため、
        # 非整数スケール（sqrt(N)）で明るさを確保しつつ量子化間隔を緩和する。
        preview_scale = float(np.sqrt(max(1, valid_stack_count)))
        return np.clip(display_frame / preview_scale, 0, 255).astype(np.uint8)

    def set_dark_frame(self, frame):
        """ダークフレームを加算平均して設定"""
        if len(self.dark_buffer) >= self.max_frames:
            self.dark_buffer.pop(0)  # 古いフレームを削除
        self.dark_buffer.append(frame.astype(np.float32))

        # 加算平均を計算
        dark_sum = np.sum(self.dark_buffer, axis=0)
        self.dark_frame = (dark_sum / len(self.dark_buffer)).astype(np.uint8)
        print(f"ダークフレームを更新しました。現在の平均化フレーム数: {len(self.dark_buffer)}")

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
        header['NAXIS3'] = image.shape[0]  # 色数（例: RGBなら3）

    hdu.writeto(filename, overwrite=True)
    print(f"FITSファイル保存: {filename}")

def size_to_label(size):
    if size is None:
        return "N/A"
    return f"{size[0]}x{size[1]}"


def label_to_size(label):
    if not label or label == "N/A":
        return None
    try:
        w_str, h_str = label.split("x")
        return int(w_str), int(h_str)
    except Exception:
        return None


def create_camera_safely(camera_num, current_gain, current_exposure, low_light_mode=False, selected_size=None):
    """カメラを安全に作成する（エラーハンドリング付き）"""
    try:
        from camera_config import CameraConfig

        if selected_size is not None:
            picam2 = CameraConfig.create_camera_with_size(camera_num, size=selected_size, buffer_count=1)
        elif low_light_mode:
            picam2 = CameraConfig.create_low_light_camera(camera_num)
        else:
            picam2 = CameraConfig.create_fast_camera(camera_num)
        
        # 現在の設定を適用
        CameraConfig.apply_camera_settings(picam2, current_exposure, current_gain)
        return picam2
    except Exception as e:
        print(f"カメラ {camera_num} の作成に失敗: {e}")
        return None


STREAM_MENU_NAMES = {
    "Stop Threshold", "Stop Ratio(%)", "Max Frames", "Info Display", "Histogram", "Flip H", "Flip V", "Stack Mode",
}
CAMERA_MENU_NAMES = STREAM_MENU_NAMES | {"Camera", "Size", "Gain", "Exposure"}


def build_control_menu(has_camera):
    """操作メニュー([c])の項目。has_camera=True はpicamera2直結モード。"""
    items = [
        ("Save PNG", "s", None),
        ("Save JPEG", "j", None),
        ("Save FITS", "f", None),
        ("Screen Rec (AVI)", "y", "scr"),
        ("Stack Reset", "r", None),
    ]
    if has_camera:
        items += [("Dark Capture", "d", None), ("Switch Camera", "k", None)]
    return ControlMenu(items)


class LiveUi:
    """ストリームモードとpicamera2モードで共通のUI状態・描画・キー処理（raw_live_stack.pyと同じ操作感）"""

    def __init__(self, args, live_stack, tag, has_camera):
        self.live_stack = live_stack
        self.tag = tag  # 保存ファイル名に入れる識別子（例: source / cam0）
        self.settings_menu = SettingsMenu(names=CAMERA_MENU_NAMES if has_camera else STREAM_MENU_NAMES)
        self.control_menu = build_control_menu(has_camera)
        self.screen_rec = ScreenRecorder(args.screen_fps)
        self.stacking_enabled = False
        self.info_display = True
        self.show_hist = True
        self.flip_h = args.flip_h
        self.flip_v = args.flip_v
        self.screen_size = get_screen_size()
        self.camera_meta = None  # カメラ使用時のみ {"exposure_us", "gain", "dark"}

    def sync_settings_menu(self, **camera_values):
        """現在の状態を設定メニューへ反映する（camera/gain/exposure/size_label はキーワードで渡す）"""
        ls = self.live_stack
        self.settings_menu.set_current_values(
            max_frames=ls.max_frames,
            stack_mode=self.stacking_enabled,
            info_display=self.info_display,
            hist_display=self.show_hist,
            flip_h=self.flip_h,
            flip_v=self.flip_v,
            stop_ratio_percent=int(ls.overflow_ratio_threshold * 100),
            stop_threshold=ls.brightness_threshold,
            **camera_values,
        )

    def set_stacking(self, enabled):
        self.stacking_enabled = enabled
        self.settings_menu.set_value("Stack Mode", enabled)
        if enabled:
            print("LiveStack 有効")
            self.live_stack.reset()
        else:
            print("LiveStack 無効")

    def _flip(self, img):
        if self.flip_h and self.flip_v:
            return cv2.flip(img, -1)
        if self.flip_h:
            return cv2.flip(img, 1)
        if self.flip_v:
            return cv2.flip(img, 0)
        return img

    def build_frames(self, frame):
        """フレームをバッファへ追加し、(保存用, 表示用) フレームを返す（スタック・反転適用済み）"""
        self.live_stack.add_to_buffer(frame)
        if self.stacking_enabled:
            result = self.live_stack.process_stack()
            if result is None:
                result = frame
            save_frame = result.copy()
            display_frame = result.copy()
        else:
            save_frame = frame.copy()
            display_frame = frame.copy()
        return self._flip(save_frame), self._flip(display_frame)

    def render(self, display_frame, save_frame, source_line, extra_lines=()):
        """縮小→info・ヒストグラム・メニュー合成→画面録画 まで行いプレビューフレームを返す"""
        ls = self.live_stack
        preview_frame = fit_display_frame(display_frame, screen_size=self.screen_size, ratio=0.85, fallback_height=600)

        if self.info_display:
            mode_text = "Live Stack Mode" if self.stacking_enabled else "Live View Mode"
            cv2.putText(preview_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(preview_frame, source_line, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if self.stacking_enabled:
                cv2.putText(preview_frame, f"Frames: {ls.stack_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                current_ratio = compute_white_ratio_uint8(save_frame, ls.brightness_threshold)
                cv2.putText(preview_frame, f"Now: {current_ratio * 100:.2f}% / Limit: {ls.overflow_ratio_threshold * 100:.2f}%", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            for i, line in enumerate(extra_lines):
                cv2.putText(preview_frame, line, (10, 150 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.show_hist:
                preview_frame = draw_ccdf_overlay(
                    preview_frame,
                    save_frame,
                    brightness_threshold=ls.brightness_threshold,
                    stop_ratio=ls.overflow_ratio_threshold,
                )

        if self.settings_menu.menu_active:
            preview_frame = self.settings_menu.draw_menu(preview_frame)
        if self.control_menu.menu_active:
            preview_frame = self.control_menu.draw_menu(preview_frame, {"scr": self.screen_rec.active})

        self.screen_rec.write(preview_frame)
        self.screen_rec.draw_indicator(preview_frame)  # 録画フレームに写らないよう書き込み後に描画
        return preview_frame

    def apply_settings(self, values):
        """設定メニューで変更された共通項目を反映する"""
        ls = self.live_stack
        if values["max_frames"] != ls.max_frames:
            ls.max_frames = values["max_frames"]
            print(f"Max Frames設定: {ls.max_frames}")

        if values["stack_mode"] != self.stacking_enabled:
            self.set_stacking(values["stack_mode"])

        if values["info_display"] != self.info_display:
            self.info_display = values["info_display"]
            print(f"情報表示: {'ON' if self.info_display else 'OFF'}")

        if values["hist_display"] != self.show_hist:
            self.show_hist = values["hist_display"]
            print(f"[hist] ヒストグラム: {'ON' if self.show_hist else 'OFF'}")

        if values["flip_h"] != self.flip_h:
            self.flip_h = values["flip_h"]
            print(f"左右反転: {'ON' if self.flip_h else 'OFF'}")
        if values["flip_v"] != self.flip_v:
            self.flip_v = values["flip_v"]
            print(f"上下反転: {'ON' if self.flip_v else 'OFF'}")

        new_ratio = max(0.01, min(0.50, values["stop_ratio_percent"] / 100.0))
        new_threshold = max(5, min(255, values["stop_threshold"]))
        if abs(new_ratio - ls.overflow_ratio_threshold) > 1e-9 or new_threshold != ls.brightness_threshold:
            ls.overflow_ratio_threshold = new_ratio
            ls.brightness_threshold = new_threshold
            print(f"停止条件: Ratio={int(new_ratio * 100)}% Threshold>={new_threshold}")

    def save_image(self, kind, save_frame):
        """kind: "png" / "jpeg" / "fits"（文字情報なしのスタック済みフレームを保存）"""
        ls = self.live_stack
        cam = self.camera_meta
        stem = f"live_stack_{self.tag}_{int(time.time())}"
        mode = "LiveStack" if self.stacking_enabled else "LiveView"

        if kind == "fits":
            metadata = {}
            if cam:
                metadata.update({"EXPOSURE": cam["exposure_us"], "EXPTIME": "microseconds", "GAIN": cam["gain"]})
            metadata.update({"STACKCNT": ls.stack_count, "MODE": mode})
            if cam:
                metadata["DARKFRM"] = "YES" if cam["dark"] else "NO"
            metadata["DATE"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_fits(save_frame, f"{stem}.fits", metadata)
            return

        filename = f"{stem}.{'jpg' if kind == 'jpeg' else 'png'}"
        try:
            from PIL import Image

            pil_image = Image.fromarray(cv2.cvtColor(save_frame, cv2.COLOR_BGR2RGB))
            if kind == "jpeg":
                import piexif

                exif_0th = {
                    piexif.ImageIFD.DateTime: time.strftime("%Y:%m:%d %H:%M:%S"),
                    piexif.ImageIFD.ImageDescription: f"Stack Count: {ls.stack_count}",
                }
                if cam:
                    exif_0th[piexif.ImageIFD.ExposureTime] = (cam["exposure_us"], 1000000)
                pil_image.save(filename, "jpeg", exif=piexif.dump({"0th": exif_0th}))
            else:
                pil_image.save(filename, "png")
        except Exception:
            cv2.imwrite(filename, save_frame)
        print(f"{'JPEG' if kind == 'jpeg' else 'PNG'}保存: {filename}")
        if cam:
            print(f"保存データ: Gain={cam['gain']}, Exposure={self.settings_menu.get_exposure_text(cam['exposure_us'])}, Stack Count={ls.stack_count}")

    def handle_key(self, key, save_frame, preview_frame):
        """両モード共通のキーを処理する。処理したらTrueを返す。"""
        if key == ord("i"):
            self.info_display = not self.info_display
            self.settings_menu.set_value("Info Display", self.info_display)
            print(f"情報表示: {'ON' if self.info_display else 'OFF'}")
        elif key == ord("H"):
            self.show_hist = not self.show_hist
            self.settings_menu.set_value("Histogram", self.show_hist)
            print(f"[hist] ヒストグラム: {'ON' if self.show_hist else 'OFF'}")
        elif key == ord("h"):
            self.flip_h = not self.flip_h
            self.settings_menu.set_value("Flip H", self.flip_h)
            print(f"左右反転: {'ON' if self.flip_h else 'OFF'}")
        elif key == ord("v"):
            self.flip_v = not self.flip_v
            self.settings_menu.set_value("Flip V", self.flip_v)
            print(f"上下反転: {'ON' if self.flip_v else 'OFF'}")
        elif key == ord("t"):
            self.set_stacking(not self.stacking_enabled)
        elif key in (ord("r"), ord("R")):
            if self.stacking_enabled:
                self.live_stack.reset()
                print("スタックリセット - 次のフレームから再開")
        elif key == ord("y"):
            self.screen_rec.toggle(preview_frame)
        elif key in (ord("s"), ord("p")):
            self.save_image("png", save_frame)
        elif key == ord("j"):
            self.save_image("jpeg", save_frame)
        elif key == ord("f"):
            self.save_image("fits", save_frame)
        else:
            return False
        return True

    def route_menu_key(self, key, key_raw):
        """[c]/操作メニューのキー処理。(続行すべきか, 差し替え後のキー) を返す。"""
        consumed, action_key = route_control_key(key, key_raw, self.settings_menu, self.control_menu)
        if action_key is not None:
            return False, action_key
        return consumed, key

    def toggle_settings(self):
        opened = toggle_settings_menu(self.settings_menu, self.control_menu)
        print("設定メニューを開きました" if opened else "設定メニューを閉じました")
        return opened


def run_stream_mode(source, args):
    """URL/カメラ番号ソースでLiveStackを実行（picamera2不要）"""
    print(f"Live Stack - Source: {source}")
    print("操作: [q]終了 [m]設定メニュー [c]操作メニュー [s]PNG [j]JPEG [f]FITS [y]画面録画 [t]スタックON/OFF [r]リセット [i]情報表示 [h]左右反転 [v]上下反転")

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"ソースを開けませんでした: {source}")
        return

    live_stack = LiveStack(max_frames=args.max_frames)
    ui = LiveUi(args, live_stack, tag="source", has_camera=False)
    ui.sync_settings_menu()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレーム受信に失敗しました。")
                break

            frame, source_bit_depth = normalize_frame_to_uint8(frame)
            h, w = frame.shape[:2]

            save_frame, display_frame = ui.build_frames(frame)
            preview_frame = ui.render(display_frame, save_frame, f"Source: {source}  {w}x{h}  {source_bit_depth}bit")

            cv2.imshow("Live Stack", preview_frame)

            key_raw = cv2.waitKeyEx(1)
            key, nav = decode_key(key_raw)

            skip, key = ui.route_menu_key(key, key_raw)
            if skip:
                continue

            if ui.settings_menu.handle_key(key_raw):
                if ui.settings_menu.apply_requested:
                    ui.apply_settings(ui.settings_menu.get_current_values())
                continue

            if key == ord("q"):
                break
            elif key == ord("m"):
                if ui.toggle_settings():
                    ui.sync_settings_menu()
            else:
                ui.handle_key(key, save_frame, preview_frame)
    except KeyboardInterrupt:
        print("\n終了中...")
    finally:
        ui.screen_rec.stop("終了につき保存")
        cap.release()
        cv2.destroyAllWindows()
        print("Live Stack終了")


def main():
    parser = argparse.ArgumentParser(description="LiveStack機能付きカメラプレビュー")
    parser.add_argument("-n", "--max-frames", type=int, default=100,
                        help="最大スタックフレーム数 (デフォルト: 100)")
    parser.add_argument("--source", type=str, default=None,
                        help="映像ソース。例: 0 または tcp://192.168.1.17:8888")
    parser.add_argument("--flip-h", action="store_true",
                        help="左右反転して起動")
    parser.add_argument("--flip-v", action="store_true",
                        help="上下反転して起動")
    parser.add_argument("--screen-fps", type=float, default=15.0,
                        help="画面録画([y]キー/操作メニュー)のフレームレート (デフォルト: 15)")
    args = parser.parse_args()

    parsed_source = parse_source_arg(args.source)
    if parsed_source is not None:
        run_stream_mode(parsed_source, args)
        return

    # source未指定時は従来のpicamera2経路を試す。利用不可ならWindows向けにUSBカメラ(0)へフォールバック。
    try:
        from camera_config import CameraConfig
    except Exception as e:
        print(f"picamera2経路を利用できません ({e})")
        print("USBカメラ(0)へフォールバックします。必要に応じて --source で明示指定してください。")
        run_stream_mode(0, args)
        return

    print("Live Stack - LiveStack機能付きカメラプレビュー（カメラ切り替え機能付き）")
    print("操作:")
    print("  [q] 終了")
    print("  [m] 設定メニュー  [c] 操作メニュー（保存・画面録画・リセット・ダーク・カメラ切替）")
    print("  [i] 情報表示ON/OFF")
    print("  [k] カメラ切り替え")
    print("  [s] PNG保存  [j] JPEG保存  [f] FITS保存")
    print("  [y] 画面録画AVI開始/停止")
    print("  [t] LiveStack ON/OFF")
    print("  [r] スタックリセット")
    print("  [h] 左右反転トグル")
    print("  [v] 上下反転トグル")
    print("  [d] ダークフレーム取得")
    print("従来のキー操作:")
    print("  [+/-] ゲイン調整  [0-9] シャッター速度")

    # カメラ初期化
    current_camera = 0
    low_light_mode = False
    current_gain = 2.0
    current_exposure = 16667  # 1/60秒

    available_sizes = CameraConfig.get_available_sizes(current_camera)
    size_labels = [size_to_label(size) for size in available_sizes]
    current_size_label = size_labels[0] if size_labels else "N/A"

    picam2 = create_camera_safely(current_camera, current_gain, current_exposure, low_light_mode, label_to_size(current_size_label))

    if picam2 is None:
        print("カメラの初期化に失敗しました。")
        return

    print(f"カメラ {current_camera} を使用中")

    # LiveStack初期化
    live_stack = LiveStack(max_frames=args.max_frames)
    print(f"最大スタックフレーム数: {args.max_frames}")

    ui = LiveUi(args, live_stack, tag=f"cam{current_camera}", has_camera=True)
    settings_menu = ui.settings_menu
    if ui.flip_h or ui.flip_v:
        print(f"フリップ: {'H' if ui.flip_h else ''}{' V' if ui.flip_v else ''}".strip())
    dark_frame_set = False  # ダークフレーム取得状態を管理

    settings_menu.set_size_choices(size_labels, current_size_label)
    ui.sync_settings_menu(camera=current_camera, gain=current_gain, exposure=current_exposure, size_label=current_size_label)

    def reset_dark_and_buffer():
        """ダークフレームとリングバッファをリセット"""
        nonlocal dark_frame_set
        try:
            live_stack.dark_frame = None
            live_stack.dark_buffer = []
            dark_frame_set = False
            print("ダークフレームとリングバッファをリセットしました。")
        except Exception as e:
            print(f"リセット中にエラーが発生しました: {e}")

    try:
        picam2.start()
        time.sleep(1)

        while True:
            # フレーム取得
            frame = picam2.capture_array()
            frame, source_bit_depth = normalize_frame_to_uint8(frame)
            h, w = frame.shape[:2]

            save_frame, display_frame = ui.build_frames(frame)
            ui.tag = f"cam{current_camera}"
            ui.camera_meta = {"exposure_us": current_exposure, "gain": current_gain, "dark": dark_frame_set}

            camera_info = f"Cam{current_camera} Gain:{current_gain} Exp:{settings_menu.get_exposure_text(current_exposure)} Size:{w}x{h} {source_bit_depth}bit"
            preview_frame = ui.render(display_frame, save_frame, camera_info,
                                      extra_lines=["Dark Frame Set"] if dark_frame_set else ())

            # プレビュー表示
            cv2.imshow("Live Stack", preview_frame)

            # キー入力処理
            key_raw = cv2.waitKeyEx(1)
            key, nav = decode_key(key_raw)

            skip, key = ui.route_menu_key(key, key_raw)
            if skip:
                continue

            # 設定メニューのキー処理（優先）
            if settings_menu.handle_key(key_raw):
                if not settings_menu.apply_requested:
                    continue

                # 設定が変更された場合の処理
                values = settings_menu.get_current_values()

                # カメラ切り替えの処理
                if values["camera"] != current_camera:
                    new_camera = values["camera"]
                    print(f"設定メニューからカメラ {new_camera} に切り替え中...")

                    # 現在のカメラを停止
                    picam2.stop()
                    picam2.close()

                    # 新しいカメラを作成
                    new_available_sizes = CameraConfig.get_available_sizes(new_camera)
                    new_size_labels = [size_to_label(size) for size in new_available_sizes]
                    requested_size_label = values["size_label"]
                    target_size_label = requested_size_label if requested_size_label in new_size_labels else (new_size_labels[0] if new_size_labels else "N/A")

                    new_picam2 = create_camera_safely(new_camera, current_gain, current_exposure, low_light_mode, label_to_size(target_size_label))

                    if new_picam2 is not None:
                        # 切り替え成功
                        picam2 = new_picam2
                        current_camera = new_camera
                        picam2.start()
                        time.sleep(1)
                        print(f"カメラ {current_camera} に切り替えました")

                        # カメラ切り替え時にバッファとスタックをリセット
                        live_stack.reset()
                        live_stack.dark_frame = None
                        live_stack.dark_buffer = []
                        dark_frame_set = False
                        print("設定メニューでのカメラ切り替えに伴いバッファとスタックをリセットしました")

                        # メニューの表示を更新
                        settings_menu.set_value("Camera", current_camera)
                        settings_menu.set_size_choices(new_size_labels, target_size_label)
                        current_size_label = target_size_label
                    else:
                        # 切り替え失敗、元のカメラに戻す
                        print(f"カメラ {new_camera} への切り替えに失敗。元のカメラに戻します。")
                        picam2 = create_camera_safely(current_camera, current_gain, current_exposure, low_light_mode, label_to_size(current_size_label))
                        if picam2 is not None:
                            picam2.start()
                            time.sleep(1)
                            print(f"カメラ {current_camera} に復帰しました")
                            # メニューの値も元に戻す
                            settings_menu.set_value("Camera", current_camera)
                        else:
                            print("カメラの復帰に失敗しました。")
                            break

                # カメラ設定を適用
                if values["gain"] != current_gain or values["exposure"] != current_exposure:
                    current_gain = values["gain"]
                    current_exposure = values["exposure"]
                    CameraConfig.apply_camera_settings(picam2, current_exposure, current_gain)
                    print(f"設定適用: Gain={current_gain}, Exposure={settings_menu.get_exposure_text(current_exposure)}")
                    reset_dark_and_buffer()

                # Size設定を適用
                if values["size_label"] != current_size_label:
                    previous_size_label = current_size_label
                    current_size_label = values["size_label"]
                    print(f"サイズ: {current_size_label}")

                    picam2.stop()
                    picam2.close()
                    picam2 = create_camera_safely(current_camera, current_gain, current_exposure, low_light_mode, label_to_size(current_size_label))
                    if picam2 is not None:
                        picam2.start()
                        time.sleep(1)
                        live_stack.reset()
                        live_stack.dark_frame = None
                        live_stack.dark_buffer = []
                        dark_frame_set = False
                        print("サイズ変更に伴いカメラを再初期化し、バッファとスタックをリセットしました")
                    else:
                        print("サイズ変更後のカメラ初期化に失敗。前の設定で復旧を試みます。")
                        current_size_label = previous_size_label
                        settings_menu.set_value("Size", current_size_label)
                        picam2 = create_camera_safely(current_camera, current_gain, current_exposure, low_light_mode, label_to_size(current_size_label))
                        if picam2 is not None:
                            picam2.start()
                            time.sleep(1)
                            print("前のサイズ設定で復旧しました。")
                        else:
                            print("カメラの復旧にも失敗しました。")
                            break

                # Max Frames / Stack Mode / 表示 / 反転 / 停止条件 は両モード共通
                ui.apply_settings(values)
                continue

            # 既存のキー処理
            if key == ord("q"):
                break
            elif key == ord("m"):  # 設定メニュー表示
                if ui.toggle_settings():
                    # 現在の設定値をメニューに反映
                    ui.sync_settings_menu(camera=current_camera, gain=current_gain, exposure=current_exposure, size_label=current_size_label)
            elif key == ord("k"):  # カメラ切り替え
                print("カメラを切り替え中...")

                # 現在のカメラを停止
                picam2.stop()
                picam2.close()

                # 次のカメラ番号を決定（0と1の間で切り替え）
                next_camera = 1 - current_camera

                # 新しいカメラを作成
                next_available_sizes = CameraConfig.get_available_sizes(next_camera)
                next_size_labels = [size_to_label(size) for size in next_available_sizes]
                target_size_label = current_size_label if current_size_label in next_size_labels else (next_size_labels[0] if next_size_labels else "N/A")

                new_picam2 = create_camera_safely(next_camera, current_gain, current_exposure, low_light_mode, label_to_size(target_size_label))

                if new_picam2 is not None:
                    # 切り替え成功
                    picam2 = new_picam2
                    current_camera = next_camera
                    picam2.start()
                    time.sleep(1)
                    print(f"カメラ {current_camera} に切り替えました")

                    # カメラ切り替え時にバッファとスタックをリセット
                    live_stack.reset()
                    live_stack.dark_frame = None
                    live_stack.dark_buffer = []
                    dark_frame_set = False
                    print("カメラ切り替えに伴いバッファとスタックをリセットしました")

                    # メニューの表示も同期
                    settings_menu.set_value("Camera", current_camera)
                    settings_menu.set_size_choices(next_size_labels, target_size_label)
                    current_size_label = target_size_label
                else:
                    # 切り替え失敗、元のカメラに戻す
                    print(f"カメラ {next_camera} への切り替えに失敗。元のカメラに戻します。")
                    picam2 = create_camera_safely(current_camera, current_gain, current_exposure, low_light_mode, label_to_size(current_size_label))
                    if picam2 is not None:
                        picam2.start()
                        time.sleep(1)
                        print(f"カメラ {current_camera} に復帰しました")
                    else:
                        print("カメラの復帰に失敗しました。")
                        break
            elif key == ord("d"):
                if not ui.stacking_enabled:
                    live_stack.set_dark_frame(frame)
                    dark_frame_set = True
                else:
                    print("ダークフレームはLiveViewモードでのみ取得可能です。")
            elif key in [ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6"), ord("7"), ord("8"), ord("9"), ord("0")]:
                # シャッター速度変更
                exposure_times = {
                    ord("1"): 1000000,
                    ord("2"): 500000,
                    ord("3"): 250000,
                    ord("4"): 125000,
                    ord("5"): 62500,
                    ord("6"): 33333,
                    ord("7"): 16667,
                    ord("8"): 8000,
                    ord("9"): 4000,
                    ord("0"): 2000
                }
                current_exposure = exposure_times[key]
                settings_menu.set_value("Exposure", current_exposure)  # メニューも同期
                CameraConfig.apply_camera_settings(picam2, current_exposure, current_gain)
                print(f"露出: {settings_menu.get_exposure_text(current_exposure)}")
                reset_dark_and_buffer()

            elif key in [ord("+"), ord("-")]:
                # ゲイン変更
                if key == ord("+") or key == ord("="):
                    current_gain = min(8.0, current_gain + 0.5)
                elif key == ord("-"):
                    current_gain = max(1.0, current_gain - 0.5)
                settings_menu.set_value("Gain", current_gain)  # メニューも同期
                CameraConfig.apply_camera_settings(picam2, current_exposure, current_gain)
                print(f"ゲイン: {current_gain}")
                reset_dark_and_buffer()
            else:
                ui.handle_key(key, save_frame, preview_frame)

    except KeyboardInterrupt:
        print("\n終了中...")
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        ui.screen_rec.stop("終了につき保存")
        if picam2 is not None:
            picam2.stop()
            picam2.close()
        cv2.destroyAllWindows()
        print("Live Stack終了")

if __name__ == "__main__":
    main()
