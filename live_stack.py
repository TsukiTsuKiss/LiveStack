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
import re
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

from hist_overlay import draw_hist_ccdf_overlay
from display_utils import clamp, decode_key, draw_info_lines, fit_display_frame, get_screen_size
from focus_utils import draw_focus_panel, new_focus_state, reset_focus_state, update_focus_state
from zoom_utils import draw_zoom_pip
from ui_menus import ControlMenu, ScreenRecorder, SettingsMenu, route_control_key, toggle_settings_menu
from config_loader import load_config, ssh_start_rpicam, ssh_kill_rpicam
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


ROI_MIN_SIZE = 16         # ROIとして受け付ける最小の辺(px)
ROI_ALIGN_MIN_SIZE = 256  # 位置合わせ窓の最小の辺(ROIがこれより小さいときはROIの中心で広げる)


def normalize_roi(x0, y0, x1, y1, width, height):
    """ROIをフレーム内に収める。小さすぎるときはNone。"""
    x0, x1 = sorted((int(x0), int(x1)))
    y0, y1 = sorted((int(y0), int(y1)))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < ROI_MIN_SIZE or y1 - y0 < ROI_MIN_SIZE:
        return None
    return (x0, y0, x1, y1)


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


def parse_tcp_port(source, default=8888):
    """tcp://host:port からportを取り出す（失敗時はdefault）"""
    try:
        return int(source.rsplit(":", 1)[1])
    except (ValueError, IndexError):
        return default


def parse_rpicam_cmd_port(cmd, default=None):
    """rpicam-vidコマンド文字列の -o tcp://.../PORT からportを取り出す（見つからなければdefault）"""
    if not cmd:
        return default
    m = re.search(r"-o\s+tcp://[^:\s]+:(\d+)", cmd)
    return int(m.group(1)) if m else default

class LiveStack:
    """LiveStack処理クラス"""
    
    def __init__(self, max_frames=100, verbose=False):  # max_framesを100に変更
        self.max_frames = max_frames  # max_framesをリングバッファサイズとして明確化
        self.verbose = verbose  # デバッグ出力制御
        self.overflow_ratio_threshold = 0.10  # 打ち切り比率（例: 0.10 = 10%）
        self.brightness_threshold = 255  # 打ち切り輝度しきい値（各チャンネル）
        self.stats_frame = None  # 停止判定と同じ加算値(255でクリップ)。Now/ヒストグラム用
        self.roi = None  # (x0,y0,x1,y1) フレームpx。停止判定・ヒストグラム・位置合わせの対象領域（None=全体/中央窓）
        self.align_ratio = 0.5  # ROIなしのときの位置合わせ窓の大きさ（フレーム幅/高さに対する比率）
        self.align_min_response = 0.05  # phaseCorrelateの応答がこれ未満なら検出失敗
        self.align_max_jump = 10.0  # 予測位置からこれ以上外れた検出は異常値(px)
        self._hann = {}
        self.dark_frame = None  # d キーで取得するまでは None（サイズを動的に合わせるため）
        self.dark_buffer = []  # ダークフレーム用リングバッファ
        self.reset()

    def clear_buffer(self):
        """リングバッファとずれ検出の状態を破棄する。"""
        self.buffer = [None] * self.max_frames
        self.buffer_pos = [None] * self.max_frames  # 各フレームの基準座標系での位置(x,y)・float
        self.buffer_index = 0
        self._ref_img = None  # ずれ検出の基準(キーフレーム)画像
        self._ref_pos = (0.0, 0.0)
        self._last_pos = (0.0, 0.0)
        self._last_vel = (0.0, 0.0)  # 検出失敗時の外挿用
        self._reject_count = 0

    def set_max_frames(self, new_max_frames):
        """最新フレームを残してリングバッファの大きさを変える。"""
        new_max_frames = max(1, int(new_max_frames))
        if new_max_frames == self.max_frames:
            return
        old_len = self.max_frames
        ordered = []
        for i in range(old_len):
            idx = (self.buffer_index + i) % old_len
            if self.buffer[idx] is not None:
                ordered.append((self.buffer[idx], self.buffer_pos[idx]))
        ordered = ordered[-new_max_frames:]
        pad = new_max_frames - len(ordered)
        self.max_frames = new_max_frames
        self.buffer = [f for f, _ in ordered] + [None] * pad
        self.buffer_pos = [p for _, p in ordered] + [None] * pad
        self.buffer_index = len(ordered) % new_max_frames

    def reset(self):
        """スタック状態をリセット（リングバッファも破棄して次のフレームから再開する）"""
        self.clear_buffer()
        self.stats_frame = None
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

    def roi_text(self):
        x0, y0, x1, y1 = self.roi
        return f"x={x0} y={y0} {x1 - x0}x{y1 - y0}"

    def roi_view(self, arr):
        """ROIがあればその範囲(ビュー)、なければ(またはサイズが合わなければ)全体を返す。"""
        roi = self.roi
        if roi is None or roi[2] > arr.shape[1] or roi[3] > arr.shape[0]:
            return arr
        return arr[roi[1]:roi[3], roi[0]:roi[2]]

    def align_window(self, w, h):
        """位置合わせ窓(x0,y0,x1,y1)。ROIがあるときROIの中心を保ち、辺が小さければ ROI_ALIGN_MIN_SIZE まで広げる。
        ROIなしは中央の align_ratio 窓。"""
        roi = self.roi
        if roi is None or roi[2] > w or roi[3] > h:
            sw = min(w, max(64, int(w * self.align_ratio)))
            sh = min(h, max(64, int(h * self.align_ratio)))
            x0, y0 = (w - sw) // 2, (h - sh) // 2
            return (x0, y0, x0 + sw, y0 + sh)
        out = []
        for lo, hi, size in ((roi[0], roi[2], w), (roi[1], roi[3], h)):
            length = min(max(hi - lo, ROI_ALIGN_MIN_SIZE), size)
            start = max(0, min((lo + hi - length) // 2, size - length))
            out += [start, start + length]
        return (out[0], out[2], out[1], out[3])

    def _prepare_align_image(self, frame):
        """位置合わせ窓をダーク減算→グレー化→背景除去して、ずれ検出用画像を作る。"""
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.align_window(w, h)
        crop = frame[y0:y1, x0:x1]
        dark = self.dark_frame
        if dark is not None and dark.shape == frame.shape:
            crop = cv2.subtract(crop, dark[y0:y1, x0:x1])
        gray = crop if crop.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(crop[:, :, :3]), cv2.COLOR_BGR2GRAY)
        # 2x2に縮小して計算量を減らす（検出したずれは2倍してpxに戻す）
        gray = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        return gray - cv2.GaussianBlur(gray, (0, 0), 16)

    def _detect_shift(self, prev, cur):
        """prev→curの移動量(dx,dy)をフレームpxのfloat(サブピクセル)で返す。失敗時はNone。"""
        hh, ww = cur.shape
        win = self._hann.get((ww, hh))
        if win is None:
            win = self._hann[(ww, hh)] = cv2.createHanningWindow((ww, hh), cv2.CV_32F)
        # phaseCorrelateは入力に窓関数を書き込むため、基準画像を保つにはコピーを渡す
        (dx, dy), response = cv2.phaseCorrelate(prev.copy(), cur.copy(), win)
        if response < self.align_min_response:
            return None
        return dx * 2.0, dy * 2.0

    def add_to_buffer(self, frame):
        """リングバッファにフレームを追加し、基準フレームに対する位置を記録する"""
        prev = self.buffer[(self.buffer_index - 1) % self.max_frames]
        if prev is not None and prev.shape != frame.shape:
            self.clear_buffer()  # サイズが変わったら過去フレームは使えない
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
                print("[align] ずれ検出失敗/異常値: 前回の移動量で外挿します")
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

        self.buffer[self.buffer_index] = frame
        self.buffer_pos[self.buffer_index] = pos
        self.buffer_index = (self.buffer_index + 1) % self.max_frames

    def _overflow_ratio(self, stack):
        """ROI(なければ全体)で max(B,G,R) がしきい値以上の画素の比率。広いときは約10万画素に間引いて計算量を抑える。"""
        view = self.roi_view(stack)
        step = max(1, int((view.shape[0] * view.shape[1] / 100000) ** 0.5))
        if step > 1:
            view = view[::step, ::step]
        metric = view
        if view.ndim == 3:
            metric = view[:, :, 0]
            for c in range(1, min(3, view.shape[2])):
                metric = cv2.max(metric, view[:, :, c])
        return np.count_nonzero(metric >= self.brightness_threshold) / max(1, metric.size)

    def process_stack(self):
        """スタック処理を実行（最新フレームに位置を合わせて過去フレームを加算。停止判定はROI）"""
        latest_index = (self.buffer_index - 1 + self.max_frames) % self.max_frames  # 負の値を防ぐ
        latest_frame = self.buffer[latest_index]
        if latest_frame is None:
            print("リングバッファが空です。スタック処理をスキップします。")
            return None

        dark = self.dark_frame
        if dark is not None and dark.shape != latest_frame.shape:
            dark = None
        if dark is not None:
            latest_frame = cv2.subtract(latest_frame, dark)  # ダークフレームを引き算
        display_frame = latest_frame.astype(np.uint16)  # 255×枚数が収まる16bitで加算する
        scratch = np.empty_like(display_frame)
        latest_pos = self.buffer_pos[latest_index] or (0.0, 0.0)
        h, w = latest_frame.shape[:2]

        # 過去のフレームを遡りながら処理
        valid_stack_count = 1  # 最新フレームを含む
        for i in range(self.max_frames - 1):
            past_index = (self.buffer_index - 2 - i + self.max_frames) % self.max_frames  # 負の値を防ぐ
            past_frame = self.buffer[past_index]
            if past_frame is None or past_frame.shape != latest_frame.shape:
                # ここより古いスロットも未投入なので探索終了
                break

            if dark is not None:
                past_frame = cv2.subtract(past_frame, dark)  # ダークフレームを引き算
            aligned_past_frame = past_frame

            # add_to_bufferで記録した位置の差だけ動かして、最新フレームの座標系に合わせる
            past_pos = self.buffer_pos[past_index] or latest_pos
            dx, dy = latest_pos[0] - past_pos[0], latest_pos[1] - past_pos[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                aligned_past_frame = cv2.warpAffine(past_frame, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))

            # まず「このフレームを加算した結果」を評価し、
            # 閾値を超えるならこのフレームは加算しない
            test_frame = cv2.add(display_frame, aligned_past_frame, dst=scratch, dtype=cv2.CV_16U)

            # オーバーフロー条件のチェック（設定した比率・輝度しきい値。ROIがあればROI内）
            overflow_ratio = self._overflow_ratio(test_frame)
            if overflow_ratio >= self.overflow_ratio_threshold:
                print(f"オーバーフロー条件に達しました (ratio={overflow_ratio:.3f} >= {self.overflow_ratio_threshold:.3f}, threshold={self.brightness_threshold})。スタック処理を終了します。")
                break

            # 閾値未満なら加算を採用
            display_frame, scratch = test_frame, display_frame
            valid_stack_count += 1

        self.stack_count = valid_stack_count  # 有効なスタック数を更新
        self.stats_frame = np.minimum(display_frame, 255).astype(np.uint8)

        # 加算値をそのまま8bit化すると櫛形が強く出るため、
        # 非整数スケール（sqrt(N)）で明るさを確保しつつ量子化間隔を緩和する。
        preview_scale = float(np.sqrt(max(1, valid_stack_count)))
        return cv2.convertScaleAbs(display_frame, alpha=1.0 / preview_scale)

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


def build_control_menu(has_camera, roi_key):
    """操作メニュー([c])の項目。has_camera=True はpicamera2直結モード。"""
    items = [
        ("Save PNG", "s", None),
        ("Save JPEG", "j", None),
        ("Save FITS", "f", None),
        ("Screen Rec (AVI)", "y", "scr"),
        ("Zoom", "z", "zoom"),
        ("Focus Aid", "F", "focus"),
        ("ROI (select/clear)", roi_key, "roi"),
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
        self.roi_key = "o" if has_camera else "k"  # picamera2直結では k がカメラ切り替えのため o
        self.roi_ui = {"select": False, "start": None, "current": None, "dragging": False}  # 座標はプレビューpx
        self._pending_roi = getattr(args, "roi", None)  # --roi は最初のフレームでサイズを見て検証する
        self.settings_menu = SettingsMenu(names=CAMERA_MENU_NAMES if has_camera else STREAM_MENU_NAMES)
        self.control_menu = build_control_menu(has_camera, self.roi_key)
        self.screen_rec = ScreenRecorder(args.screen_fps)
        self.stacking_enabled = False
        self.info_display = True
        self.show_hist = True
        self.flip_h = args.flip_h
        self.flip_v = args.flip_v
        self.screen_size = get_screen_size()
        self.camera_meta = None  # カメラ使用時のみ {"exposure_us", "gain", "dark"}
        # フォーカスアシスト: 最新の単フレーム(スタック・反転前)の領域を測る。中心は表示(反転後)での割合
        self.focus = new_focus_state()
        self.focus_center = (0.5, 0.5)
        self._last_frame = None
        self._frame_id = 0
        self._gray = None
        self._gray_id = -1
        self._preview_size = (0, 0)
        # ズームPiP: 切り出し範囲は表示(反転後)での割合(fx0, fy0, fx1, fy1)
        self.zoom_on = False
        self.zoom_rect = None
        self._zoom_drag = {"start": None, "cur": None}
        self._zoom_region = None  # 直近に描画した切り出し範囲(表示px)。フォーカス測定に使う
        self.zoom_pip_size = getattr(args, "zoom_pip_size", 640)
        self.zoom_click_size = getattr(args, "zoom_click_size", 128)
        self.zoom_corner = getattr(args, "zoom_corner", "br")

    def attach_mouse(self, window_name):
        """Zoom ON中はクリック/ドラッグで切り出し範囲を選び、Focus Aidのみ ON中は左クリックで測定枠を移動する。"""
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _click_rect(self, fx, fy):
        """クリック位置を中心に zoom_click_size 画素四方となる範囲(表示の割合)。"""
        if self._last_frame is None:
            return None
        h, w = self._last_frame.shape[:2]
        hw, hh = min(0.5, self.zoom_click_size / 2 / w), min(0.5, self.zoom_click_size / 2 / h)
        x0 = min(max(fx - hw, 0.0), 1.0 - 2 * hw)
        y0 = min(max(fy - hh, 0.0), 1.0 - 2 * hh)
        return (x0, y0, x0 + 2 * hw, y0 + 2 * hh)

    def _preview_to_frame(self, x, y):
        """プレビューpxをフレームpxへ変換する（反転を戻す）。"""
        pw, ph = self._preview_size
        h, w = self._last_frame.shape[:2]
        fx, fy = x / pw, y / ph
        if self.flip_h:
            fx = 1.0 - fx
        if self.flip_v:
            fy = 1.0 - fy
        return fx * w, fy * h

    def set_roi(self, roi):
        ls = self.live_stack
        ls.roi = roi
        if self.stacking_enabled:
            ls.reset()
        if roi is None:
            print("[roi] クリア（全体で判定・中央窓で位置合わせ）")
        else:
            print(f"[roi] 設定: {ls.roi_text()}")
            if min(roi[2] - roi[0], roi[3] - roi[1]) < ROI_ALIGN_MIN_SIZE:
                print(f"[roi] 辺が{ROI_ALIGN_MIN_SIZE}px未満のため、位置合わせ窓はROIの中心で{ROI_ALIGN_MIN_SIZE}pxまで広げます（停止判定・ヒストグラムはROIそのもの）")

    def toggle_roi(self):
        """ROIをクリア、選択中なら取消、どちらでもなければドラッグ選択モードにする。"""
        ru = self.roi_ui
        if ru["select"]:
            ru.update(select=False, dragging=False, start=None, current=None)
            print("[roi] 選択を取消")
        elif self.live_stack.roi is not None:
            self.set_roi(None)
        else:
            ru["select"] = True
            print(f"[roi] プレビュー上でドラッグして範囲を指定してください（[{self.roi_key}]で取消）")

    def _on_roi_mouse(self, event, x, y):
        ru = self.roi_ui
        if event == cv2.EVENT_LBUTTONDOWN:
            ru.update(start=(x, y), current=(x, y), dragging=True)
        elif event == cv2.EVENT_MOUSEMOVE and ru["dragging"]:
            ru["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and ru["dragging"]:
            x0, y0 = ru["start"]
            ru.update(dragging=False, select=False, start=None, current=None)
            if abs(x - x0) < 8 and abs(y - y0) < 8:
                print("[roi] 選択を取消（ドラッグで範囲を指定してください）")
                return
            ax, ay = self._preview_to_frame(x0, y0)
            bx, by = self._preview_to_frame(x, y)
            h, w = self._last_frame.shape[:2]
            roi = normalize_roi(ax, ay, bx, by, w, h)
            if roi is None:
                print(f"[roi] 小さすぎます（{ROI_MIN_SIZE}px以上）。もう一度 [{self.roi_key}] で選択してください")
            else:
                self.set_roi(roi)

    def _on_mouse(self, event, x, y, flags, param):
        pw, ph = self._preview_size
        if pw <= 0 or ph <= 0:
            return
        if self.roi_ui["select"] and self._last_frame is not None:
            self._on_roi_mouse(event, x, y)
            return
        fx, fy = min(1.0, max(0.0, x / pw)), min(1.0, max(0.0, y / ph))
        if self.zoom_on:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._zoom_drag = {"start": (x, y), "cur": (x, y)}
            elif event == cv2.EVENT_MOUSEMOVE and self._zoom_drag["start"]:
                self._zoom_drag["cur"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self._zoom_drag["start"]:
                x0, y0 = self._zoom_drag["start"]
                self._zoom_drag = {"start": None, "cur": None}
                if abs(x - x0) < 8 and abs(y - y0) < 8:
                    self.zoom_rect = self._click_rect(fx, fy)
                else:
                    self.zoom_rect = (min(x0, x) / pw, min(y0, y) / ph, max(x0, x) / pw, max(y0, y) / ph)
        elif event == cv2.EVENT_LBUTTONDOWN and self.focus["on"]:
            self.focus_center = (fx, fy)

    def _draw_zoom(self, preview_frame, display_frame):
        """選択範囲の枠とPiPを描き、PiPの(ox, oy, 幅, 高さ)を返す。範囲未選択ならNone。"""
        d = self._zoom_drag
        if d["start"] and d["cur"]:
            cv2.rectangle(preview_frame, d["start"], d["cur"], (0, 255, 255), 1)
        if self.zoom_rect is None:
            return None
        ph, pw = preview_frame.shape[:2]
        dh, dw = display_frame.shape[:2]
        fx0, fy0, fx1, fy1 = self.zoom_rect
        x0, y0, x1, y1 = round(fx0 * dw), round(fy0 * dh), round(fx1 * dw), round(fy1 * dh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        self._zoom_region = (x0, y0, x1, y1)
        cv2.rectangle(preview_frame, (round(fx0 * pw), round(fy0 * ph)), (round(fx1 * pw), round(fy1 * ph)), (0, 255, 255), 1)
        return draw_zoom_pip(preview_frame, display_frame[y0:y1, x0:x1], self.zoom_pip_size, self.zoom_corner)

    def _draw_focus(self, preview_frame, pip_pos=None):
        frame = self._last_frame
        if frame is None:
            return
        h, w = frame.shape[:2]
        if self._zoom_region is not None:
            zx0, zy0, zx1, zy1 = self._zoom_region
            ux0, ux1 = (w - zx1, w - zx0) if self.flip_h else (zx0, zx1)
            uy0, uy1 = (h - zy1, h - zy0) if self.flip_v else (zy0, zy1)
            region = (ux0, uy0, ux1, uy1)
        else:
            side = max(48, min(h, w) // 5)
            fx, fy = self.focus_center
            fx_u = 1.0 - fx if self.flip_h else fx
            fy_u = 1.0 - fy if self.flip_v else fy
            x0 = int(min(max(fx_u * w - side / 2, 0), w - side))
            y0 = int(min(max(fy_u * h - side / 2, 0), h - side))
            region = (x0, y0, x0 + side, y0 + side)
        if self._gray_id != self._frame_id:
            gray_src = frame if frame.ndim == 2 else np.ascontiguousarray(frame[:, :, :3])
            self._gray = gray_src if frame.ndim == 2 else cv2.cvtColor(gray_src, cv2.COLOR_BGR2GRAY)
            self._gray_id = self._frame_id
        update_focus_state(self.focus, self._gray, region, 255, self._frame_id, bin2=False)
        ph, pw = preview_frame.shape[:2]
        if self._zoom_region is None:
            sx, sy = pw / w, ph / h
            rx0, rx1 = (w - region[2], w - region[0]) if self.flip_h else (region[0], region[2])
            ry0, ry1 = (h - region[3], h - region[1]) if self.flip_v else (region[1], region[3])
            cv2.rectangle(preview_frame, (round(rx0 * sx), round(ry0 * sy)), (round(rx1 * sx), round(ry1 * sy)), (0, 255, 255), 1)
        if not self.focus["hist"]:
            return
        if pip_pos is not None:
            # PiPを覆わないよう、下隔なら真上・上隔なら真下に同じ幅で置く
            px, py, pw_p, ph_p = pip_pos
            panel_y = py - 84 - 6 if self.zoom_corner in ("bl", "br") else py + ph_p + 6
            draw_focus_panel(preview_frame, px, panel_y, self.focus, width=max(pw_p, 230))
        else:
            draw_focus_panel(preview_frame, pw - 240, ph - 94, self.focus)

    def _draw_roi(self, preview_frame):
        """ROI枠（選択中のドラッグ枠・確定済みROI・位置合わせ窓）を描く。"""
        ls = self.live_stack
        ru = self.roi_ui
        color = (255, 0, 255)  # Zoom(黄)と区別する
        if ru["dragging"] and ru["start"] and ru["current"]:
            cv2.rectangle(preview_frame, ru["start"], ru["current"], color, 1)
        if ls.roi is not None and self._last_frame is not None and (self.info_display or ru["select"]):
            h, w = self._last_frame.shape[:2]
            ph, pw = preview_frame.shape[:2]

            def to_preview(x0, y0, x1, y1):
                if self.flip_h:
                    x0, x1 = w - x1, w - x0
                if self.flip_v:
                    y0, y1 = h - y1, h - y0
                return (round(x0 * pw / w), round(y0 * ph / h)), (round(x1 * pw / w), round(y1 * ph / h))

            p0, p1 = to_preview(*ls.roi)
            cv2.rectangle(preview_frame, p0, p1, color, 1)
            cv2.putText(preview_frame, "ROI", (p0[0] + 4, max(14, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            win = ls.align_window(w, h)
            if win != ls.roi:
                q0, q1 = to_preview(*win)
                cv2.rectangle(preview_frame, q0, q1, (160, 90, 160), 1)  # 位置合わせ窓（ROIより広いとき）
        if ru["select"]:
            cv2.putText(preview_frame, f"ROI: drag to select  ([{self.roi_key}] cancel)", (10, preview_frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

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
        self._last_frame = frame
        self._frame_id += 1
        if self._pending_roi is not None:
            x, y, rw, rh = self._pending_roi
            self._pending_roi = None
            roi = normalize_roi(x, y, x + rw, y + rh, frame.shape[1], frame.shape[0])
            if roi is None:
                print(f"[roi] --roi {x} {y} {rw} {rh} は無効(範囲外または小さすぎる)のため無視します")
            else:
                self.live_stack.roi = roi
                print(f"[roi] 読み込み: {self.live_stack.roi_text()}")
        if self.stacking_enabled:
            self.live_stack.add_to_buffer(frame)
            result = self.live_stack.process_stack()
            if result is None:
                result = frame
                self.live_stack.stats_frame = None
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
        self._preview_size = (preview_frame.shape[1], preview_frame.shape[0])

        # Now とヒストグラムは停止判定と同じ対象で出す（スタック中は加算値、表示画像は√N補正済みで判定とずれる）。ROIがあればROI内
        stats_src = ls.stats_frame if (self.stacking_enabled and ls.stats_frame is not None) else self._last_frame
        stats_src = ls.roi_view(stats_src)

        if self.info_display:
            mode_text = "Live Stack Mode" if self.stacking_enabled else "Live View Mode"
            cv2.putText(preview_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(preview_frame, source_line, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if self.stacking_enabled:
                cv2.putText(preview_frame, f"Frames: {ls.stack_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                current_ratio = compute_white_ratio_uint8(stats_src, ls.brightness_threshold)
                cv2.putText(preview_frame, f"Now: {current_ratio * 100:.2f}% / Limit: {ls.overflow_ratio_threshold * 100:.2f}%", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            lines = list(extra_lines)
            if ls.roi is not None:
                lines.append(f"ROI: {ls.roi_text()}")
            for i, line in enumerate(lines):
                cv2.putText(preview_frame, line, (10, 150 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.show_hist:
                preview_frame = draw_ccdf_overlay(
                    preview_frame,
                    stats_src,
                    brightness_threshold=ls.brightness_threshold,
                    stop_ratio=ls.overflow_ratio_threshold,
                )

        self._draw_roi(preview_frame)
        self._zoom_region = None
        pip_pos = self._draw_zoom(preview_frame, display_frame) if self.zoom_on else None
        if self.focus["on"]:
            self._draw_focus(preview_frame, pip_pos)

        if self.settings_menu.menu_active:
            preview_frame = self.settings_menu.draw_menu(preview_frame)
        if self.control_menu.menu_active:
            preview_frame = self.control_menu.draw_menu(preview_frame, {"scr": self.screen_rec.active, "zoom": self.zoom_on, "focus": self.focus["on"], "roi": ls.roi is not None or self.roi_ui["select"]})

        self.screen_rec.write(preview_frame)
        self.screen_rec.draw_indicator(preview_frame)  # 録画フレームに写らないよう書き込み後に描画
        return preview_frame

    def apply_settings(self, values):
        """設定メニューで変更された共通項目を反映する"""
        ls = self.live_stack
        if values["max_frames"] != ls.max_frames:
            ls.set_max_frames(values["max_frames"])
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
        elif key == ord("z"):
            self.zoom_on = not self.zoom_on
            if not self.zoom_on:
                self.zoom_rect = None
                self._zoom_drag = {"start": None, "cur": None}
            print(f"[zoom] {'ON' if self.zoom_on else 'OFF'}（クリック/ドラッグで拡大する範囲を選択）")
        elif key == ord("F"):
            self.focus["on"] = not self.focus["on"]
            reset_focus_state(self.focus)
            print(f"[focus] フォーカスアシスト: {'ON' if self.focus['on'] else 'OFF'}（左クリックで測定領域を移動）")
        elif key == ord(self.roi_key):
            self.toggle_roi()
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


def run_stream_mode(source, args, ssh_info=None):
    """URL/カメラ番号ソースでLiveStackを実行（picamera2不要）。ssh_info=(client, pid, ssh_host)があれば終了時にSSH停止する。"""
    print(f"Live Stack - Source: {source}")
    print("操作: [q]終了 [m]設定メニュー [c]操作メニュー [s]PNG [j]JPEG [f]FITS [y]画面録画 [z]ズーム [F]フォーカスアシスト [t]スタックON/OFF [r]リセット [i]情報表示 [h]左右反転 [v]上下反転")

    cap = open_capture(source)
    if not cap.isOpened():
        print(f"ソースを開けませんでした: {source}")
        if ssh_info:
            ssh_kill_rpicam(ssh_info[0], ssh_info[1], ssh_info[2], proc_name="rpicam-vid")
        return

    live_stack = LiveStack(max_frames=args.max_frames)
    ui = LiveUi(args, live_stack, tag="source", has_camera=False)
    ui.attach_mouse("Live Stack")
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
        if ssh_info:
            ssh_kill_rpicam(ssh_info[0], ssh_info[1], ssh_info[2], proc_name="rpicam-vid")
        print("Live Stack終了")


def main():
    parser = argparse.ArgumentParser(description="LiveStack機能付きカメラプレビュー")
    parser.add_argument("-n", "--max-frames", type=int, default=100,
                        help="最大スタックフレーム数 (デフォルト: 100)")
    parser.add_argument("--source", type=str, default=None,
                        help="映像ソース。例: 0 または tcp://192.168.1.17:8888")
    parser.add_argument("--config", type=str, default=None,
                        help="host JSONを読み込み source/ssh_host等を設定 (例: config/host_raspi_3.json)")
    parser.add_argument("--ssh-host", type=str, default=None, help="SSH接続先ホスト。指定時にPi側でrpicam-vidを自動起動")
    parser.add_argument("--ssh-user", type=str, default="pi", help="SSHユーザー名 (デフォルト: pi)")
    parser.add_argument("--ssh-key", type=str, default="~/.ssh/id_rsa", help="SSH秘密鍵パス (デフォルト: ~/.ssh/id_rsa)")
    parser.add_argument("--rpicam-vid-cmd", type=str, default=None, help="Pi側で実行するrpicam-vidコマンド。--listen/-oは自動補完される")
    parser.add_argument("--no-ssh", action="store_true", help="SSH自動起動を無効化")
    parser.add_argument("--flip-h", action="store_true",
                        help="左右反転して起動")
    parser.add_argument("--flip-v", action="store_true",
                        help="上下反転して起動")
    parser.add_argument("--screen-fps", type=float, default=15.0,
                        help="画面録画([y]キー/操作メニュー)のフレームレート (デフォルト: 15)")
    parser.add_argument("--zoom-pip-size", type=int, default=640,
                        help="ズームPiPの表示サイズ px (デフォルト: 640)")
    parser.add_argument("--zoom-click-size", type=int, default=128,
                        help="クリック時に切り出す元フレームの画素数 (デフォルト: 128)")
    parser.add_argument("--zoom-corner", choices=["tl", "tr", "bl", "br"], default="br",
                        help="ズームPiPの配置隅 (デフォルト: br)")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                        help="停止判定・ヒストグラム・位置合わせの対象領域 (受信フレームのpx)。未指定=全体で判定・中央窓で位置合わせ。"
                             "[k]キー(picamera2直結は[o])でもドラッグ指定できる。設定JSONのroi(raw_live_stack用のraw px)は読み込まない")

    pre, _ = parser.parse_known_args()
    if pre.config:
        cfg = load_config(pre.config)
        if cfg:
            parser.set_defaults(**{k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("sensor", "host", "meta", "roi")})
            print(f"[config] 読み込み: {pre.config}")

    args = parser.parse_args()

    if args.no_ssh:
        args.ssh_host = None

    ssh_info = None
    if args.ssh_host:
        if not args.rpicam_vid_cmd:
            parser.error("--ssh-host 指定時は --rpicam-vid-cmd も必須です")
        # rpicam_vid_cmd の -o tcp://.../PORT を優先する(rpicam-raw用sourceとポートが異なるため)
        cmd_port = parse_rpicam_cmd_port(args.rpicam_vid_cmd)
        port = cmd_port if cmd_port is not None else parse_tcp_port(args.source)
        args.source = f"tcp://{args.ssh_host}:{port}"
        client, pid = ssh_start_rpicam(args.ssh_host, args.ssh_user, args.ssh_key, args.rpicam_vid_cmd, port, log_name="rpicam_vid")
        ssh_info = (client, pid, args.ssh_host)
        print(f"[SSH] --source を {args.source} に設定")
        print("[SSH] rpicam-vid 起動待ち (2秒)...")
        time.sleep(2)
    elif args.source is None:
        parser.error("--source または --config に source、もしくは --ssh-host を指定してください")

    parsed_source = parse_source_arg(args.source)
    if parsed_source is not None:
        run_stream_mode(parsed_source, args, ssh_info=ssh_info)
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
    ui.attach_mouse("Live Stack")
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
