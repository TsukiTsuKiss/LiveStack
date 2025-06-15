#!/usr/bin/env python3
"""
LiveStack機能付きカメラプレビューアプリケーション
- リアルタイムカメラプレビュー
- 位置合わせによるフレームスタッキング
- ノイズ軽減と画質向上
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

from camera_config import CameraConfig
import cv2
import numpy as np
import time
from astropy.io import fits  # FITS保存用ライブラリをインポート
from PIL import Image, ExifTags
import piexif

class LiveStack:
    """LiveStack処理クラス"""
    
    def __init__(self, max_frames=100, verbose=False):  # max_framesを100に変更
        self.max_frames = max_frames  # max_framesをリングバッファサイズとして明確化
        self.verbose = verbose  # デバッグ出力制御
        self.reset()
        self.buffer = [None] * max_frames  # Noneで初期化
        self.buffer_index = 0
        self.dark_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # 真っ黒なフレームで初期化
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
                
                # オーバーフロー検出（画面の10%が255に達したかチェック）
                if self.fixed_stack_count is None:
                    # 全チャンネルが255（真っ白）のピクセルを検出
                    white_pixels = np.sum(np.all(self.stacked_image >= 255, axis=2))
                    total_pixels = self.stacked_image.shape[0] * self.stacked_image.shape[1]
                    overflow_ratio = white_pixels / total_pixels
                    
                    if overflow_ratio >= 0.10:  # 10%オーバーフロー
                        self.fixed_stack_count = self.stack_count
                        print(f"*** オーバーフロー検出: {overflow_ratio:.3f} (10%以上) スタック数固定: {self.fixed_stack_count} ***")
                
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
                print(f"フレームが存在しません: index={past_index}")
                continue

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

            # 表示用フレームに加算
            display_frame += aligned_past_frame
            valid_stack_count += 1

            # オーバーフロー条件のチェック
            white_pixels = np.sum(np.all(display_frame >= 255, axis=2))
            total_pixels = display_frame.shape[0] * display_frame.shape[1]
            overflow_ratio = white_pixels / total_pixels
            if overflow_ratio >= 0.10:
                print("オーバーフロー条件に達しました。スタック処理を終了します。")
                break

        self.stack_count = valid_stack_count  # 有効なスタック数を更新
        return np.clip(display_frame, 0, 255).astype(np.uint8)

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

def main():
    print("Live Stack - LiveStack機能付きカメラプレビュー")
    print("操作:")
    print("  [q] 終了")
    print("  [s] 保存")
    print("  [t] LiveStack ON/OFF")
    print("  [r] スタックリセット")
    print("  [h] 高解像度切り替え")
    print("  [+] ゲイン上げ (最大8.0)")
    print("  [-] ゲイン下げ (最小1.0)")
    print("シャッター速度:")
    print("  [1] 1秒   [2] 1/2秒   [3] 1/4秒   [4] 1/8秒   [5] 1/16秒")
    print("  [6] 1/30秒 [7] 1/60秒  [8] 1/125秒 [9] 1/250秒 [0] 1/500秒")
    
    # カメラ初期化
    high_res_mode = False
    low_light_mode = False
    current_gain = 2.0
    current_exposure = 16667  # 1/60秒
    
    picam2 = CameraConfig.create_fast_camera()
    
    # LiveStack初期化
    live_stack = LiveStack(max_frames=100)  # max_framesを100に変更
    stacking_enabled = False
    dark_frame_set = False  # ダークフレーム取得状態を管理

    try:
        picam2.start()
        time.sleep(1)

        while True:
            # フレーム取得
            frame = picam2.capture_array()

            # リングバッファにフレームを追加
            live_stack.add_to_buffer(frame)

            # LiveStack処理
            if stacking_enabled:
                display_frame = live_stack.process_stack()
                if display_frame is None:
                    display_frame = frame
                cv2.putText(display_frame, "Live Stack Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 加算しているフレーム数を表示
                stack_info = f"Frames: {live_stack.stack_count}"
                cv2.putText(display_frame, stack_info, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # ダークフレーム取得状態を表示
                if dark_frame_set:
                    cv2.putText(display_frame, "Dark Frame Set", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                display_frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=20)
                cv2.putText(display_frame, "Live View Mode", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # ダークフレーム取得状態を表示
                if dark_frame_set:
                    cv2.putText(display_frame, "Dark Frame Set", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # カメラ設定情報を表示
            camera_info = f"Gain:{current_gain} Exp:1/{int(1000000/current_exposure)}"
            cv2.putText(display_frame, camera_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # プレビュー表示
            cv2.imshow("Live Stack", display_frame)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                if not stacking_enabled:
                    live_stack.set_dark_frame(frame)
                    dark_frame_set = True
                else:
                    print("ダークフレームはLiveViewモードでのみ取得可能です。")
            elif key == ord("s"):
                filename = f"live_stack_{int(time.time())}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"画像保存: {filename}")
            elif key == ord("t"):
                stacking_enabled = not stacking_enabled
                if stacking_enabled:
                    print("LiveStack 有効")
                    live_stack.reset()
                else:
                    print("LiveStack 無効")
            elif key == ord("r"):
                if stacking_enabled:
                    live_stack.reset()
                    print("スタックリセット - 次のフレームから再開")
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
                CameraConfig.apply_camera_settings(picam2, current_exposure, current_gain)
                print(f"露出: 1/{int(1000000/current_exposure)}秒")

                # ダークフレームとリングバッファをリセット
                try:
                    frame_shape = frame.shape if frame is not None else (480, 640, 3)
                    live_stack.dark_frame = np.zeros(frame_shape, dtype=np.uint8)
                    live_stack.dark_buffer = []  # リングバッファをリセット
                    dark_frame_set = False
                    print("ダークフレームとリングバッファをリセットしました。")
                except Exception as e:
                    print(f"リセット中にエラーが発生しました: {e}")

            elif key in [ord("+"), ord("-")]:
                # ゲイン変更
                if key == ord("+") or key == ord("="):
                    current_gain = min(8.0, current_gain + 0.5)
                elif key == ord("-"):
                    current_gain = max(1.0, current_gain - 0.5)
                CameraConfig.apply_camera_settings(picam2, current_exposure, current_gain)
                print(f"ゲイン: {current_gain}")

                # ダークフレームとリングバッファをリセット
                try:
                    frame_shape = frame.shape if frame is not None else (480, 640, 3)
                    live_stack.dark_frame = np.zeros(frame_shape, dtype=np.uint8)
                    live_stack.dark_buffer = []  # リングバッファをリセット
                    dark_frame_set = False
                    print("ダークフレームとリングバッファをリセットしました。")
                except Exception as e:
                    print(f"リセット中にエラーが発生しました: {e}")
            elif key == ord("f"):
                # FITS保存
                metadata = {
                    "EXPOSURE": current_exposure,  # 露光時間
                    "EXPTIME": "microseconds",  # 露光時間の単位（短縮）
                    "GAIN": current_gain,
                    "STACKCNT": live_stack.stack_count,
                    "DATE": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                filename = f"live_stack_{int(time.time())}.fits"
                save_fits(display_frame, filename, metadata)

            elif key == ord("j"):
                # JPEG保存
                filename = f"live_stack_{int(time.time())}.jpg"
                raw_frame = frame.copy()  # 文字を書き込む前のフレームをコピー
                pil_image = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
                exif_dict = {
                    "0th": {
                        piexif.ImageIFD.DateTime: time.strftime("%Y:%m:%d %H:%M:%S"),
                        piexif.ImageIFD.ExposureTime: (current_exposure, 1000000),
                        piexif.ImageIFD.ImageDescription: f"Stack Count: {live_stack.stack_count}"
                    }
                }
                exif_bytes = piexif.dump(exif_dict)
                pil_image.save(filename, "jpeg", exif=exif_bytes)
                print(f"JPEGファイル保存: {filename}")

            elif key == ord("p"):
                # PNG保存
                filename = f"live_stack_{int(time.time())}.png"
                raw_frame = frame.copy()  # 文字を書き込む前のフレームをコピー
                pil_image = Image.fromarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
                pil_image.save(filename, "png")
                print(f"PNGファイル保存: {filename}")
                print(f"保存データ: Gain={current_gain}, Exposure=1/{int(1000000/current_exposure)}, Stack Count={live_stack.stack_count}")

    except KeyboardInterrupt:
        print("\n終了中...")
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Live Stack終了")

if __name__ == "__main__":
    main()
