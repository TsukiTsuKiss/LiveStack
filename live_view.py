#!/usr/bin/env python3
"""
シンプルなライブプレビューアプリケーション
- リアルタイムカメラプレビュー
- 最小限のタイムラグ
- 基本的な操作のみ
- カメラ切り替え機能付き
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

from camera_config import CameraConfig
import cv2
import time

def create_camera_safely(camera_num):
    """カメラを安全に作成する（エラーハンドリング付き）"""
    try:
        return CameraConfig.create_fast_camera(camera_num)
    except Exception as e:
        print(f"カメラ {camera_num} の作成に失敗: {e}")
        return None

def main():
    print("Live View - シンプルなカメラプレビュー（カメラ切り替え機能付き）")
    print("操作: [q]終了 [s]保存 [c]カメラ切り替え [0-1]カメラ番号直接指定")
    
    # カメラ初期化
    current_camera = 0
    picam2 = create_camera_safely(current_camera)
    
    if picam2 is None:
        print("カメラの初期化に失敗しました。")
        return
    
    print(f"カメラ {current_camera} を使用中")
    
    try:
        picam2.start()
        time.sleep(1)
        
        while True:
            # フレーム取得
            frame = picam2.capture_array()
            
            # 軽微な輝度調整
            frame_bgr = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            
            # カメラ情報をフレームに表示
            cv2.putText(frame_bgr, f"Camera {current_camera}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # プレビュー表示
            cv2.imshow("Live View", frame_bgr)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"live_view_cam{current_camera}_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame_bgr)
                print(f"画像保存: {filename}")
            elif key == ord("c"):
                # カメラ切り替え
                print("カメラを切り替え中...")
                
                # 現在のカメラを停止
                picam2.stop()
                picam2.close()
                
                # 次のカメラ番号を決定（0と1の間で切り替え）
                next_camera = 1 - current_camera
                
                # 新しいカメラを作成
                new_picam2 = create_camera_safely(next_camera)
                
                if new_picam2 is not None:
                    # 切り替え成功
                    picam2 = new_picam2
                    current_camera = next_camera
                    picam2.start()
                    time.sleep(1)
                    print(f"カメラ {current_camera} に切り替えました")
                else:
                    # 切り替え失敗、元のカメラに戻す
                    print(f"カメラ {next_camera} への切り替えに失敗。元のカメラに戻します。")
                    picam2 = create_camera_safely(current_camera)
                    if picam2 is not None:
                        picam2.start()
                        time.sleep(1)
                        print(f"カメラ {current_camera} に復帰しました")
                    else:
                        print("カメラの復帰に失敗しました。")
                        break
            elif key in [ord("0"), ord("1")]:
                # カメラ番号直接指定
                target_camera = int(chr(key))
                if target_camera != current_camera:
                    print(f"カメラ {target_camera} に切り替え中...")
                    
                    # 現在のカメラを停止
                    picam2.stop()
                    picam2.close()
                    
                    # 指定されたカメラを作成
                    new_picam2 = create_camera_safely(target_camera)
                    
                    if new_picam2 is not None:
                        # 切り替え成功
                        picam2 = new_picam2
                        current_camera = target_camera
                        picam2.start()
                        time.sleep(1)
                        print(f"カメラ {current_camera} に切り替えました")
                    else:
                        # 切り替え失敗、元のカメラに戻す
                        print(f"カメラ {target_camera} への切り替えに失敗。元のカメラに戻します。")
                        picam2 = create_camera_safely(current_camera)
                        if picam2 is not None:
                            picam2.start()
                            time.sleep(1)
                            print(f"カメラ {current_camera} に復帰しました")
                        else:
                            print("カメラの復帰に失敗しました。")
                            break
                else:
                    print(f"既にカメラ {current_camera} を使用中です")

    except KeyboardInterrupt:
        print("\n終了中...")
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        if picam2 is not None:
            picam2.stop()
            picam2.close()
        cv2.destroyAllWindows()
        print("Live View終了")

if __name__ == "__main__":
    main()
