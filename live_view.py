#!/usr/bin/env python3
"""
シンプルなライブプレビューアプリケーション
- リアルタイムカメラプレビュー
- 最小限のタイムラグ
- 基本的な操作のみ
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))

from camera_config import CameraConfig
import cv2
import time

def main():
    print("Live View - シンプルなカメラプレビュー")
    print("操作: [q]終了 [s]保存")
    
    # カメラ初期化
    picam2 = CameraConfig.create_fast_camera()
    
    try:
        picam2.start()
        time.sleep(1)
        
        while True:
            # フレーム取得
            frame = picam2.capture_array()
            
            # 軽微な輝度調整
            frame_bgr = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
            
            # プレビュー表示
            cv2.imshow("Live View", frame_bgr)
            
            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"live_view_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame_bgr)
                print(f"画像保存: {filename}")

    except KeyboardInterrupt:
        print("\n終了中...")
    except Exception as e:
        print(f"エラー: {e}")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Live View終了")

if __name__ == "__main__":
    main()
