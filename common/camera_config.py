# 共通カメラ設定
from picamera2 import Picamera2
from libcamera import Transform
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

class CameraConfig:
    """カメラ設定の共通クラス"""
    
    @staticmethod
    def create_camera(camera_num=0, resolution=(1640, 1232), buffer_count=1):
        """カメラインスタンスを作成して設定"""
        picam2 = Picamera2(camera_num)
        
        # プレビュー設定
        preview_config = picam2.create_preview_configuration(
            main={"size": resolution, "format": "RGB888"},
            buffer_count=buffer_count,
            transform=Transform(hflip=False, vflip=False)
        )
        picam2.configure(preview_config)
        
        # 手動露出設定（タイムラグ削減）
        picam2.set_controls({
            "AeEnable": False, 
            "ExposureTime": 16667,  # 1/60秒（60fps相当）
            "AnalogueGain": 2.0     # ゲインを上げて明るさを確保
        })
        
        return picam2
    
    @staticmethod
    def create_high_res_camera(camera_num=0):
        """高解像度カメラ設定"""
        return CameraConfig.create_camera(camera_num=camera_num, resolution=(2028, 1520))
    
    @staticmethod
    def create_fast_camera(camera_num=0):
        """高速プレビュー用カメラ設定"""
        return CameraConfig.create_camera(camera_num=camera_num, resolution=(1280, 960), buffer_count=1)
    
    @staticmethod
    def create_low_light_camera(camera_num=0):
        """暗環境LiveStack用カメラ設定"""
        picam2 = Picamera2(camera_num)
        
        # プレビュー設定
        preview_config = picam2.create_preview_configuration(
            main={"size": (1280, 960), "format": "RGB888"},
            buffer_count=1,
            transform=Transform(hflip=False, vflip=False)
        )
        picam2.configure(preview_config)
        
        # 暗環境用設定
        picam2.set_controls({
            "AeEnable": False, 
            "ExposureTime": 33333,  # 1/30秒（より長い露出）
            "AnalogueGain": 4.0     # より高いゲイン
        })
        
        return picam2
    
    @staticmethod
    def apply_camera_settings(picam2, exposure_time=16667, gain=2.0):
        """カメラの露出とゲインを動的に調整"""
        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": exposure_time,
            "AnalogueGain": gain
        })
