# 共通カメラ設定
from picamera2 import Picamera2
from libcamera import Transform
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

class CameraConfig:
    """カメラ設定の共通クラス"""

    @staticmethod
    def _valid_sensor_modes(picam2):
        """16bitを除外した有効なsensor mode一覧を返す"""
        modes = picam2.sensor_modes
        return [
            m for m in modes
            if m.get("bit_depth", 0) != 16
            and "16" not in str(m.get("format", ""))
        ]

    @staticmethod
    def _pick_best_mode(picam2, prefer_small=True):
        """
        sensor_modes から最適な RAW モードを選んで返す。

        【方針】
        - 16bit モード (SRGGB16 等) は ISP 統計が壊れるため除外。
        - prefer_small=True  → 最小解像度（最高 fps）モードを返す。
        - prefer_small=False → 最大解像度（フル画角）モードを返す。
        - 有効なモードが一つもない場合は None を返す。

        戻り値: {"format": str, "size": (w, h)} または None
        """
        valid = CameraConfig._valid_sensor_modes(picam2)

        if not valid:
            return None  # 全モードが 16bit → picamera2 の自動選択に委ねる

        key_fn = lambda m: m["size"][0] * m["size"][1]
        best = min(valid, key=key_fn) if prefer_small else max(valid, key=key_fn)
        return {"format": str(best["format"]), "size": best["size"]}

    @staticmethod
    def get_available_sizes(camera_num=0):
        """利用可能な解像度サイズ一覧を返す（小さい順、重複なし）"""
        picam2 = Picamera2(camera_num)
        try:
            valid = CameraConfig._valid_sensor_modes(picam2)
            unique_sizes = sorted(
                {tuple(m["size"]) for m in valid},
                key=lambda s: s[0] * s[1]
            )
            return unique_sizes
        finally:
            picam2.close()

    @staticmethod
    def create_camera_with_size(camera_num=0, size=None, buffer_count=1):
        """指定解像度サイズでカメラを作成（見つからない場合は最小サイズにフォールバック）"""
        picam2 = Picamera2(camera_num)

        valid = CameraConfig._valid_sensor_modes(picam2)
        selected_mode = None
        if size is not None:
            for mode in valid:
                if tuple(mode["size"]) == tuple(size):
                    selected_mode = mode
                    break

        if selected_mode is None:
            fallback = CameraConfig._pick_best_mode(picam2, prefer_small=True)
            raw_params = fallback
            main_size = fallback["size"] if fallback else None
        else:
            raw_params = {"format": str(selected_mode["format"]), "size": selected_mode["size"]}
            main_size = selected_mode["size"]

        main_cfg = {"format": "RGB888"}
        if main_size:
            main_cfg["size"] = main_size

        preview_config = picam2.create_preview_configuration(
            main=main_cfg,
            raw=raw_params,
            buffer_count=buffer_count,
            transform=Transform(hflip=False, vflip=False)
        )
        picam2.configure(preview_config)

        actual = picam2.camera_configuration()["main"]["size"]
        logging.info(f"Camera {camera_num}: output {actual[0]}x{actual[1]}, raw={raw_params}")

        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": 16667,
            "AnalogueGain": 2.0
        })

        return picam2

    @staticmethod
    def create_camera(camera_num=0, prefer_small=True, buffer_count=1):
        """
        カメラインスタンスを作成して設定。

        解像度はセンサーモードから自動取得するため、センサーごとに
        テーブルを管理する必要はない。クロップも発生しない。

        prefer_small=True  → 最小解像度（高 fps プレビュー向け）
        prefer_small=False → 最大解像度（高解像度保存向け）
        """
        picam2 = Picamera2(camera_num)

        raw_params = CameraConfig._pick_best_mode(picam2, prefer_small=prefer_small)

        # センサーモードと同じサイズを main に指定 → クロップゼロ
        # raw_params が None の場合は picamera2 の自動選択に委ねる
        main_size = raw_params["size"] if raw_params else None
        main_cfg = {"format": "RGB888"}
        if main_size:
            main_cfg["size"] = main_size

        preview_config = picam2.create_preview_configuration(
            main=main_cfg,
            raw=raw_params,
            buffer_count=buffer_count,
            transform=Transform(hflip=False, vflip=False)
        )
        picam2.configure(preview_config)

        actual = picam2.camera_configuration()["main"]["size"]
        logging.info(f"Camera {camera_num}: output {actual[0]}x{actual[1]}, raw={raw_params}")

        # 手動露出設定（タイムラグ削減）
        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": 16667,  # 1/60秒
            "AnalogueGain": 2.0
        })

        return picam2

    @staticmethod
    def create_high_res_camera(camera_num=0):
        """最大解像度カメラ設定（センサー全面・クロップなし）"""
        return CameraConfig.create_camera(camera_num=camera_num, prefer_small=False)

    @staticmethod
    def create_fast_camera(camera_num=0):
        """高速プレビュー用カメラ設定（最小解像度・クロップなし）"""
        return CameraConfig.create_camera(camera_num=camera_num, prefer_small=True, buffer_count=1)

    @staticmethod
    def create_low_light_camera(camera_num=0):
        """暗環境 LiveStack 用カメラ設定（最小解像度・クロップなし）"""
        picam2 = Picamera2(camera_num)

        raw_params = CameraConfig._pick_best_mode(picam2, prefer_small=True)
        main_size = raw_params["size"] if raw_params else None
        main_cfg = {"format": "RGB888"}
        if main_size:
            main_cfg["size"] = main_size

        preview_config = picam2.create_preview_configuration(
            main=main_cfg,
            raw=raw_params,
            buffer_count=1,
            transform=Transform(hflip=False, vflip=False)
        )
        picam2.configure(preview_config)

        actual = picam2.camera_configuration()["main"]["size"]
        logging.info(f"Camera {camera_num} (low-light): output {actual[0]}x{actual[1]}, raw={raw_params}")

        # 暗環境用設定
        picam2.set_controls({
            "AeEnable": False,
            "ExposureTime": 33333,  # 1/30秒
            "AnalogueGain": 4.0
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
