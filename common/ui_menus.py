"""
共通UI部品

live_stack.py / raw_live_stack.py で共通利用する設定メニュー・操作メニュー・画面録画。
どちらのアプリも同じ操作感になるよう、キー処理と描画をここに集約する。
"""

import datetime
import time

import cv2

from display_utils import decode_key


class SettingsMenu:
    """設定メニュー([m])。値を持つ項目の一覧を上下左右キーで編集する。

    names を指定すると、その名前の項目だけを残す（アプリごとに使える項目が違うため）。
    """

    def __init__(self, names=None):
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
                "name": "Info Display",
                "value": True
            },
            {
                "name": "Histogram",
                "value": True
            },
            {
                "name": "Flip H",
                "value": False
            },
            {
                "name": "Flip V",
                "value": False
            },
            {
                "name": "SER Flip",
                "value": False
            },
            {
                "name": "Align Size(%)",
                "value": 50,
                "min": 10,
                "max": 100,
                "step": 10
            },
            {
                "name": "Align",
                "value": True
            },
            {
                "name": "Dark Sub",
                "value": True
            },
            {
                "name": "Dark Flip",
                "value": "None",
                "values": ["None", "H", "V", "HV", "Auto"]
            },
            {
                "name": "Stack Method",
                "value": "Sum",
                "values": ["Sum", "Lighten", "Darken", "Median", "Sigma Clip", "Sigma Sum"]
            },
            {
                "name": "Stack Mode",
                "value": False
            }
        ]
        if names is not None:
            self.settings = [s for s in self.settings if s["name"] in names]
        self.selected_item = 0
        self.menu_active = False

    def get_setting(self, name):
        """名前で項目を取得する（無ければNone）"""
        for setting in self.settings:
            if setting["name"] == name:
                return setting
        return None

    def set_value(self, name, value):
        """名前で項目の値を更新する（メニューに無い項目は無視）"""
        setting = self.get_setting(name)
        if setting is not None:
            setting["value"] = value

    def handle_key(self, key_raw):
        """キー入力処理"""
        if not self.menu_active:
            return False

        self.apply_requested = False

        key, nav = decode_key(key_raw)

        if nav == "up":
            self.selected_item = (self.selected_item - 1) % len(self.settings)
            return True
        elif nav == "down":
            self.selected_item = (self.selected_item + 1) % len(self.settings)
            return True
        elif nav == "left":
            self.change_value(-1)
            self.apply_requested = True
            return True
        elif nav == "right":
            self.change_value(1)
            self.apply_requested = True
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

        if setting["name"] in ("Camera", "Stack Method", "Dark Flip"):
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

        elif setting["name"] in ("Max Frames", "Stop Ratio(%)", "Stop Threshold", "Align Size(%)"):
            new_value = setting["value"] + (direction * setting["step"])
            setting["value"] = max(setting["min"], min(setting["max"], new_value))

        elif setting["name"] in ("Stack Mode", "Info Display", "Histogram", "Flip H", "Flip V", "SER Flip", "Align", "Dark Sub"):
            setting["value"] = not setting["value"]

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
        menu_h = 120 + len(self.settings) * 35 + 20
        menu_w = 550
        x0, y0 = 500, 10
        x1 = x0 + menu_w
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y0 + menu_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # タイトル
        cv2.putText(frame, "Settings Menu", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Up/Down: Select Item  Left/Right: Change Value  c: Control Menu",
                   (x0 + 10, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Left/Right: Apply Immediately  Enter: Re-Apply  ESC: Cancel",
                   (x0 + 10, y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 設定項目を表示
        for i, setting in enumerate(self.settings):
            y_pos = y0 + 120 + i * 35
            color = (0, 255, 0) if i == self.selected_item else (255, 255, 255)

            # 値のテキスト生成
            if setting["name"] == "Camera":
                value_text = f"{setting['value']}"
            elif setting["name"] == "Exposure":
                value_text = self.get_exposure_text(setting["value"])
            elif setting["name"] == "Size":
                value_text = setting["value"]
            elif setting["name"] == "SER Flip":
                value_text = "ON" if setting["value"] else "OFF"
                # SER録画の反転は画面側の反転(Flip H/V)がONのときだけ効く
                flips = {t["name"]: t["value"] for t in self.settings if t["name"] in ("Flip H", "Flip V")}
                if setting["value"] and flips and not any(flips.values()):
                    value_text += " (inactive: Flip H/V OFF)"
            elif setting["name"] in ("Stack Mode", "Info Display", "Histogram", "Flip H", "Flip V", "Align", "Dark Sub"):
                value_text = "ON" if setting["value"] else "OFF"
            elif setting["name"] in ("Stop Ratio(%)", "Align Size(%)"):
                value_text = f"{int(setting['value'])}%"
            elif setting["name"] == "Stop Threshold":
                value_text = f"{int(setting['value'])}"
            elif setting["name"] == "Max Frames" and "max" in setting:
                value_text = f"Stack:{int(setting['value'])}  (MemMax:{int(setting['max'])})"
            elif setting["name"] == "Gain":
                value_text = f"{setting['value']:.1f}"
            else:
                value_text = str(setting["value"])

            text = f"{setting['name']}: {value_text}"
            cv2.putText(frame, text, (x0 + 30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 選択中の項目にカーソル表示
            if i == self.selected_item:
                cv2.putText(frame, ">", (x0 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
            "stack_method": values_by_name.get("Stack Method", "Sum"),
            "dark_flip": values_by_name.get("Dark Flip", "None"),
            "info_display": bool(values_by_name.get("Info Display", True)),
            "hist_display": bool(values_by_name.get("Histogram", True)),
            "flip_h": bool(values_by_name.get("Flip H", False)),
            "flip_v": bool(values_by_name.get("Flip V", False)),
            "ser_flip": bool(values_by_name.get("SER Flip", False)),
            "stop_threshold": int(values_by_name.get("Stop Threshold", 255)),
            "stop_ratio_percent": int(values_by_name.get("Stop Ratio(%)", 10)),
            "align_size_percent": int(values_by_name.get("Align Size(%)", 50)),
            "align_enabled": bool(values_by_name.get("Align", True)),
            "dark_enabled": bool(values_by_name.get("Dark Sub", True)),
        }

    def set_current_values(self, camera=0, gain=2.0, exposure=16667, max_frames=100, stack_mode=False, info_display=True, hist_display=True, flip_h=False, flip_v=False, ser_flip=False, stack_method="Sum", dark_flip="None", size_label="N/A", stop_ratio_percent=10, stop_threshold=255, align_size_percent=50, align_enabled=True, dark_enabled=True):
        """現在の設定値を更新（キーワード指定で呼ぶこと）"""
        values = {
            "Camera": camera,
            "Size": size_label,
            "Gain": gain,
            "Exposure": exposure,
            "Max Frames": max_frames,
            "Stack Mode": stack_mode,
            "Stack Method": stack_method,
            "Dark Flip": dark_flip,
            "Info Display": info_display,
            "Histogram": bool(hist_display),
            "Flip H": bool(flip_h),
            "Flip V": bool(flip_v),
            "SER Flip": bool(ser_flip),
            "Stop Threshold": int(stop_threshold),
            "Stop Ratio(%)": int(stop_ratio_percent),
            "Align Size(%)": int(align_size_percent),
            "Align": bool(align_enabled),
            "Dark Sub": bool(dark_enabled),
        }

        for setting in self.settings:
            name = setting.get("name")
            if name not in values:
                continue

            value = values[name]
            if name in ["Stop Threshold", "Stop Ratio(%)", "Align Size(%)"] and "min" in setting and "max" in setting:
                value = max(setting["min"], min(setting["max"], int(value)))

            setting["value"] = value

    def set_size_choices(self, size_labels, current_label=None):
        """Size項目の選択肢を更新"""
        setting = self.get_setting("Size")
        if setting is None:
            return
        if not size_labels:
            setting["values"] = ["N/A"]
            setting["value"] = "N/A"
            return

        setting["values"] = size_labels
        if current_label in size_labels:
            setting["value"] = current_label
        else:
            setting["value"] = size_labels[0]


class ControlMenu:
    """操作メニュー([c])。選んだ項目を対応するキー操作として実行する（Shiftが使えない環境向け）。

    items: [(表示名, 実行するキー文字, 状態名またはNone), ...]
    状態名を持つ項目は draw_menu の states から ON/OFF を表示する。
    """

    def __init__(self, items):
        self.items = list(items)
        self.menu_active = False
        self.selected_item = 0

    def toggle(self):
        self.menu_active = not self.menu_active

    def handle_key(self, key_raw):
        """(キーを消費したか, 実行するキー文字コードまたはNone) を返す"""
        if not self.menu_active:
            return False, None
        key, nav = decode_key(key_raw)
        if nav == "up":
            self.selected_item = (self.selected_item - 1) % len(self.items)
            return True, None
        if nav == "down":
            self.selected_item = (self.selected_item + 1) % len(self.items)
            return True, None
        if key in (10, 13):
            return True, ord(self.items[self.selected_item][1])
        if key == 27:
            self.menu_active = False
            return True, None
        return False, None

    def draw_menu(self, frame, states):
        """states: 状態項目名 -> bool"""
        if not self.menu_active:
            return frame
        menu_w = 420
        menu_h = 90 + len(self.items) * 30 + 10
        x0, y0 = 500, 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + menu_w, y0 + menu_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "Control Menu", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Up/Down: Select  Enter: Run  ESC/c: Close  m: Settings",
                    (x0 + 10, y0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        for i, (label, _key, state_name) in enumerate(self.items):
            y_pos = y0 + 95 + i * 30
            color = (0, 255, 0) if i == self.selected_item else (255, 255, 255)
            text = label
            if state_name is not None:
                text += f": {'ON' if states.get(state_name) else 'OFF'}"
            cv2.putText(frame, text, (x0 + 30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            if i == self.selected_item:
                cv2.putText(frame, ">", (x0 + 5, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        return frame


def route_control_key(key, key_raw, settings_menu, control_menu):
    """[c]での操作メニュー開閉と、操作メニュー内のキー処理。

    設定メニューが開いていれば閉じて操作メニューへ切り替える。
    戻り値: (キーを消費したか, 実行するキーコード)。実行キーがあれば呼び出し側は key をそれに置き換える。
    """
    if key == ord("c"):
        if settings_menu.menu_active:
            settings_menu.menu_active = False
            control_menu.menu_active = True
        else:
            control_menu.toggle()
        return True, None
    consumed, action_key = control_menu.handle_key(key_raw)
    return consumed, action_key


def toggle_settings_menu(settings_menu, control_menu):
    """[m]での設定メニュー開閉（操作メニューは閉じる）。開いたらTrueを返す。"""
    settings_menu.menu_active = not settings_menu.menu_active
    if settings_menu.menu_active:
        control_menu.menu_active = False
    return settings_menu.menu_active


class ScreenRecorder:
    """画面録画。info/ヒストグラム/メニュー等を合成した最終プレビューフレームをAVI(MJPG)に保存する。"""

    def __init__(self, fps=15.0):
        self.fps = max(float(fps), 1.0)
        self._interval = 1.0 / self.fps
        self._writer = None
        self._size = None
        self._next_t = 0.0
        self.filename = None
        self.frames = 0

    @property
    def active(self):
        return self._writer is not None

    def toggle(self, frame):
        """開始/停止を切り替える。frame は開始時のサイズ決定に使う。"""
        if self._writer is None:
            self._start(frame)
        else:
            self.stop()

    def _start(self, frame):
        self._size = (frame.shape[1], frame.shape[0])
        self.filename = f"{datetime.datetime.now():%Y%m%d_%H%M%S}_screen.avi"
        writer = cv2.VideoWriter(self.filename, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, self._size)
        if not writer.isOpened():
            print("[SCR] 画面録画を開始できません (VideoWriter open失敗)")
            return
        self._writer = writer
        self.frames = 0
        self._next_t = time.perf_counter()
        print(f"[SCR] 画面録画開始: {self.filename}  {self._size[0]}x{self._size[1]} {self.fps:g}fps (MJPG)")

    def stop(self, reason="停止"):
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        print(f"[SCR] 画面録画{reason}: {self.filename}  {self.frames}フレーム")

    def write(self, frame):
        """録画中なら、指定fpsに合わせてフレームを書き込む"""
        if self._writer is None:
            return
        now = time.perf_counter()
        if now < self._next_t:
            return
        out = frame
        if (frame.shape[1], frame.shape[0]) != self._size:
            out = cv2.resize(frame, self._size)
        # 処理が遅れた分は同じフレームを複製して実時間の長さを保つ
        n_dup = min(int((now - self._next_t) / self._interval) + 1, 10)
        for _ in range(n_dup):
            self._writer.write(out)
        self.frames += n_dup
        self._next_t = now + self._interval if n_dup == 10 else self._next_t + n_dup * self._interval

    def draw_indicator(self, frame):
        """録画中の表示。録画フレームに写らないよう write() の後で呼ぶ。"""
        if self._writer is None:
            return
        cv2.circle(frame, (18, 18), 8, (0, 140, 255), -1)
        cv2.putText(frame, f"SCR {self.frames}f", (32, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1, cv2.LINE_AA)
