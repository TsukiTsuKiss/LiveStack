#!/usr/bin/env python3
"""
デモ動画の自動生成（SNS紹介用）

raw_live_stack.py にキー操作を自動で送り、アプリ自身の画面録画([y]キー)でAVI(MJPG)を作る。
メニュー・info・ヒストグラムがそのまま映り、操作内容は下部のキャプションで示す（録画のみ。アプリ本体は変更しない）。

使い方:
  python demo/make_demo.py --ser 20261013_225204.ser            # 全シーン
  python demo/make_demo.py --ser xxx.ser --scenes live,menu      # シーンを指定
  python demo/make_demo.py --ser xxx.ser --mp4                   # ffmpegでmp4(H.264)も作る
  python demo/make_demo.py --ser xxx.ser --wb 1.6,1.0,1.4        # ホワイトバランス B,G,R（既定は1.0,1.0,1.0=なし）
  python demo/make_demo.py --ser xxx.ser --dark dark.fits        # 実ダークを使う（向きは自動判定。省略時は黒レベルだけの擬似ダーク）

出力: demo/out/<シーン名>.avi（--mp4 指定時は .mp4 も）。保存系の操作で作られる画像・SERは一時ディレクトリに出して削除する。
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

UP, DOWN, LEFT, RIGHT = 0xFF52, 0xFF54, 0xFF51, 0xFF53  # decode_key() が解釈するLinux(GTK/X11)のキーシム
ENTER, ESC = 13, 27
SPECIAL = {"UP": UP, "DOWN": DOWN, "LEFT": LEFT, "RIGHT": RIGHT, "ENTER": ENTER, "ESC": ESC}


class Timeline:
    """(時刻, キー, キャプション) の並びを作る。press/cap は hold 秒だけ時間を進める。"""

    def __init__(self):
        self.t = 1.0  # 起動直後の余裕
        self.events = []

    def press(self, key, caption=None, hold=1.0):
        self.events.append((self.t, key, caption))
        self.t += hold

    def cap(self, caption, hold=2.0):
        self.events.append((self.t, None, caption))
        self.t += hold

    def repeat(self, key, n, caption=None, gap=0.08, hold=0.6):
        for i in range(n):
            self.events.append((self.t, key, caption if i == 0 else None))
            self.t += gap
        self.t += hold

    def goto(self, cur, target, n, caption=None, hold=0.45):
        """メニューの選択項目を cur から target へ（近い方向に循環して）動かす。新しい位置を返す。"""
        steps = (target - cur) % n
        if steps == 0:
            if caption:
                self.cap(caption, 1.0)
        elif steps > n // 2:
            self.repeat("UP", n - steps, caption, gap=0.25, hold=hold)
        else:
            self.repeat("DOWN", steps, caption, gap=0.25, hold=hold)
        return target


# 設定メニュー(m)の項目（ストリーム受信/SER再生時）
S_THRESH, S_RATIO, S_MAXF, S_INFO, S_HIST, S_FLIPH, S_FLIPV, S_SERFLIP, S_ALIGNSZ, S_ALIGN, S_DARK, S_DARKFLIP, S_METHOD, S_STACK = range(14)
S_N = 14
# 操作メニュー(c)の項目
C_PNG, C_JPEG, C_FITS, C_NPY, C_SER, C_SCR, C_ZOOM, C_CLICKWB, C_WBRESET, C_GAMMA, C_STACKRESET, C_DARKCAP, C_DARKCLR, C_JSON = range(14)
C_N = 14


def scene_live(tl):
    tl.cap("Live View: RAW 12bit playback with info + histogram overlay", 3.5)
    tl.press("i", "[i] Info overlay OFF", 2.0)
    tl.press("i", "[i] Info overlay ON", 2.0)
    tl.press("H", "[H] Histogram / CCDF OFF", 2.0)
    tl.press("H", "[H] Histogram / CCDF ON", 2.0)
    tl.press("a", "[a] Auto stretch OFF (raw 12bit -> 8bit)", 2.5)
    tl.press("a", "[a] Auto stretch ON", 2.0)
    tl.press("h", "[h] Flip horizontal", 2.0)
    tl.press("h", "[h] Flip horizontal OFF", 1.5)
    tl.press("v", "[v] Flip vertical", 2.0)
    tl.press("v", "[v] Flip vertical OFF", 1.5)
    tl.press("b", "[b] Bayer pattern cycle (colors change)", 1.5)
    tl.press("b", None, 1.5)
    tl.press("b", None, 1.5)
    tl.press("b", "Back to the original Bayer pattern", 2.0)


def scene_menu(tl):
    tl.cap("Settings menu", 1.5)
    tl.press("m", "[m] Open settings menu  (Up/Down: select, Left/Right: change)", 2.5)
    cur = S_THRESH
    tl.cap("Stop Threshold / Stop Ratio: when summing stops (overflow)", 2.0)
    cur = tl.goto(cur, S_RATIO, S_N)
    tl.repeat("RIGHT", 6, "Stop Ratio: 10% -> 16%", gap=0.3, hold=1.2)
    tl.repeat("LEFT", 6, "Stop Ratio: back to 10%", gap=0.3, hold=1.0)
    cur = tl.goto(cur, S_MAXF, S_N, "Max Frames (ring buffer size)")
    tl.repeat("RIGHT", 25, "Max Frames +25", gap=0.06, hold=1.2)
    cur = tl.goto(cur, S_INFO, S_N, "ON/OFF items toggle with Left/Right")
    tl.press("RIGHT", "Info Display OFF", 1.8)
    tl.press("RIGHT", "Info Display ON", 1.5)
    cur = tl.goto(cur, S_HIST, S_N)
    tl.press("RIGHT", "Histogram OFF", 1.8)
    tl.press("RIGHT", "Histogram ON", 1.5)
    cur = tl.goto(cur, S_FLIPH, S_N)
    tl.press("RIGHT", "Flip H ON", 2.0)
    tl.press("RIGHT", "Flip H OFF", 1.2)
    cur = tl.goto(cur, S_SERFLIP, S_N, "SER Flip: apply flip to SER recording (needs Flip H/V)")
    tl.press("RIGHT", "SER Flip ON  (inactive while Flip H/V are OFF)", 3.0)
    tl.press("RIGHT", "SER Flip OFF", 1.2)
    cur = tl.goto(cur, S_DARKFLIP, S_N, "Dark Flip: match the dark to a flipped SER (Auto detects it)")
    tl.cap("Dark Flip: auto-detected from the hot pixels shared by the dark and the frames", 3.0)
    cur = tl.goto(cur, S_METHOD, S_N, "Stack Method / Stack Mode", hold=0.3)
    tl.repeat("RIGHT", 1, "Stack Method: Lighten", hold=1.5)
    tl.repeat("RIGHT", 1, "Stack Method: Darken", hold=1.5)
    tl.repeat("RIGHT", 1, "Stack Method: Median", hold=1.5)
    tl.repeat("RIGHT", 1, "Stack Method: Sigma Clip", hold=1.5)
    tl.repeat("RIGHT", 1, "Stack Method: Sigma Sum", hold=1.5)
    tl.repeat("RIGHT", 1, "Stack Method: Sum", hold=1.5)
    cur = tl.goto(cur, S_STACK, S_N)
    tl.press("RIGHT", "Stack Mode ON: LiveStack starts", 4.0)
    tl.press("RIGHT", "Stack Mode OFF", 1.5)
    tl.press("c", "[c] Jump to the Control menu", 2.0)
    tl.press("m", "[m] ...and back to the Settings menu", 2.0)
    tl.press("ESC", "[Esc] Close", 2.0)


def scene_stack(tl):
    tl.cap("LiveStack: sum frames until the image saturates (Stop Threshold 2048)", 2.5)
    tl.press("t", "[t] Stack ON  -  Sum stops at the overflow condition", 8.0)
    # メニューで方式を順に切り替える（閉じて結果を見せる）
    methods = [
        "Lighten (max): star trails, meteors, satellites",
        "Darken (min): remove bright streaks",
        "Median: rejects outliers",
        "Sigma Clip: mean after outlier rejection",
        "Sigma Sum: sigma-clipped sum, same scale as Sum",
        "Sum (default)",
    ]
    cur = S_THRESH
    for i, cap in enumerate(methods):
        tl.press("m", "[m] Settings -> Stack Method", 1.2)
        if i == 0:
            cur = tl.goto(cur, S_METHOD, S_N)
        tl.press("RIGHT", cap, 1.5)
        tl.press("m", cap, 8.0)
    tl.press("t", "[t] Stack OFF", 3.0)


def scene_control(tl):
    tl.cap("Control menu: run actions without remembering keys", 2.0)
    tl.press("c", "[c] Open control menu  (Up/Down: select, Enter: run)", 2.5)
    cur = C_PNG
    tl.press("ENTER", "Save PNG", 1.8)
    cur = tl.goto(cur, C_JPEG, C_N)
    tl.press("ENTER", "Save JPEG", 1.8)
    cur = tl.goto(cur, C_FITS, C_N)
    tl.press("ENTER", "Save FITS", 2.2)
    cur = tl.goto(cur, C_SER, C_N, "SER Rec: record raw frames (no stacking) for later")
    tl.press("ENTER", "SER Rec ON  (REC indicator, top right)", 3.5)
    tl.press("ENTER", "SER Rec OFF", 2.0)
    cur = tl.goto(cur, C_WBRESET, C_N)
    tl.press("ENTER", "WB Reset", 1.8)
    cur = tl.goto(cur, C_GAMMA, C_N)
    tl.press("ENTER", "Gamma Reset", 1.8)
    cur = tl.goto(cur, C_JSON, C_N)
    tl.press("ENTER", "Save Config JSON (WB / Gamma / thresholds ...)", 2.2)
    cur = tl.goto(cur, C_STACKRESET, C_N)
    tl.press("m", "[m] Switch to Settings menu", 2.0)
    tl.press("c", "[c] ...and back to Control menu", 2.0)
    tl.press("ESC", "[Esc] Close", 2.0)


def scene_player(tl):
    tl.cap("SER player (record now, process later)", 3.0)
    tl.press("t", "[t] Stack ON", 3.0)  # SERの1周(8fpsで数秒)が終わる前に止める
    tl.press(" ", "[Space] Pause", 2.5)
    tl.repeat(".", 3, "[.] Step forward", gap=0.8, hold=1.2)
    tl.repeat(",", 2, "[,] Step back (stack resets)", gap=0.8, hold=1.2)
    tl.press("]", "[ ] ] Jump to the last frame", 2.0)
    tl.press("[", "[ [ ] Jump to the first frame", 2.0)
    tl.press(" ", "[Space] Play", 5.0)
    tl.press("l", "[l] Loop: continuous (no reset at the end)", 7.0)
    tl.press("l", "[l] Loop: off (stop at the end)", 9.0)
    tl.press(" ", "[Space] Play again from the start", 4.0)
    tl.press("l", "[l] Loop: reset (default)", 5.0)


SCENES = {
    "live": ("01_live_view", scene_live, []),
    "menu": ("02_settings_menu", scene_menu, []),
    "stack": ("03_stack_methods", scene_stack, ["--max-frames", "20", "--ser-loop-mode", "continuous"]),
    "control": ("04_control_menu", scene_control, []),
    "player": ("05_ser_player", scene_player, ["--max-frames", "30"]),
}


def write_pedestal_dark(ser_path, path):
    """黒レベル(ペデスタル)だけの擬似ダークをFITSで作る。

    SERには専用のダークが無いため、先頭フレームの下位0.1%点（>>4で12bit換算）の95%を全画素の値にする。
    これでスタック表示が黒浮きせず、Dark Sub のON/OFFも見せられる（実機のダークの代わりのデモ用）。
    """
    import struct

    import numpy as np
    from astropy.io import fits

    with open(ser_path, "rb") as f:
        head = f.read(178)
        _cid, _le, w, h, _depth, _n = struct.unpack_from("<6I", head, 18)
        frame = np.frombuffer(f.read(w * h * 2), dtype="<u2").reshape(h, w)
    level = float(np.percentile(frame >> 4, 0.1)) * 0.95
    hdu = fits.PrimaryHDU(np.full((h, w), level, np.float32))
    hdu.header["DARK_CNT"] = 1
    hdu.header["BAYER"] = "DEMO"
    hdu.writeto(path, overwrite=True)
    return level


def run_child(scene_key, ser_path, wb, dark_path):
    """子プロセス側: アプリにキー操作を注入して画面録画する。"""
    import runpy
    import time

    import cv2

    name, builder, extra = SCENES[scene_key]
    tl = Timeline()
    builder(tl)
    events = [(0.3, "y", "")] + tl.events + [(tl.t + 0.5, "y", ""), (tl.t + 1.5, "q", None)]

    sys.path.insert(0, os.path.join(ROOT, "common"))
    import ui_menus

    caption = [""]
    orig_write = ui_menus.ScreenRecorder.write

    def write_with_caption(self, frame):
        # 録画フレームにだけ操作キャプションを重ねる（アプリ本体は変更しない）
        if caption[0] and self.active:
            h, w = frame.shape[:2]
            (tw, th), _ = cv2.getTextSize(caption[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            x0 = max(10, (w - tw) // 2)
            y0 = h - 30
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0 - 14, y0 - th - 14), (x0 + tw + 14, y0 + 14), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, caption[0], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        orig_write(self, frame)

    ui_menus.ScreenRecorder.write = write_with_caption

    state = {"t0": None, "i": 0}
    orig_wait = cv2.waitKeyEx

    def wait_with_script(delay):
        r = orig_wait(delay)
        now = time.time()
        if state["t0"] is None:
            state["t0"] = now
        t = now - state["t0"]
        while state["i"] < len(events) and events[state["i"]][0] <= t:
            _t, key, cap = events[state["i"]]
            state["i"] += 1
            if cap is not None:
                caption[0] = cap
            if key is not None:
                return SPECIAL[key] if key in SPECIAL else ord(key)
        return r

    cv2.waitKeyEx = wait_with_script
    if dark_path:
        shutil.copy(dark_path, "demo_dark.fits")  # 実ダーク（コピーを使うので元のファイルは書き換えない）
        dark_args = ["--dark-flip", "auto"]
    else:
        write_pedestal_dark(ser_path, "demo_dark.fits")
        dark_args = []
    # 画面に出るパスをファイル名だけにする（ホームディレクトリ名などを動画に映さない）
    shown = os.path.basename(ser_path)
    os.symlink(ser_path, shown)
    sys.argv = ["raw_live_stack.py", "--source", shown, "--ser-fps", "8", "--stop-threshold", "2048",
                "--dark-file", "demo_dark.fits",
                "--wb-b", wb[0], "--wb-g", wb[1], "--wb-r", wb[2]] + dark_args + extra
    runpy.run_path(os.path.join(ROOT, "raw_live_stack.py"), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description="デモ動画の自動生成")
    parser.add_argument("--ser", required=True, help="再生するSERファイル")
    parser.add_argument("--scenes", default=",".join(SCENES), help=f"カンマ区切り: {','.join(SCENES)}")
    parser.add_argument("--mp4", action="store_true", help="ffmpegでmp4(H.264)も作る")
    parser.add_argument("--dark", help="実ダークのFITS (例: dark.fits)。向きは --dark-flip auto で自動判定する。省略時は黒レベルだけの擬似ダーク")
    parser.add_argument("--wb", default="1.0,1.0,1.0", help="ホワイトバランスのゲイン B,G,R (デフォルト: 1.0,1.0,1.0=なし)")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    ser_path = os.path.abspath(args.ser)
    wb = [v.strip() for v in args.wb.split(",")]
    if len(wb) != 3:
        parser.error("--wb は B,G,R の3値をカンマ区切りで指定してください")
    dark_path = os.path.abspath(args.dark) if args.dark else None
    if dark_path and not os.path.exists(dark_path):
        parser.error(f"ダークが見つかりません: {args.dark}")
    if args.child:
        run_child(args.child, ser_path, wb, dark_path)
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    for key in [s.strip() for s in args.scenes.split(",") if s.strip()]:
        if key not in SCENES:
            parser.error(f"不明なシーン: {key}")
        name = SCENES[key][0]
        print(f"=== {name} ===", flush=True)
        work = tempfile.mkdtemp(prefix="livestack_demo_")
        try:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--ser", ser_path, "--wb", args.wb] + (["--dark", dark_path] if dark_path else []) + ["--child", key],
                           cwd=work, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            avis = sorted(glob.glob(os.path.join(work, "**", "*_screen.avi"), recursive=True))
            if not avis:
                print("  録画ファイルが作られませんでした")
                continue
            dst = os.path.join(OUT_DIR, f"{name}.avi")
            shutil.move(avis[-1], dst)
            print(f"  -> {dst}", flush=True)
            if args.mp4:
                mp4 = os.path.join(OUT_DIR, f"{name}.mp4")
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", dst, "-c:v", "libx264", "-crf", "18",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], check=True)
                print(f"  -> {mp4}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
