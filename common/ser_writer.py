"""SERファイルのストリーム書き出し（16bit Bayer, リトルエンディアン, SER Player互換ヘッダー）。"""

import datetime
import struct

# ColorID=BayerパターンID: SER Player互換（RGGB↔BGGR, GRBG↔GBRG を反転）
COLOR_ID_MAP = {"RGGB": 11, "GRBG": 10, "GBRG": 9, "BGGR": 8}

_EPOCH = datetime.datetime(1, 1, 1)


def now_ticks():
    """現在のUTC時刻を .NET DateTime.Ticks (100ns単位) で返す。"""
    return int((datetime.datetime.utcnow() - _EPOCH).total_seconds() * 10_000_000)


def _text40(s):
    return (s or "").encode()[:40].ljust(40, b"\x00")


class SerWriter:
    """ヘッダーを書いて開き、write_frame() で追記し、close() でトレーラーとフレーム数を確定する。"""

    FRAME_COUNT_OFFSET = 38

    def __init__(self, path, width, height, bayer_key, observer="", instrument="", telescope=""):
        self.path = path
        self.frame_count = 0
        self._timestamps = []
        ticks = now_ticks()
        utc_offset = datetime.datetime.now().astimezone().utcoffset().total_seconds()
        ticks_local = ticks + int(utc_offset * 10_000_000)
        self._f = open(path, "wb")
        self._f.write(
            b"LUCAM-RECORDER"
            + struct.pack("<I", 0)
            + struct.pack("<I", COLOR_ID_MAP.get(bayer_key, 0))
            + struct.pack("<I", 0)    # LittleEndian: 0=SER Player互換 (フラグ意味が逆のため)
            + struct.pack("<I", width)
            + struct.pack("<I", height)
            + struct.pack("<I", 16)   # PixelDepthPerPlane: 16bit固定
            + struct.pack("<I", 0)    # フレーム数: close() でパッチ
            + _text40(observer)
            + _text40(instrument)
            + _text40(telescope)
            + struct.pack("<q", ticks_local)
            + struct.pack("<q", ticks)
        )

    def write_frame(self, frame_u16):
        self._f.write(frame_u16.tobytes())
        self._timestamps.append(now_ticks())
        self.frame_count += 1

    def close(self):
        if self._f is None:
            return
        for ts in self._timestamps:
            self._f.write(struct.pack("<q", ts))
        self._f.seek(self.FRAME_COUNT_OFFSET)
        self._f.write(struct.pack("<I", self.frame_count))
        self._f.close()
        self._f = None
        self._timestamps.clear()
