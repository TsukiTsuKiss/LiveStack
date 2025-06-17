# Camera Live Applications

Raspberry Pi Camera2用のリアルタイムプレビューアプリケーション

## ファイル構成

```
camera-live/
├── live_view.py      # シンプルなライブプレビュー
├── live_stack.py     # LiveStack機能付きプレビュー
├── common/
│   └── camera_config.py  # 共通カメラ設定
└── README.md
```

## アプリケーション

### 1. Live View (`live_view.py`)
シンプルなリアルタイムカメラプレビュー

**特徴:**
- 最小タイムラグ
- 高速動作
- 基本的な操作のみ

**操作:**
- `q`: 終了
- `s`: 画像保存
- `h`: 高解像度モード切り替え

### 2. Live Stack (`live_stack.py`)
リアルタイムでフレームを加算スタックし、ノイズ軽減と画質向上を実現するプレビューアプリケーション。

#### 主な改善内容
- **リングバッファ導入**: フレームを効率的に管理し、最新フレームを基準に過去フレームを位置合わせ。
- **加算処理の最適化**: 加算平均ではなく純粋な加算を採用。
- **位置合わせ精度向上**: サブピクセル補正、閾値・オフセット制限、明るさ調整ワーク領域を導入。
- **UI表示の改善**: 加算フレーム数、Gain、Expの画面表示を追加。
- **オーバーフロー検知**: 画面の10%以上が白くなる場合にスタック数を固定。

#### 使用方法
1. アプリケーションを起動し、`[t]`キーでLiveStackモードをON/OFF切り替え。
2. `[r]`キーでスタックをリセット。
3. `[+]/[-]`キーでゲインを調整。
4. `[1]`〜`[0]`キーでシャッター速度を変更。

#### 注意点
- スタック数が正しく増加しない問題を修正済み。
- 動作不良時は非スタッキングモードに戻す機能を実装。
- 最新の改善により、安定した動作が確認されています。

### 3. 保存機能
Live Stackモードでは以下の形式で画像を保存できます。

**保存形式:**
- **FITS**: RGB対応、NAXIS3を色数として設定。メタデータ（露光時間、ゲイン、スタック数、日時）を付与。
- **JPEG**: EXIFデータを付与（撮影日時、露光時間、スタック数など）。文字が含まれないフレームを保存。
- **PNG**: 文字が含まれないフレームを保存。EXIFデータは付与されません。

**操作:**
- `f`: FITS形式で保存
- `j`: JPEG形式で保存
- `p`: PNG形式で保存

#### 注意点
- JPEG保存時にEXIFデータが正しく付与されるよう修正済み。
- PNG保存時にはEXIFデータは付与されませんが、保存時に画面表示の文字が含まれないよう対応済み。

## 実行方法

```bash
# 仮想環境をアクティベート
cd /home/tsuki/MyApps/camera-live
source ../preview/.venv/bin/activate

# シンプルプレビュー
python3 live_view.py

# LiveStack機能付き
python3 live_stack.py
```

## 設定

`common/camera_config.py`で以下を調整可能:
- 解像度
- バッファサイズ
- 露出設定
- ゲイン設定

## LiveStack機能

- **位置合わせ**: ORB特徴点とホモグラフィー変換
- **スタッキング**: 加算平均によるノイズ軽減
- **最大フレーム数**: デフォルト20フレーム
- **リアルタイム処理**: フレームレートを維持

## 必要なライブラリのインストール

このアプリケーションを使用するには以下のPythonライブラリが必要です。

**インストール方法:**

1. `astropy` (FITSファイル保存用)
   ```bash
   pip install astropy
   ```

2. `Pillow` (JPEG/PNG保存用、EXIFデータ付与)
   ```bash
   pip install Pillow
   ```

3. `piexif` (EXIFデータ操作用)
   ```bash
   pip install piexif
   ```

4.`opencv`  (画像処理用)
   ```bash
   pip install opencv-python
   ```

5.`picamera2`  (RasPiカメラ用、既に入っているかも)
   ```bash
   pip install picamera2
   ```

**仮想環境を使用する場合:**
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install astropy Pillow piexif opencv-python picamera2
```
