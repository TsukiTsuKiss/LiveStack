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
シンプルなリアルタイムカメラプレビュー（カメラ切り替え機能付き）

**特徴:**
- 最小タイムラグ
- 高速動作
- 基本的な操作のみ
- カメラ0と1の切り替え対応

**操作:**
- `q`: 終了
- `s`: 画像保存  
- `c`: カメラ切り替え（0 ↔ 1）
- `0`: カメラ0に直接切り替え
- `1`: カメラ1に直接切り替え

### 2. Live Stack (`live_stack.py`)
リアルタイムでフレームを加算スタックし、ノイズ軽減と画質向上を実現するプレビューアプリケーション（カメラ切り替え機能付き）。

#### 主な機能
- **LiveStack処理**: フレーム位置合わせと加算によるノイズ軽減
- **設定メニュー**: リアルタイムでの各種設定変更
- **カメラ切り替え**: 複数カメラ（0と1）のリアルタイム切り替え
- **多様な保存形式**: FITS、JPEG、PNG対応

#### 操作方法

**基本操作:**
- `q`: 終了
- `m`: 設定メニュー開閉
- `i`: 情報表示ON/OFF
- `c`: カメラ切り替え（0 ↔ 1）
- `s`: 画像保存  
- `t`: LiveStack ON/OFF切り替え
- `r`: スタックリセット
- `d`: ダークフレーム取得

**保存操作:**
- `f`: FITS形式で保存
- `j`: JPEG形式で保存
- `p`: PNG形式で保存

**従来のキー操作:**
- `+`/`-`: ゲイン調整
- `1`〜`0`: シャッター速度変更

#### 設定メニュー機能
`[m]`キーで設定メニューを開くと、以下の設定をリアルタイムで変更できます：

**設定項目:**
- **Camera**: 0/1 - 使用するカメラの選択（NEW）
- **Gain**: 1.0～8.0（0.5刻み） - カメラの感度調整
- **Exposure**: 10秒～1/2000秒（15段階） - 露出時間、天体撮影向けの長時間露出対応
- **Max Frames**: 1～100（1刻み） - スタッキングに使用する最大フレーム数
- **Stack Mode**: ON/OFF - LiveStackモードの切り替え
- **Info Display**: ON/OFF - 画面上の情報表示切り替え

**操作方法:**
- **上下矢印**: 設定項目選択
- **左右矢印**: 選択した項目の値変更
- **Enter**: 設定適用
- **ESC**: キャンセル

#### カメラ切り替え機能
**2つの切り替え方法:**
1. **設定メニュー**: `[m]`キーでメニューを開き、Camera項目で左右矢印キーで切り替え
2. **直接キー**: `[c]`キーで即座にカメラ0と1を切り替え

**安全機能:**
- カメラ切り替え時の自動バッファリセット
- 切り替え失敗時の元カメラへの自動復帰
- 現在のカメラ番号をプレビュー画面に表示
- 保存ファイル名にカメラ番号を自動付与

#### 注意点
- スタック数が正しく増加しない問題を修正済み。
- 動作不良時は非スタッキングモードに戻す機能を実装。
- 最新の改善により、安定した動作が確認されています。

### 3. 保存機能
Live Stackモードでは以下の形式で画像を保存できます。

**保存形式:**
- **FITS**: RGB対応、NAXIS3を色数として設定。メタデータ（露光時間、ゲイン、スタック数、日時）を付与。オーバーレイテキストを除外した元フレームデータを保存（NEW）。
- **JPEG**: EXIFデータを付与（撮影日時、露光時間、スタック数など）。文字が含まれないフレームを保存。
- **PNG**: 文字が含まれないフレームを保存。EXIFデータは付与されません。

**操作:**
- `f`: FITS形式で保存
- `j`: JPEG形式で保存
- `p`: PNG形式で保存

#### 注意点
- JPEG保存時にEXIFデータが正しく付与されるよう修正済み。
- PNG保存時にはEXIFデータは付与されませんが、保存時に画面表示の文字が含まれないよう対応済み。
- FITS保存では情報表示テキストが除外され、純粋な画像データのみを保存（NEW）。

#### 最新の改善点（2025/07/07）
- **カメラ切り替え機能**: 複数カメラ（0と1）のリアルタイム切り替えに対応
- **設定メニューにカメラ選択追加**: 設定メニューからカメラ番号を選択可能
- **安全なカメラ切り替え**: 切り替え失敗時の自動復帰機能
- **ファイル名にカメラ番号付与**: 保存ファイル名にカメラ番号を自動追加
- **メニュー表示の最適化**: カメラ表示の冗長性を解消（Camera: 0）
- **情報表示切り替え機能**: `[i]`キーで画面上のテキスト表示をON/OFF可能
- **設定メニューに情報表示項目追加**: 設定メニューから情報表示のON/OFFを設定可能
- **FITS保存の改善**: オーバーレイテキストを除外し、元のフレームデータのみを保存するよう修正
- **双方向同期**: キー操作と設定メニューの設定値が相互に同期される仕組みを実装

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
- カメラ番号（複数カメラ対応）

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
