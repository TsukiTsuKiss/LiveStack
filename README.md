# Camera Live Applications

> **淡い天体を実況したい** ―― Raspberry Pi + PC によるRAWライブスタックアプリ群。
> rpicam-raw の生ストリームをTCP経由でPCに受け、12bit RAWデベイヤ・加算スタック・ダーク減算・WB・FITS/SER保存までをリアルタイムで行う。Windows / Linux 対応。

## 目的

このリポジトリには、次の2つの目的がある。

1. Raspberry Pi Camera2向けに、ライブ表示・スタック・画像保存・RAW処理を含む実用的な撮像アプリ群をPythonで構築する。
2. AIを活用したプログラム作成プロセスを体験し、設計・実装・検証の進め方を実践的に学ぶ。

背景として、スタック処理・画像処理・SER出力はCで個別実装した資産があり、Raspberry Piのカメラシステム変更のタイミングでPython実装へ展開している。

## スクリプト機能比較

| 機能 | live_stack.py | raw_live_stack.py |
|---|:---:|:---:|
| **入力ソース** | picamera2 / URL | TCP RAWストリーム |
| **RAW（Bayer）処理** | × | 〇 |
| **加算スタック** | 〇 | 〇 |
| **ダークフレーム減算** | 〇 | 〇 |
| **WBゲイン調整** | × | 〇 |
| **ガンマ補正** | × | 〇 |
| **自動ストレッチ** | × | 〇 |
| **ヒストグラム+CCDF** | 〇 | 〇 |
| **フリップ（H/V）** | 〇 | 〇 |
| **PNG保存** | 〇 | 〇 |
| **JPEG保存** | 〇 | 〇 |
| **FITS保存** | 〇 | 〇 |
| **NPY(RAW16)保存** | × | 〇 |
| **SER録画** | × | 〇 |
| **カメラ切り替え** | 〇 | × |
| **設定メニュー** | 〇 | 〇 |
| **SSH自動起動** | × | 〇 |

> 詳細は各スクリプトのセクションを参照。

---

## ファイル構成

```
camera-live/
├── live_stack.py         # LiveStack機能付きプレビュー（カメラ直結/URL対応）
├── raw_live_stack.py     # rpicam-raw TCP受信 + LiveStack（RAW専用）
├── 動作実績.md            # 機材ごとの送受信コマンド実績
├── common/
│   ├── camera_config.py  # 共通カメラ設定
│   ├── display_utils.py  # 表示ユーティリティ
│   ├── hist_overlay.py   # ヒストグラム+CCDFオーバーレイ
│   └── raw_utils.py      # RAW受信・デコード・WB/ガンマ共通処理
└── README.md
```

## 関連ドキュメント

- [仕様.md](仕様.md): wire-format / 判定仕様 / UI仕様
- [動作実績.md](動作実績.md): 機材・解像度ごとの動作確認コマンド集

## アプリケーション

### 1. RAW Live Stack (`raw_live_stack.py`)
`rpicam-raw` の生TCPストリームを受信しながらリアルタイムにフレームをスタックするRAW専用アプリ。

**主な特徴:**
- **raw16バッファ**: バッファを12bit Bayer（uint16）のまま保持し、加算スタック後に固定スケール（÷16）で8bit変換して表示することで暗い星が枚数増加とともに浮かび上がる
- **スタックスレッド分離**: `process_stack` をバックグラウンドスレッドで実行し、メインスレッドはキー入力・フレーム受信・表示を独立して継続するため、重いスタック計算中でも操作が止まらない
- **ダークフレーム**: `[d]` でレンズキャップ状態のフレームを取得・平均化。`[D]` でクリア。終了時に `dark.fits` へ自動保存、次回起動時に自動読み込み
- **設定ファイル**: `--config` でJSON設定ファイルを読み込み。`[C]` キーでWB/ガンマ/Stop値を含む現在の設定をJSONに上書き保存
- **SSH連携**: `--ssh-host`/`--ssh-user`/`--ssh-key`/`--rpicam-cmd` を指定するとPC側1コマンドでPi上のrpicam-rawを自動起動・終了。`--no-ssh` でJSONの設定を無視して手動起動モードに切り替え可能
- **12bitネイティブヒストグラム+CCDF**: raw16空間のデータをそのままヒストグラム表示（1/4サブサンプリングで高速化）
- **ストレッチマーカー（▲△）**: Stack OFF時に自動ストレッチの下限・上限をヒストグラム上に表示
- **WB/ガンマ最適化**: リサイズ後の小さい画像にWB/ガンマを適用（フルサイズ処理を回避して高速化）
- **PNG保存**: `[s]` 押下時にフルサイズ画像にWB/ガンマを適用してから保存
- **FITS保存**: `[f]` でuint16フルサイズ保存（平均化した加算スタック）

**起動例（IMX678 確定設定）:**
```bash
# 送信側 (Raspberry Pi)
rpicam-raw --width 3840 --height 2160 --framerate 0.3 \
           --shutter 600000 --gain 1 -t 0 --listen -o tcp://0.0.0.0:8888

# 受信側 (Windows PC)
python raw_live_stack.py --source tcp://192.168.1.17:8888 \
    --bits 12 --bayer BGGR --raw-width 3856 --raw-height 2180 --stride 5792

# 設定をJSONに保存して次回から簡単起動
python raw_live_stack.py --source tcp://192.168.1.17:8888 \
    --bits 12 --bayer BGGR --raw-width 3856 --raw-height 2180 --stride 5792 \
    --save-config imx678.json
python raw_live_stack.py --config imx678.json
```

**起動例（SSH連携 ワンコマンド起動。imx585.json に ssh_host/rpicam_cmd を記載済みの場合）:**
```bash
# PC側だけで起動（rpicam-rawのSSH起動・終了も自動）
python raw_live_stack.py --config imx585.json

# Pi側で既にrpicam-rawを手動起動済みのとき（SSH自動起動をスキップ）
python raw_live_stack.py --config imx585.json --no-ssh
```

**起動例（12U運用。IMX708 など）:**
```bash
# 送信側 (Raspberry Pi)
rpicam-raw --mode 4608:2592:12:U -t 0 --listen -o tcp://0.0.0.0:8888 --shutter 1000000

# 受信側 (Windows PC)
python raw_live_stack.py --source tcp://192.168.1.63:8888 \
   --wire-format 12u --u12-shift auto --bayer RGGB --raw-width 4608 --raw-height 2592
```

**12U/12P 指定の考え方:**

- `12:P` を使える場合は `--wire-format 12p` を推奨（従来互換で扱いやすい）。
- `12:U` を使う場合は `--wire-format 12u --u12-shift auto` を推奨。
- `10:U` を使う場合は `--wire-format 10u --u10-shift auto` を推奨。
- `--u12-shift` は `auto/0/4` を選択可能。
- `--u10-shift` は `auto/0/6` を選択可能。
- `12:U` でもヒストグラム・閾値判定は有効12bit（上限4095）として扱う。

**10U 運用例（IMX219 など）:**
```bash
# 送信側 (Raspberry Pi) 例
rpicam-raw --mode 1640:1232:10:U -t 0 --listen -o tcp://0.0.0.0:8888

# 受信側 (Windows PC)
python raw_live_stack.py --source tcp://192.168.1.63:8888 \
   --wire-format 10u --u10-shift auto --bayer BGGR --raw-width 1640 --raw-height 1232
```

**操作:**
- `q`: 終了
- `m`: 設定メニュー
- `i`: 情報表示 ON/OFF
- `t`: LiveStack ON/OFF
- `R`: LiveStackリセット
- `s`: PNG保存（WB/ガンマ適用後、フルサイズ、tEXtメタデータ付き）
- `j`: JPEG保存（WB/ガンマ適用後、フルサイズ、EXIF付き）
- `f`: FITS保存（uint16、スタック平均値）
- `r`: NPY保存（16bit RAW）
- `h`: 左右反転 ON/OFF
- `v`: 上下反転 ON/OFF
- `H`: ヒストグラム+CCDF ON/OFF
- `a`: 自動ストレッチ ON/OFF
- `b`: Bayerパターン切り替え
- `n`: 次のフォーマット候補
- `+` / `-`: skip ±1ストライド
- `w`: 白点クリックWB ON/OFF
- `W`: WBリセット
- `g`: ガンマ調整モード（左右キーで ±0.05）
- `G`: ガンマリセット（0.80）
- `d`: ダークフレーム取得（レンズキャップして押す、複数回で加算平均）
- `D`: ダークフレームクリア（終了時に保存されなくなる）
- `C`: 設定をJSONに保存（`--config` 指定時はそのパス、未指定時は `config.json`）

**SSH連携オプション（CLIまたはJSONで指定）:**
- `--ssh-host <IP>`: Pi側ホスト。指定時に起動・終了を自動制御
- `--ssh-user <user>`: SSHユーザー名（デフォルト: `pi`）
- `--ssh-key <path>`: SSH秘密鍵パス（デフォルト: `~/.ssh/id_rsa`）
- `--rpicam-cmd <cmd>`: Pi側で実行するコマンド（`--listen -o tcp://...` は自動補完）
- `--no-ssh`: JSONに `ssh_host` が設定されていても手動起動モードで接続

### 2. Live Stack (`live_stack.py`)
リアルタイムでフレームを加算スタックし、ノイズ軽減と画質向上を実現するプレビューアプリケーション（カメラ切り替え機能付き）。

#### 主な機能
- **LiveStack処理**: フレーム位置合わせと加算によるノイズ軽減
- **設定メニュー**: リアルタイムでの各種設定変更
- **カメラ切り替え**: 複数カメラ（0と1）のリアルタイム切り替え
- **統計オーバーレイ**: ヒストグラムとCCDF（累積）を同時表示し、停止閾値と比率を可視化
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
- `h`: 左右反転トグル
- `v`: 上下反転トグル
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
- **Camera**: 0/1 - 使用するカメラの選択
- **Size**: カメラごとの利用可能解像度一覧（例: `640x480`、`1456x1088`）から選択
- **Gain**: 1.0～8.0（0.5刻み） - カメラの感度調整
- **Exposure**: 10秒～1/2000秒（15段階） - 露出時間、天体撮影向けの長時間露出対応
- **Max Frames**: 1～メモリ計算上限（1刺み） - スタッキングに使用する最大フレーム数。`raw_live_stack.py` では起動時の解像度とOS利用可能メモリから上限を自動計算（起動時に `--max-frames` で指定した値もクランプされる）。メニュー表示は `Stack:N  (MemMax:M)` 形式
- **Stack Mode**: ON/OFF - LiveStackモードの切り替え
- **Info Display**: ON/OFF - 画面上の情報表示切り替え
- **Stop Threshold**: 5～255（5刻み） - 「閾値を超えた画素」を判定する輝度しきい値
- **Stop Ratio(%)**: 1～50（1刻み） - 停止条件となる割合

**操作方法:**
- **上下矢印**: 設定項目選択
- **左右矢印**: 選択した項目の値変更
- **Enter**: 設定を確定して適用（メニューは開いたまま）
- **ESC**: 変更を破棄して閉じる

※ `m` でメニューを閉じた場合も、未確定変更は破棄されます。

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
- **FITS**: RGB対応、NAXIS3を色数として設定。メタデータ（露光時間、ゲイン、スタック数、日時）を付与。オーバーレイテキストを除外した元フレームデータを保存。
- **JPEG**: EXIFデータを付与（撮影日時、露光時間、スタック数など）。文字が含まれないフレームを保存。
- **PNG**: 文字が含まれないフレームを保存。EXIFデータは付与されません。

**操作:**
- `f`: FITS形式で保存
- `j`: JPEG形式で保存
- `p`: PNG形式で保存

#### 注意点
- JPEG保存時にEXIFデータが正しく付与されるよう修正済み。
- PNG保存時にはEXIFデータは付与されませんが、保存時に画面表示の文字が含まれないよう対応済み。
- FITS保存では情報表示テキストが除外され、純粋な画像データのみを保存。

## 実行方法

```bash
# 仮想環境をアクティベート
cd /home/tsuki/MyApps/camera-live
source ../preview/.venv/bin/activate

# RAW TCPストリーム受信 + LiveStack
python raw_live_stack.py --config imx585.json

# SSH自動起動を使わず手動起動モードで接続
python raw_live_stack.py --config imx585.json --no-ssh

# LiveStack機能付きプレビュー（ローカルカメラ）
python3 live_stack.py

# 最大スタックフレーム数を指定して起動
python3 live_stack.py --max-frames 50

# フリップを指定して起動
python3 live_stack.py --flip-h          # 左右反転
python3 live_stack.py --flip-v          # 上下反転
python3 live_stack.py --flip-h --flip-v # 両方

# ヘルプ
python3 live_stack.py --help
```

## リモートソース受信（USBカメラ番号 / TCP URL）

### 受信側（このアプリ）

```bash
# ローカルUSBカメラ（例: 0番）
python3 live_stack.py --source 0

# リモートTCPストリーム
python3 live_stack.py --source tcp://192.168.1.17:8888

# RAW TCPストリーム (rpicam-raw) + LiveStack
python raw_live_stack.py --source tcp://192.168.1.17:8888 --bits 12 --bayer BGGR --raw-width 3856 --raw-height 2180 --stride 5792
```

### 送信側（Raspberry Pi: rpicam-vid の例）

```bash
# 低解像度で軽量配信（確認用）
rpicam-vid -t 0 -n --width 640 --height 400 --framerate 5 --codec h264 --listen -o tcp://0.0.0.0:8888 --bitrate 5000000

# 1080p配信
rpicam-vid -t 0 -n --width 1920 --height 1080 --framerate 5 --codec h264 --listen -o tcp://0.0.0.0:8888 --bitrate 5000000

# 長時間露出 + 高ゲイン
rpicam-vid -t 0 -n --width 1920 --height 1080 --framerate 5 --codec h264 --listen -o tcp://0.0.0.0:8888 --bitrate 5000000 --shutter 1000000 --gain 8
```

補足:
- まずは低解像度（640x400）で疎通確認し、その後1080pへ上げると安定します。
- 送信例は上記3本が最小セットです。必要に応じて `--shutter` / `--gain` を追加調整してください。

### 送信側（Raspberry Pi: rpicam-raw の例）

```bash
rpicam-raw --width 3840 --height 2160 --framerate 0.3 \
   --shutter 600000 --gain 1 -t 0 --listen -o tcp://0.0.0.0:8888
```

## 設定

`common/camera_config.py`で以下を調整可能:
- 解像度: センサーモード一覧から選択（`Size` 項目）
- バッファサイズ
- 露出設定
- ゲイン設定
- カメラ番号（複数カメラ対応）

## 表示スケーリング

- 内部処理・保存は元解像度のまま維持
- プレビュー表示のみ自動縮小
   - 画面サイズ取得成功時: 画面の **85%** 以内に収まるよう縮小
   - 取得失敗時: 高さ **600px** 基準で縮小

## LiveStack機能

- **位置合わせ**: ORB特徴点とホモグラフィー変換
- **スタッキング**: 加算平均によるノイズ軽減
- **最大フレーム数**: デフォルト100フレーム
- **リアルタイム処理**: フレームレートを維持

## 必要なライブラリのインストール

このアプリケーションを使用するには以下のPythonライブラリが必要です。

**インストール方法:**

```bash
pip install -r requirements.txt
```

`picamera2` は Raspberry Pi 側のみ必要です（PC環境では不要）:

```bash
pip install picamera2
```

**仮想環境を使用する場合:**
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

## 変更履歴

#### 2026/08/25 SSH連携によるrpicam-raw自動起動
- **SSH連携実装**: `--ssh-host`/`--ssh-user`/`--ssh-key`/`--rpicam-cmd` でPC側で1コマンドでPiのrpicam-rawを自動起動・停止。鍵認証のみ対応。
- **`--no-ssh`フラグ**: JSONに `ssh_host` があってもコマンドライン指定で手動起動モードに切り替え可能。
- **`--source`自動導出**: `--ssh-host` 定義時に `--source` 未指定なら `tcp://{ssh_host}:8888` を自動設定。
- **EXIF強化**: `--rpicam-cmd` から `--shutter`/`--gain` をパースしてJPEG `ExposureTime`・`UserComment`に埋め込む。
- **`imx585.json`作成**: SSH設定を含むプロファイル。ワンコマンド起動用。

#### 2026/08/25 live_view.py / raw_live_view.py 削除
- **ビューワーファイル削除**: `live_stack.py` / `raw_live_stack.py` の機能が上位互換なため不要になった `live_view.py` と `raw_live_view.py` を削除。
- **ドキュメント整理**: 機能比較表が2スクリプト構成に更新。各ドキュメントから削除済ファイルへの参照を除去。

#### 2026/08/25 raw_live_stack.py: Max Frames メモリ連動クランプ
- **`--max-frames` クランプ追加**: 解像度とOS利用可能メモリ（safety_ratio=0.5）から安全上限を自動計算。`--max-frames` が上限超過時は警告を表示しクランプする。デフォルト値100は変更なし。
- **メニュー表示変更**: Max Framesの表示を `Stack:N  (MemMax:M)` 形式に変更し、現在値と計算上限を同時表示。
- **`common/raw_utils.py` 拡張**: `get_available_memory_bytes()` / `estimate_max_frames_limit()` 追加。Windows（ctypes）およびLinux（/proc/meminfo）に対応。psutil不要。

#### 2026/08/25 raw_live_stack.py: フリップ機能追加・キー割当統一
- **フリップ機能追加**: `raw_live_stack.py` に `--flip-h`/`--flip-v` 引数と `[h]`/`[v]` キートグルを追加（`live_stack.py` と同様の仕様）
- **キー割当変更**: `raw_live_stack.py` のヒストグラム切替キーを `[h]` から `[H]` に変更（`[h]`はフリップ用と衝突するため）

#### 2026/05/19 raw_live_stack.py: スレッド分離・パフォーマンス改善
- **スタックスレッド分離**: `process_stack` をバックグラウンドスレッド化し、重い計算中でもキー入力・表示が止まらないよう改善
- **raw16バッファ化**: スタックバッファを12bit Bayer（uint16）のまま保持し、加算値を固定スケール（÷16）で8bit変換して暗い星を段階的に浮かび上がらせる
- **WB/ガンマ最適化**: リサイズ後の小画像に適用することで処理量を約1/16に削減
- **ヒストグラム高速化**: 1/4サブサンプリング（`[::4, ::4]`）で計算量を1/16に削減
- **ダークフレームキー**: `[d]`（取得・加算平均）/ `[D]`（クリア）を追加
- **PNG保存**: フルサイズにWB/ガンマを適用してから保存するよう修正
- **FITS保存**: uint16でフルサイズ保存（スタック平均値）
- **12bitネイティブヒストグラム**: raw16空間でヒストグラム+CCDF表示、stretch▲△マーカー追加

#### 2026/05/10 リモートソース対応・判定整合改善
- **source対応**: `live_stack.py` に `--source` を追加（カメラ番号または `tcp://...` を指定可能）。
- **Windows起動改善**: `picamera2` などの依存ライブラリを遅延import化し、`--source` 利用時に非Raspberry Pi環境でも起動可能に改善。
- **メニュー操作改善**: `waitKeyEx` ベースに変更し、Windowsで矢印キーが正しく動作するよう修正。
- **停止判定の整合**: 閾値超過フレームを「加算前」に判定して停止する方式へ修正。
- **判定指標の統一**: CCDF表示と停止判定を `max(B,G,R)` で統一し、表示と判定のズレを縮小。
- **ビット幅表示**: 入力フレームの推定ビット幅をオーバーレイ表示。
- **Stop Thresholdの範囲拡張**: 下限を `127` から `5` に変更（5刻み）。

#### 2026/03/14 機能追加
- **コマンドライン引数対応**: `-n` / `--max-frames` オプションで起動時に最大スタックフレーム数を指定可能（デフォルト: 100）
- **フリップ機能追加**: `--flip-h`（左右反転）・`--flip-v`（上下反転）オプションで起動時の反転を指定可能。`h`/`v`キーでトグル切り替えも可能。反転は出力側で即時反映

#### 2026/02/28 統計表示・停止条件調整
- **ヒストグラム追加**: 右上に統計グラフ（ヒストグラム + CCDF）を表示し、閾値/比率の判定を可視化。
- **停止条件の調整機能**: `Stop Threshold`（127～255）と`Stop Ratio(%)`（1～50）を設定メニューに追加。
- **表示/操作の改善**: グラフ可読性を改善（対数正規化・太線化・注記位置調整）、`Enter`は適用のみ（閉じない）に変更。
- **共通部品化**: グラフ描画を `common/hist_overlay.py` に集約し、`live_stack.py` で共有。

#### 2026/02/28 UI・表示改善
- **Size選択の導入**: `FAST/HIGH` 2択から、カメラごとの利用可能解像度一覧選択へ変更。
- **メニュー確定フローの明確化**: 左右キーは値変更のみ、`Enter` で全設定を確定適用、`ESC`/`m` は未確定変更を破棄。
- **表示専用スケーリング**: 内部処理画像と表示画像を分離し、プレビューのみ画面内に収まるよう自動縮小。

#### 2026/02/21 センサー自動対応
- **センサーモード自動検出**: `picamera2.sensor_modes` を使用し、センサーごとの RAW フォーマット/サイズをコード変更なしで自動取得
- **16bitモード自動除外**: ISP統計が壊れる `SRGGB16` 等を自動的に除外し、映像が真っ黒になる問題を恒久的に解決
- **クロップなし出力**: 出力解像度をセンサーのネイティブサイズに合わせることで、libcamera による自動クロップ（上が切れる問題）を解消
- **ダークフレームのサイズ管理改善**: 初期値を `None` にして実フレームのサイズで動的に確保。サイズ不一致によるクラッシュを防止
- **新センサー追加時のコード変更不要**: テーブルへの手動追記が不要になり、接続するだけで自動対応

#### 2025/12/02 バグ修正
- **保存フレームの分離**: `save_frame`と`display_frame`を完全に分離し、保存される画像にテキスト情報が重ねられないように修正
- **Stackモード保存の修正**: `process_stack()`で処理されたスタック済みフレームが正しく保存されるように修正
- **全保存形式の統一**: JPEG、PNG、FITS保存時に全て`save_frame`を使用し、テキストオーバーレイのないクリーンな画像データを保存

#### 2025/07/07 カメラ切り替え・設定メニュー追加
- **カメラ切り替え機能**: 複数カメラ（0と1）のリアルタイム切り替えに対応
- **設定メニューにカメラ選択追加**: 設定メニューからカメラ番号を選択可能
- **安全なカメラ切り替え**: 切り替え失敗時の自動復帰機能
- **ファイル名にカメラ番号付与**: 保存ファイル名にカメラ番号を自動追加
- **メニュー表示の最適化**: カメラ表示の冗長性を解消（Camera: 0）
- **情報表示切り替え機能**: `[i]`キーで画面上のテキスト表示をON/OFF可能
- **設定メニューに情報表示項目追加**: 設定メニューから情報表示のON/OFFを設定可能
- **FITS保存の改善**: オーバーレイテキストを除外し、元のフレームデータのみを保存するよう修正
- **双方向同期**: キー操作と設定メニューの設定値が相互に同期される仕組みを実装
