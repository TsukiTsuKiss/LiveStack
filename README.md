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
│   ├── ui_menus.py       # 設定メニュー・操作メニュー・画面録画（両アプリ共通）
│   └── raw_utils.py      # RAW受信・デコード・WB/ガンマ共通処理
├── demo/
│   └── make_demo.py      # デモ動画の自動生成（キー操作の自動入力 + 画面録画）
└── README.md
```

## デモ動画の作成（SNS紹介用）

`demo/make_demo.py` が、`raw_live_stack.py` にキー操作を自動で送り、アプリ自身の画面録画（`[y]`）でAVI（MJPG）を作ります。info・ヒストグラム・設定メニュー・操作メニューがそのまま映り、押したキーと内容が下部のキャプションに出ます（キャプションは録画にだけ重ねる。アプリ本体は変更しない）。カメラ不要で、録画済みのSERファイルを再生して作ります。

```bash
python demo/make_demo.py --ser 20261013_225204.ser                 # 全シーン
python demo/make_demo.py --ser xxx.ser --scenes live,menu           # シーンを指定
python demo/make_demo.py --ser xxx.ser --mp4                        # ffmpegでmp4(H.264, crf18)も作る
python demo/make_demo.py --ser xxx.ser --wb 1.6,1.0,1.4             # ホワイトバランス B,G,R（デフォルトは1.0,1.0,1.0=なし）
```

| シーン | 内容 |
|---|---|
| `live` | ライブビュー: info / ヒストグラム / ストレッチ / 反転 / Bayer切替 |
| `menu` | 設定メニュー: Stop系・Max Frames・ON/OFF項目・Stack Method・Stack Mode、`m`/`c` の切替 |
| `stack` | スタック: Sum の飽和停止と、比較明・比較暗・中央値・シグマクリップ・シグマクリップ加算の比較 |
| `control` | 操作メニュー: 保存、SER録画、リセット、JSON保存 |
| `player` | SERプレーヤー: 一時停止・コマ送り・ジャンプ・ループモード |

- 出力は `demo/out/<シーン名>.avi`（`.gitignore` の `*.avi` / `*.mp4` で除外済み）。保存系の操作で作られる画像・SER・設定JSONは一時ディレクトリに出して、終了時に削除します。
- 実行中はアプリのウィンドウが開きます（表示環境が必要）。シーンの長さは30〜80秒程度で、実時間で録画するため全シーンの生成に5分ほどかかります。
- 実ダークを使うなら `--dark dark.fits`（コピーを使うので元のファイルは変更しない。向きは `--dark-flip auto` で自動判定）。指定しない場合、SERにはダークが入っていないため、先頭フレームの黒レベルから擬似ダーク（一定値）を作って `Dark Sub` を有効にします（スタック表示の黒浮きを抑えるため。実機のダークの代わりのデモ用）。画面に映るソース名はファイル名のみです。
- 操作の流れは `demo/make_demo.py` の各 `scene_*` 関数（時刻・キー・キャプション）で編集できます。メニューの項目順を変えたら、先頭の項目番号の定数も合わせてください。

---

## 関連ドキュメント

- [仕様.md](仕様.md): wire-format / 判定仕様 / UI仕様
- [動作実績.md](動作実績.md): 機材・解像度ごとの動作確認コマンド集

## アプリケーション

### 1. RAW Live Stack (`raw_live_stack.py`)
`rpicam-raw` の生TCPストリームを受信しながらリアルタイムにフレームをスタックするRAW専用アプリ。

**主な特徴:**
- **raw16バッファ**: バッファを12bit Bayer（uint16）のまま保持し、加算スタック後に固定スケール（÷16）で8bit変換して表示することで暗い星が枚数増加とともに浮かび上がる
- **スタックスレッド分離**: `process_stack` をバックグラウンドスレッドで実行し、メインスレッドはキー入力・フレーム受信・表示を独立して継続するため、重いスタック計算中でも操作が止まらない
- **ダークフレーム**: `[d]` でレンズキャップ状態のフレームを取得・平均化。`[D]` でクリア。終了時に `dark.fits` へ自動保存（そのセッションで `d` により取得・更新したときだけ。読み込んだだけのダークは書き換えない）、次回起動時に自動読み込み。ダーク中の突出画素をホットピクセルとして検出し、スタック時に周囲の同色画素で補間（`--hot-sigma`、デフォルト 10、0で無効）
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

保存ファイル名は `yyyymmdd_HHMMSS.拡張子`（ローカル時刻。SERは録画開始時刻）。`raw_live_stack.py` では静止画・録画とも `--rec-dir`（既定 `recordings/`）以下に保存する。UTCはファイル内に記録される（FITS: `DATE-OBS`、PNG: `DateTimeUTC`、JPEG: EXIF UserComment、SER: ヘッダー`DateTimeUTC`とトレーラー）。NPYには日時情報なし。- `h`: 左右反転 ON/OFF
- `v`: 上下反転 ON/OFF
- `H`: ヒストグラム+CCDF ON/OFF
- `a`: 自動ストレッチ ON/OFF
- `b`: Bayerパターン切り替え
- `n`: 次のフォーマット候補
- `+` / `-`: skip ±1ストライド
- `w`: 白点クリックWB ON/OFF
- `W` / `u`: WBリセット
- `g`: ガンマ調整モード（左右キーで ±0.05）
- `G`: ガンマリセット（0.80）
- `d`: ダークフレーム取得（レンズキャップして押す、複数回で加算平均）
- `D`: ダークフレームクリア（終了時に保存されなくなる）
- `C`: 設定をJSONに保存（`--config` 指定時はそのパス、未指定時は `config.json`）
- `c`: 操作メニュー開閉（上下キーで選択、Enterで実行、ESC/`c`で閉じる。左右キーは使わない）。画像保存（PNG/JPEG/FITS/NPY）、SER録画、画面録画、Zoom、白点クリックWB、WB/ガンマ/スタックのリセット、ダーク取得/クリア、JSON保存をメニューから実行できる（左右/上下反転は設定メニューの `Flip H`/`Flip V`）（Linuxで使えない大文字キーの代替）
- `y`: 画面録画AVI開始/停止（info・ヒストグラム・各メニュー・ズームPiP等の表示込みでプレビューを `yyyymmdd_HHMMSS_screen.avi` に保存。MJPG、`--screen-fps` で指定、デフォルト15fps。録画中は左上に `SCR` 表示（録画映像には写らない）。SNS投稿用は `ffmpeg -i in.avi -c:v libx264 -pix_fmt yuv420p out.mp4` で変換）
- `S` / `x`: SER録画開始/停止トグル。`recordings/` 下の日時フォルダに `*_raw.ser`（カメラRAW）と `*_stack.ser`（スタック済み。スタックON中のみ）、使ったダーク・設定JSONを保存（`--no-rec-raw` / `--no-rec-stack` / `--no-rec-folder` で無効化可）
- Linux の Qt 版 OpenCV では Shift が反映されないため、大文字キー（`S`/`W` 等）は使えない。`S` は `x`、`W` は `u` で代用する。`A`/`E`/`H` は設定メニューの `Align`/`Dark Sub`/`Histogram` から切替できる。他の大文字キー機能も `c` の操作メニューから実行できる
- `z`: フォーカスアシストズーム ON/OFF（ON中にクリック/ドラッグで拡大エリアを選択）

**最大フレーム数オプション:**
- `--max-frames N`: リングバッファの最大フレーム数（デフォルト 100）。`--max-frames max`（または `0`）で、解像度と空きメモリから見積もった上限（空きメモリの50%）まで使う。メモリの大きいPCで長時間スタックしたいときに便利（例: 3840x2160で空き約120GBなら約1960フレーム）。JSON設定では `"max_frames": "max"` も可

**SER反転オプション:**
- `--ser-flip`: スタックSER（`*_stack.ser`）にも `--flip-h`/`--flip-v`（`[h]`/`[v]`）の状態を反映（録画開始時に固定、ColorIDも更新）。既定ではSERは生Bayerのまま反転しない。理由は `仕様.md` の「SER録画機能」を参照

**SER入力テストモード（カメラ不要）:**
- `--source file.ser`: `raw_live_stack.py` でSERファイルをループ再生して入力する。メニューやキー操作の確認用。幅・高さ・Bayerはヘッダーから取得（`--raw-width` / `--stride` / `--wire-format` / `--crop-*` / SSHは無視）
- `--bits`: 有効ビット深度（SERは16bit容器のため未指定時は12）。`--ser-lsb` を付けない限り上位詰め(MSB)として読む
- `--ser-fps`: 再生フレームレート（デフォルト 5）
- `--ser-loop-mode reset|continuous|off`: 末尾での動作（デフォルト `reset`）。`reset`=先頭に戻りスタックをリセット、`continuous`=リセットせず再生とスタックを続ける、`off`=末尾で停止。録画したフレームが足りずオーバーフローまで加算できないとき、`continuous` で同じ映像を繰り返し加算できる（リングバッファが `Max Frames` を超える分は古いフレームから置き換わるので、`--max-frames` を十分に大きくしておく。位置合わせONだと、末尾→先頭でずれが大きい場合に検出失敗として外挿される）。実行中は `[l]` で切替
- `[+]`/`[-]`（skip）は無効。スタックオンで録画したSERはダーク減算済みの加算値なので、`[E]` でダーク減算をOFFにして使う
- プレーヤー操作: `[space]` 一時停止/再生、`[,]`/`[.]` 1コマ戻し/送り（一時停止になる）、`[`/`]` 先頭/末尾へジャンプ（`Home`/`End` も可）、`[l]` ループ切替（リセットあり → リセットなし（継続） → ループなし の順）。戻る・ジャンプ・ループ（リセットあり）時はスタックをリセット
- PNG/JPEG/NPY/FITS の保存ファイル名にフレーム番号が入る（例: `20261012_101500_f0015.png`）

**SSH連携オプション（CLIまたはJSONで指定）:**
- `--ssh-host <IP>`: Pi側ホスト。指定時に起動・終了を自動制御
- `--ssh-user <user>`: SSHユーザー名（デフォルト: `pi`）
- `--ssh-key <path>`: SSH秘密鍵パス（デフォルト: `~/.ssh/id_rsa`）
- `--rpicam-cmd <cmd>`: Pi側で実行するコマンド（`--listen -o tcp://...` は自動補完）
- `--no-ssh`: JSONに `ssh_host` が設定されていても手動起動モードで接続

**SSH連携のセットアップ（初回のみ）:**
接続は鍵認証のみです（パスワード認証は使えません）。パスワードで手入力のSSHができても、鍵を作って登録するまではこのアプリから接続できません。

1. 既存の鍵がないか確認する（あれば手順2は実行しない。上書きすると古い鍵が使えなくなる）:
   ```bash
   ls ~/.ssh/id_rsa ~/.ssh/id_rsa.pub
   ```
2. PCで鍵を作る（アプリの既定値 `~/.ssh/id_rsa` になる。パスフレーズは空にする。アプリから自動で使うため）:
   ```bash
   ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa
   ```
3. 公開鍵をPiに登録する（パスワードを1回聞かれる）:
   ```bash
   ssh-copy-id -i ~/.ssh/id_rsa.pub <ユーザー名>@<PiのIP>
   ```
   Windows には標準で `ssh-copy-id` がないため、PowerShell では次のようにする:
   ```powershell
   type $env:USERPROFILE\.ssh\id_rsa.pub | ssh <ユーザー名>@<PiのIP> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
   ```
4. パスワードなしで入れることを確認する（`ok` と表示されれば成功）:
   ```bash
   ssh -i ~/.ssh/id_rsa <ユーザー名>@<PiのIP> echo ok
   ```
5. JSON（またはオプション）に設定する:
   ```json
   "ssh_host": "192.168.1.14",
   "ssh_user": "<ユーザー名>",
   "ssh_key": "~/.ssh/id_rsa",
   "rpicam_cmd": "rpicam-raw --width 3840 --height 2160 --framerate 0.3 -t 0 --shutter 5000000 --gain 0 --nopreview"
   ```
6. `python raw_live_stack.py --config` で起動する（`--source` は `ssh_host` から自動設定される）。

**つまずきやすい点:**
- `ssh_key` には**秘密鍵**（`.pub` の付かないファイル）を指定する。`~/.ssh/known_hosts` は接続先の記録で鍵ではないため、`not a valid OPENSSH private key file` になる。
- `ssh_host` が `null` のままだと、`rpicam_cmd` があってもSSHは使われず、手動起動済みのrpicam-rawへ接続しに行く。起動していなければ `Connection refused` になる。`[SSH]` の行が出ているかで見分けられる。
- 手動でrpicam-rawを起動した場合は `--no-ssh` を付ける。
- 起動に失敗したときはPi側の `/tmp/rpicam_raw.log` にrpicam-rawの出力が残る。
- 鍵の種類は自由（`ed25519` なら `ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519` で作り、`ssh_key` を `~/.ssh/id_ed25519` にする）。Pi側のOpenSSHがRSAを受け付けない設定の場合は `ed25519` を使う。ファイル名の違いだけで、置き場所（Windowsも `~/.ssh/`）は同じ。
- 鍵ファイルの中身（秘密鍵）は公開しない。リポジトリには入れない（`~/.ssh` はリポジトリの外）。`ssh_key` のパス文字列だけなら公開して問題ない。

**ズームフォーカスアシストオプション（CLIまたはJSONで指定）:**
- `--zoom-pip-size <px>`: PiP表示サイズ（デフォルト: `640`）
- `--zoom-click-size <px>`: クリック時に切り出す元解像度ピクセル数（デフォルト: `128`、倍率 = pip_size ÷ click_size）
- `--zoom-corner <tl|tr|bl|br>`: PiP配置隅（`tl`=左上 / `tr`=右上 / `bl`=左下 / `br`=右下、デフォルト: `br`）

### 2. Live Stack (`live_stack.py`)
リアルタイムでフレームを加算スタックし、ノイズ軽減と画質向上を実現するプレビューアプリケーション（カメラ切り替え機能付き）。

#### 主な機能
- **LiveStack処理**: フレーム位置合わせと加算によるノイズ軽減
- **設定メニュー**: リアルタイムでの各種設定変更
- **カメラ切り替え**: 複数カメラ（0と1）のリアルタイム切り替え
- **統計オーバーレイ**: ヒストグラムとCCDF（累積）を同時表示し、停止閾値と比率を可視化
- **多様な保存形式**: FITS、JPEG、PNG対応

#### 操作方法

`raw_live_stack.py` と同じ操作感です（メニューは共通モジュール `common/ui_menus.py`）。

**基本操作:**
- `q`: 終了
- `m`: 設定メニュー開閉（上下で項目選択、左右で値変更・即時反映。ON/OFFは左右どちらでも反転）
- `c`: 操作メニュー開閉（上下で選択、Enterで実行、ESC/`c`で閉じる。`m`/`c` で相互に切替）
- `i`: 情報表示ON/OFF
- `H`: ヒストグラム/CCDF表示ON/OFF（設定メニューの `Histogram` でも切替）
- `t`: LiveStack ON/OFF切り替え
- `r`: スタックリセット
- `h` / `v`: 左右 / 上下反転トグル
- `y`: 画面録画AVI開始/停止（info・ヒストグラム・メニュー込み。`--screen-fps` でfps指定、デフォルト15）
- `k`: カメラ切り替え（0 ↔ 1、picamera2直結モードのみ。旧 `c`）
- `d`: ダークフレーム取得（picamera2直結モードのみ）

**保存操作:**
- `s` / `p`: PNG形式で保存
- `j`: JPEG形式で保存
- `f`: FITS形式で保存

**操作メニュー（`c`）の項目:** Save PNG / Save JPEG / Save FITS / Screen Rec (AVI) / Stack Reset。picamera2直結モードでは Dark Capture / Switch Camera も表示。Linux の Qt 版 OpenCV では大文字キーが使えないため、メニューから実行できます。

**設定メニュー（`m`）の項目:** Stop Threshold / Stop Ratio(%) / Max Frames / Info Display / Histogram / Flip H / Flip V / Stack Mode（picamera2直結モードでは Camera / Size / Gain / Exposure も表示）。

**従来のキー操作（picamera2直結モード）:**
- `+`/`-`: ゲイン調整
- `1`〜`0`: シャッター速度変更

#### 設定メニュー機能
上下キーで項目選択、左右キーで値変更（ON/OFFは左右どちらでも反転）。`raw_live_stack.py` では即時反映、ESC/`m` で閉じます。保存・リセット等の実行は別の操作メニュー（`c`、上下+Enter）を使います。設定メニューと操作メニューは `m` / `c` で相互に切り替えられます。

`[m]`キーで設定メニューを開くと、以下の設定をリアルタイムで変更できます：

**設定項目:**
- **Camera**: 0/1 - 使用するカメラの選択
- **Size**: カメラごとの利用可能解像度一覧（例: `640x480`、`1456x1088`）から選択
- **Gain**: 1.0～8.0（0.5刻み） - カメラの感度調整
- **Exposure**: 10秒～1/2000秒（15段階） - 露出時間、天体撮影向けの長時間露出対応
- **Max Frames**: 1～メモリ計算上限（1刺み） - スタッキングに使用する最大フレーム数。`raw_live_stack.py` では起動時の解像度とOS利用可能メモリから上限を自動計算（起動時に `--max-frames` で指定した値もクランプされる）。メニュー表示は `Stack:N  (MemMax:M)` 形式
- **Stack Mode**: ON/OFF - LiveStackモードの切り替え
- **Info Display**: ON/OFF - 画面上の情報表示切り替え
- **Histogram**: ON/OFF - ヒストグラム/CCDF表示の切り替え（`[H]`キーでも切替）。`raw_live_stack.py` のみ
- **Flip H / Flip V**: ON/OFF - 左右/上下反転（`[h]`/`[v]`キーでも切替）。`raw_live_stack.py` のみ
- **Rec Raw / Rec Stack / Rec Folder**: ON/OFF - `[x]` 録画でRAW SER・スタックSER・日時フォルダ（ダーク/JSONのコピー）を保存するか（`--no-rec-raw` / `--no-rec-stack` / `--no-rec-folder` と同じ。次回の録画開始から有効）。`raw_live_stack.py` のみ
- **SER Flip**: ON/OFF - 左右/上下反転をスタックSERにだけ反映する（RAW SERは常に無反転）（`--ser-flip` と同じ。録画開始時に固定されるため次回の録画から有効）。画面側の `Flip H`/`Flip V` がONのときだけ効き、両方OFFのときはメニューに `inactive` と表示される。`raw_live_stack.py` のみ
- **Stop Threshold**: 5～255（5刻み） - 「閾値を超えた画素」を判定する輝度しきい値
- **Stop Ratio(%)**: 1～50（1刻み） - 停止条件となる割合
- **Align Size(%)**: 10～100（10刻み） - 位置合わせ検出窓の大きさ（画像幅/高さに対する割合、デフォルト 50）。`raw_live_stack.py` のみ適用
- **Align**: ON/OFF - 位置合わせの有効/無効（`[A]`キーでも切替、`--no-align` で無効起動）。切替時にスタックをリセット。`raw_live_stack.py` のみ適用
- **Stack Method**: Sum / Lighten / Darken / Median / Sigma Clip - スタック方式（`--stack-method` で起動時指定）。`Sum` は加算（飽和で自動停止）。それ以外は最新 `Max Frames` 枚（位置合わせ後）から画素ごとに計算する。`Lighten`=比較明（星の軌跡・流星・衛星。位置合わせOFF推奨）、`Darken`=比較暗（衛星・飛行機の筋の除去）、`Median`=中央値、`Sigma Clip`=シグマクリップ平均（外れ値除去しつつノイズ低減、`--clip-sigma` で棄却幅）、`Sigma Sum`=シグマクリップ加算（棄却後の平均×枚数。加算と同じスケールで、保存時の精度とダイナミックレンジが保たれる。16bitを超える分は飽和）。`Sigma Sum` は `Sum` と同じ停止条件（`Stop Threshold`/`Stop Ratio(%)`）で枚数を決める。シグマクリップは枚数が少ないと外れ値を除けない（1つの外れ値の最大偏差は (N-1)/√N σ のため、棄却幅2.5ではN≥9、2.0ではN≥6が目安）。少ない枚数では `--clip-sigma` を小さくする。`Sum` と `Sigma Sum` 以外では `Stop Threshold`/`Stop Ratio(%)` は使われない。切替時もバッファは保持され、次の更新で再計算される。`raw_live_stack.py` のみ
- **Dark Flip**: None / H / V / HV / Auto - ダークの反転。追加した理由: 反転して録画したSER（`SER Flip` / `--ser-flip`）は、元の向きで撮ったダークと画素が合わず、ダーク減算が効かない（背景ノイズがむしろ増える）ため、SERを録り直さずにダーク側を映像の向きに合わせられるようにした。向きは表示の性質なので、普段は `SER Flip` を使わず生のまま録画して表示側（`Flip H` / `Flip V`）で揃えるのが推奨で、その場合 `Dark Flip` は `None` のままでよい。ダークと映像の向きが違うとき（`SER Flip` / `--ser-flip` で反転して録画したSERなど）は、`--dark-flip none|h|v|hv|auto` または設定メニューの `Dark Flip` でダークを反転して使う（ダーク本体は元の向きのまま保持し、引き算とホットピクセル検出にだけ反転を適用）。`auto` は最初のフレームで、ダークと生フレームのホットピクセルの一致数が最も多い向きを選ぶ（4K で約1秒。一致が少なく判定できないときは `none` のまま警告する）。ヘッダーの `BAYER` 表記は当てにならない（終了時の保存で書き換わることがある）ため、画素の一致で判定する。ダークを新たに `d` で取得すると、既存ダークに反転を焼き込んで平均し、Dark Flip は `none` に戻る。 `raw_live_stack.py` のみ
- **Dark Sub**: ON/OFF - ダーク減算とホットピクセル補間の有効/無効。スタックOFFでも、ダークが取得済み（または読込済み）で `Dark Sub` がONなら、表示・統計・ヒストグラム・PNG/JPEG/FITS保存はダーク減算後の映像になる（情報欄に `Dark: ON/OFF (n枚)` を表示）。`[d]`（ダーク取得）・NPY保存・SER録画は常に減算前の生フレームを使う（ダークの二重減算を避け、後からスタックし直せるようにするため）。（`[E]`キーでも切替、`--no-dark-sub` で無効起動）。ダーク減算済みのSERを入力する場合などに使用。切替時にスタックをリセット。`raw_live_stack.py` のみ適用

**操作方法:**
- **上下矢印**: 設定項目選択
- **左右矢印**: 選択した項目の値変更
- **Enter**: 設定を確定して適用（メニューは開いたまま）
- **ESC**: 変更を破棄して閉じる

※ `m` でメニューを閉じた場合も、未確定変更は破棄されます。

#### カメラ切り替え機能
**2つの切り替え方法:**
1. **設定メニュー**: `[m]`キーでメニューを開き、Camera項目で左右矢印キーで切り替え
2. **直接キー**: `[k]`キーで即座にカメラ0と1を切り替え（操作メニュー `c` の `Switch Camera` でも可）

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
- `s` / `p`: PNG形式で保存

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

- **位置合わせ**: `raw_live_stack.py` は `phaseCorrelate` による平行移動検出（基準フレームに対する絶対位置、Bayer色面ごとのサブピクセル移動）。詳細は仕様.md を参照
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

#### 2026/10/16 raw_live_stack.py: `[x]` 録画を「RAW SER＋スタックSER＋日時フォルダ」に変更
- 静止画（PNG/JPEG/FITS/NPY）も `--rec-dir` 直下に日時名で保存（1ファイルなので日時フォルダは作らない）。
- 録画1回ごとに、録画用フォルダ（`--rec-dir`、既定 `recordings/`。`.gitignore` に登録済み）の下に日時フォルダ（`yyyymmdd_HHMMSS/`）を作り、`yyyymmdd_HHMMSS_raw.ser`（カメラRAW・無反転。後処理用）と `yyyymmdd_HHMMSS_stack.ser`（スタック済みフレーム。すぐ公開する用。スタックON中のみ）の2本を保存。使ったダーク（`dark.fits`）と設定JSON（`--config`/`--meta` のファイル。未指定なら実効設定を `config.json` として保存）も同じフォルダに入れる。
- `--no-rec-raw` / `--no-rec-stack` / `--no-rec-folder`（設定メニュー `Rec Raw` / `Rec Stack` / `Rec Folder`、config は `no_rec_raw` 等）で個別に無効化。`--no-rec-folder` では `--rec-dir` 直下に `*_raw.ser` / `*_stack.ser` だけを保存（ダーク・JSONはコピーしない）。
- `--ser-flip` はスタックSERにのみ効く（RAW SERは常に無反転）ため、RAW SER＋ダークの組み合わせでは `Dark Flip` は不要。

#### 2026/10/15 raw_live_stack.py: Dark Flip（ダークの向き合わせ）
- **`--dark-flip none|h|v|hv|auto` / 設定メニュー `Dark Flip` 追加**: 反転して録画したSERに、元の向きで撮ったダークを合わせて引けるようにした。`auto` はホットピクセルの一致で向きを判定（実データで左右反転を1秒で判定、背景ノイズの標準偏差が約10〜20%低下。向き違いのままだと逆に増える）。
- **ダークの自動保存を修正**: 終了時にダークを保存するのは、そのセッションで取得・更新したときだけにした（SER再生で読み込んだだけのダークの `BAYER` ラベルが書き換わるのを防ぐ）。
- **`demo/make_demo.py`**: `--dark` で実ダークを使えるようにした。

#### 2026/10/14 SSH連携の鍵手順を id_rsa（既定値）に合わせて修正
- セットアップ手順を `ssh-keygen -t rsa -b 4096 -N "" -f ~/.ssh/id_rsa` に変更（既存鍵の確認、Windowsでの公開鍵登録、ed25519を使う場合の注記を追加）。

#### 2026/10/14 SSH連携のセットアップ手順を追記
- 鍵の作成・Piへの登録・確認・設定の手順と、つまずきやすい点（`ssh_key` に `known_hosts` を指定、`ssh_host` が `null`、`--no-ssh`）を「SSH連携のセットアップ」に追記。

#### 2026/10/13 デモ動画の自動生成
- **`demo/make_demo.py` 追加**: SERファイルを再生しながらキー操作を自動入力し、画面録画（`[y]`）でシーン別のデモAVI（`--mp4` でH.264も）を作る。メニュー操作はキャプション付きで映る。

#### 2026/10/13 raw_live_stack.py: SER再生のループモード追加
- **`--ser-loop-mode` / `[l]` 3段階化**: 末尾で先頭に戻るときスタックをリセットしない `continuous` を追加（フレーム不足でオーバーフローまで加算したいとき用）。

#### 2026/10/13 raw_live_stack.py: スタックOFFでもダーク減算を表示に適用
- `Dark Sub` ON のとき、スタックOFFの表示・保存（PNG/JPEG/FITS）にもダーク減算とホットピクセル補間を適用。`[d]`・NPY・SER録画は減算前の生フレームのまま。情報欄に `Dark:` 行を追加。

#### 2026/10/13 raw_live_stack.py: スタック方式の追加
- **`--stack-method` / 設定メニュー `Stack Method` 追加**: 加算（従来どおり）に加えて、比較明・比較暗・中央値・シグマクリップ平均・シグマクリップ加算（`sigmasum`）を選択可能。Sum以外は最新 `Max Frames` 枚を使い、重さは目安として 4K で シグマクリップ N=30 が約3秒、中央値 N=30 が約4秒（N=100 で約40秒）。ワーカースレッドで処理するため表示は止まらず、更新頻度が下がる。`--clip-sigma` を追加。

#### 2026/10/13 live_stack.py: raw_live_stack.py と操作性を統一
- **UI部品を `common/ui_menus.py` に共通化**: `SettingsMenu` / `ControlMenu` / `ScreenRecorder` を両アプリで共有。
- **`live_stack.py` に追加**: 操作メニュー（`c`）、画面録画（`y` / `--screen-fps`）、設定メニューの `Histogram` / `Flip H` / `Flip V`、左右キーの即時反映、メニューの項目並び（Stop系を先頭・Stack Modeを最下位）。
- **キー割当の変更**: カメラ切り替えを `c` → `k` に、`s` を JPEG → PNG に変更（`raw_live_stack.py` と統一。`p` は引き続きPNG）。

#### 2026/10/13 raw_live_stack.py: Linux(Qt版OpenCV)対応
- **`--max-frames max` / `0` 追加**: メモリ見積もり上限まで自動で使う。
- **右矢印でSER録画が始まる問題を修正**: 矢印キーのコードが `S` と衝突していた。矢印・Home/Endを専用判定に分離（`common/display_utils.py` の `decode_key()`）。
- **TCP受信でSER録画フレーム数が0のままになる不具合を修正**。
- **Shift不可への対処**: SER録画に `[x]`、WBリセットに `[u]` を追加。
- **画面録画追加**: `[y]`（操作メニュー `Screen Rec (AVI)`）でオーバーレイ込みのプレビューをAVI（MJPG）に録画。`--screen-fps` を追加。
- **設定メニューに `SER Flip` 追加**、設定メニューと操作メニューを `m`/`c` で相互に切替可能に。
- **操作メニュー追加**: `[c]` で開き、保存・SER録画・Zoom・各種リセット・JSON保存をメニューから実行（大文字キーが使えないLinux向け）。
- **設定メニュー**: `Histogram` 項目を追加。`Stop Threshold`/`Stop Ratio(%)` を先頭、`Stack Mode` を最下位に移動。
- **WBウィンドウ**: ツールバー非表示・サイズ固定・スライダーラベルのフォント表示を修正。

#### 2026/10/09 raw_live_stack.py: 位置合わせ精度の改善とホットピクセル補間
- **基準フレーム方式**: フレーム間のずれの累積をやめ、基準（キーフレーム）に対する絶対位置で位置合わせ。誤差の蓄積がなくなり、星が流れなくなった。
- **サブピクセル移動**: Bayer の4色面ごとにサブピクセルで移動（従来は偶数pxへ丸め）。
- **`phaseCorrelate` の入力破壊を回避**: 窓関数が入力に書き込まれるためコピーを渡すよう修正。
- **異常値除外**: 予測位置から外れた検出は捨て、直前の移動量で外挿。
- **ホットピクセル補間**: ダークから検出して周囲の同色画素で置換。`--hot-sigma` を追加。

#### 2026/10/04 raw_live_stack.py: 位置合わせを phaseCorrelate 方式に変更
- **不具合修正**: 同サイズ画像で `matchTemplate` を呼んでいたためずれが常に (0,0) となり、位置合わせが未動作だった。
- **方式変更**: 直前フレームとのずれを `phaseCorrelate`（中央窓・2x2ビニング）で検出して保持し、最新から遡って累積して平行移動加算。
- **`Align Size(%)` 追加**: 設定メニュー・`--align-size`（デフォルト 50）で検出窓の大きさを指定。`[perf]` ログに `align: XXms` を表示。

#### 2026/08/26 raw_live_stack.py: フォーカスアシストズーム機能追加
- **`[z]`キーでフォーカスアシストズーム ON/OFF**: ON中はクリックまたはドラッグでPiP表示エリアを選択。
- **元解像度クロップ**: 縮小済みプレビューでなく `display_frame`（縮小前のフル解像度）から切り出し、WB/ガンマを適用後に `pip_size` へ拡大表示。
- **クリック**: `zoom_click_size` 原解像度画素を中心から切り出し（デフォルト: 128px、`pip_size=640` なら 5倍相当）。
- **ドラッグ**: ドラッグ範囲を元解像度座標に逆マップして切り出し。
- **設定化**: `zoom_pip_size`/`zoom_click_size`/`zoom_corner`（`tl`/`tr`/`bl`/`br`）をJSONまたはCLIで指定可能。

#### 2026/08/25 live_view.py / raw_live_view.py 削除
- **ビューワーファイル削除**: `live_stack.py` / `raw_live_stack.py` の機能が上位互換なため不要になった `live_view.py` と `raw_live_view.py` を削除。
- **ドキュメント整理**: 機能比較表が2スクリプト構成に更新。各ドキュメントから削除済ファイルへの参照を除去。

#### 2026/08/25 SSH連携によるrpicam-raw自動起動
- **SSH連携実装**: `--ssh-host`/`--ssh-user`/`--ssh-key`/`--rpicam-cmd` でPC側で1コマンドでPiのrpicam-rawを自動起動・停止。鍵認証のみ対応。
- **`--no-ssh`フラグ**: JSONに `ssh_host` があってもコマンドライン指定で手動起動モードに切り替え可能。
- **`--source`自動導出**: `--ssh-host` 定義時に `--source` 未指定なら `tcp://{ssh_host}:8888` を自動設定。
- **EXIF強化**: `--rpicam-cmd` から `--shutter`/`--gain` をパースしてJPEG `ExposureTime`・`UserComment`に埋め込む。
- **`imx585.json`作成**: SSH設定を含むプロファイル。ワンコマンド起動用。

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
