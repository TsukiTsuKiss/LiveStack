"""JSON設定ファイルの階層読み込み(sensor/host/meta参照)とSSH経由rpicamコマンド起動の共通処理。
raw_live_stack.py / live_stack.py の両方で利用する。
"""
import json
import os


def load_config(path):
    """設定JSONを再帰的に読み込んでdictを返す。
    meta=観測機材メタ情報、host=ラズパイ個体依存(IP/SSH)、sensor=センサー依存(RAW形式等)を
    子ファイルとして持てる(観測用.json -> host.json -> sensor.json)。
    子ファイルは継承元(優先度低)、自ファイルの明示値が優先度高。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[config] 読み込み失敗: {path}  ({e})")
        return {}
    base_dir = os.path.dirname(os.path.abspath(path))
    merged = {}
    for sub_key, label in (("sensor", "sensor"), ("host", "host"), ("meta", "meta")):
        sub_path = cfg.get(sub_key)
        if not sub_path:
            continue
        sub_abs = os.path.join(base_dir, sub_path)
        sub_cfg = load_config(sub_abs)
        if sub_cfg:
            merged.update({k: v for k, v in sub_cfg.items() if not k.startswith("_")})
            print(f"[config] {label}読み込み: {sub_abs}")
    merged.update(cfg)
    return merged


def ssh_start_rpicam(ssh_host, ssh_user, ssh_key, rpicam_cmd, port, log_name="rpicam"):
    """SSH鍵認証でPiに接続し rpicam-raw/rpicam-vid をバックグラウンド起動する。(client, pid)を返す。"""
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_path = os.path.expanduser(ssh_key)
    # look_for_keys=Trueのままだと指定鍵が拒否された際に~/.ssh/id_rsa等を自動探索し、
    # 暗号化鍵に当たってPasswordRequiredExceptionになるため無効化する
    client.connect(ssh_host, username=ssh_user, key_filename=key_path, look_for_keys=False)

    cmd = rpicam_cmd.strip()
    if "--listen" not in cmd:
        cmd += " --listen"
    if "-o tcp://" not in cmd:
        cmd += f" -o tcp://0.0.0.0:{port}"
    if " -t " not in cmd:
        cmd += " -t 0"

    remote_cmd = f"nohup {cmd} > /tmp/{log_name}.log 2>&1 & echo $!"
    _, stdout, _ = client.exec_command(remote_cmd)
    pid_line = stdout.readline().strip()
    try:
        pid = int(pid_line)
    except ValueError:
        pid = None

    print(f"[SSH] {ssh_host}: {cmd}")
    print(f"[SSH] {log_name} 起動 PID={pid}")
    return client, pid


def ssh_kill_rpicam(client, pid, ssh_host, proc_name="rpicam-raw"):
    """SSH経由でrpicam-raw/rpicam-vidプロセスを停止し接続を閉じる。"""
    if client is None:
        return
    try:
        kill_cmd = f"kill {pid}" if pid else f"pkill -f {proc_name}"
        client.exec_command(f"{kill_cmd} 2>/dev/null; true")
        print(f"[SSH] {ssh_host}: {proc_name} 停止 (PID={pid})")
    except Exception as e:
        print(f"[SSH] プロセス停止エラー: {e}")
    finally:
        try:
            client.close()
        except Exception:
            pass
