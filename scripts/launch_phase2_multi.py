"""Bootstrap target teaching nodes and launch three Phase-2 training variants
in parallel, each inside its own tmux session on its node.

Runs on the SOURCE node (172.18.40.119) using the adu env python.

Variants:
    A: haze_a_small_tight  @ 172.18.40.119  width=16 GT     L1+0.05*L_feat+0.05*L_perc
    B: haze_b_large_tight  @ 172.18.40.131  width=32 GT     L1+0.05*L_feat+0.05*L_perc
    C: haze_c_large_pseudo @ 172.18.40.137  width=32 pseudo L1+0.05*L_perc

Session name on each node: ``phase2_<tag>``.
Logs: ``results/phase2_<tag>.log`` (tee'd) + ``results/phase2_<tag>_status.txt``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import paramiko

PY = "/home/teaching/miniconda3/envs/adu/bin/python"
REMOTE_DIR = "dehazing-compression"
EPOCHS = int(os.environ.get("EPOCHS", "200"))

PASSWORD = os.environ.get("TARGET_PASSWORD", "ds123")
USER = os.environ.get("TARGET_USER", "teaching")

NODES = [
    {
        "label": "A",
        "ip": "172.18.40.119",
        "tag": "haze_a_small_tight",
        "train_args": "--width 16 --lambda-feat 0.05 --lambda-perc 0.05",
        "bootstrap": False,
    },
    {
        "label": "B",
        "ip": "172.18.40.115",
        "tag": "haze_b_large_tight",
        "train_args": "--width 32 --lambda-feat 0.05 --lambda-perc 0.05",
        "bootstrap": True,
    },
    {
        "label": "C",
        "ip": "172.18.40.103",
        "tag": "haze_c_large_pseudo",
        "train_args": "--width 32 --lambda-feat 0.00 --lambda-perc 0.05 --use-pseudo-as-target",
        "bootstrap": True,
    },
]


def connect(ip: str) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(ip, username=USER, password=PASSWORD, timeout=10,
              allow_agent=False, look_for_keys=False,
              banner_timeout=10, auth_timeout=10)
    return c


def run(cli: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str]:
    _in, out, err = cli.exec_command(cmd, timeout=timeout)
    o = out.read().decode(errors="ignore")
    e = err.read().decode(errors="ignore")
    rc = out.channel.recv_exit_status()
    return rc, o + e


def launch_on(ip: str, tag: str, args: str) -> None:
    print(f"\n=== launch tag={tag} on {ip} ===", flush=True)
    cli = connect(ip)
    try:
        train_cmd = (
            f"cd ~/{REMOTE_DIR} && "
            f"echo STARTED $(date -Iseconds) > results/phase2_{tag}_status.txt && "
            f"{PY} phase2_distill/train.py "
            f"  --tag {tag} --epochs {EPOCHS} "
            f"  --batch 8 --patch 128 --workers 4 "
            f"  --lr-hi 1e-3 --lr-lo 1e-6 "
            f"  --pseudo-dir experiments/soft_labels/dehamer_indoor "
            f"  --val-interval 5 --ckpt-interval 10 "
            f"  {args} "
            f"  2>&1 | tee results/phase2_{tag}.log && "
            f"echo DONE $(date -Iseconds) >> results/phase2_{tag}_status.txt"
        )
        session = f"phase2_{tag}"
        # Kill stale session, open a fresh one, send the command.
        rc, out = run(cli, f"tmux kill-session -t {session} 2>/dev/null; "
                           f"tmux new-session -d -s {session} && "
                           f"tmux send-keys -t {session} {repr(train_cmd)} C-m && "
                           f"echo launched {session} on $(hostname)", timeout=30)
        print(out, flush=True)
        if rc != 0:
            print(f"WARNING launch rc={rc}", flush=True)
    finally:
        cli.close()


def main() -> None:
    # Bootstrap non-source nodes first.
    src_ip = NODES[0]["ip"]
    print(f"source node: {src_ip}", flush=True)
    here = Path(__file__).resolve().parent
    for node in NODES:
        if node["bootstrap"]:
            print(f"\n*** bootstrapping {node['label']} @ {node['ip']} ***", flush=True)
            rc = os.system(f"{PY} {here / 'bootstrap_node.py'} {node['ip']}")
            if rc != 0:
                print(f"bootstrap {node['ip']} failed rc={rc}", file=sys.stderr)
                sys.exit(2)

    # Launch training on each node.
    for node in NODES:
        launch_on(node["ip"], node["tag"], node["train_args"])

    print("\n=== all 3 launched ===", flush=True)
    for node in NODES:
        print(f"  {node['label']}: phase2_{node['tag']}  @ {node['ip']}")
    print("\nattach:  ssh teaching@<ip> ; tmux attach -t phase2_<tag>")
    print("status:  python scripts/phase2_multi_status.py --tail 5  (local)")


if __name__ == "__main__":
    main()
