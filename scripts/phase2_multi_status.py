"""Poll all three Phase-2 nodes concurrently and show current status.

Run locally:  python scripts/phase2_multi_status.py [--tail 5]

Reports per node:
  - hostname + uptime
  - tmux sessions matching phase2_*
  - last line of results/phase2_<tag>_status.txt
  - last N lines of results/phase2_<tag>.log (stripping progress-bar carriage returns)
  - any VAL PSNR lines seen so far
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
from dataclasses import dataclass

import paramiko

NODES = {
    "A (small_tight)":  ("172.18.40.103", "teaching", "ds123", "haze_a_small_tight"),  # moved from 119 (disk full)
    "B (large_tight)":  ("172.18.40.103", "teaching", "ds123", "haze_b_large_tight"),  # moved from 115 (disk full)
    "C (large_pseudo)": ("172.18.40.103", "teaching", "ds123", "haze_c_large_pseudo"),
}

REMOTE = "dehazing-compression"


@dataclass
class Report:
    label: str
    host: str
    up: str
    sessions: str
    status: str
    last: str
    val: list[str]


def _run(cli: paramiko.SSHClient, cmd: str, timeout: float = 10) -> str:
    _, out, _ = cli.exec_command(cmd, timeout=timeout)
    return out.read().decode(errors="ignore").rstrip()


def probe(label: str, host: str, user: str, password: str, tag: str, tail: int) -> Report:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(host, username=user, password=password, timeout=6,
                  allow_agent=False, look_for_keys=False)
        up = _run(c, "hostname && uptime")
        sess = _run(c, "tmux ls 2>/dev/null | grep -F phase2_ || echo '(none)'")
        status = _run(c, f"cat {REMOTE}/results/phase2_{tag}_status.txt 2>/dev/null | tail -5 || echo '(no status file)'")
        last = _run(c, f"tail -{tail} {REMOTE}/results/phase2_{tag}.log 2>/dev/null | tr '\\r' '\\n' | tail -{tail}")
        val = _run(c, f"grep -E 'VAL  PSNR' {REMOTE}/results/phase2_{tag}.log 2>/dev/null | tail -5 | sed 's/^ *//'")
        return Report(label, host, up, sess, status, last, val.splitlines())
    except Exception as e:
        return Report(label, host, f"ERROR: {type(e).__name__}: {e}", "", "", "", [])
    finally:
        c.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, default=5, help="Lines of training log to show.")
    args = ap.parse_args()

    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {
            ex.submit(probe, lbl, host, u, pw, tag, args.tail): lbl
            for lbl, (host, u, pw, tag) in NODES.items()
        }
        reports = [f.result() for f in cf.as_completed(futures)]

    order = list(NODES.keys())
    reports.sort(key=lambda r: order.index(r.label))
    for r in reports:
        print("=" * 70)
        print(f"NODE {r.label}   @ {r.host}")
        print("-" * 70)
        print(r.up)
        print(f"\n[tmux]\n{r.sessions}")
        print(f"\n[status.txt]\n{r.status}")
        print(f"\n[log tail]\n{r.last}")
        if r.val:
            print(f"\n[VAL trajectory]")
            for v in r.val:
                print(f"  {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
