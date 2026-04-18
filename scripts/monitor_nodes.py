"""Probe all cluster nodes in servers.csv for CPU/RAM/GPU usage.

Concurrent (ThreadPoolExecutor) paramiko SSH probe. Outputs a sorted table.
Optional JSON output with `--json` for programmatic consumption (picker).

Safety: this script ONLY reads system metrics. It never kills or starts
processes on remote nodes.

Usage:
    python scripts/monitor_nodes.py                # pretty-printed report
    python scripts/monitor_nodes.py --json status.json  # + machine-readable
    python scripts/monitor_nodes.py --timeout 4    # per-node connect timeout
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    import paramiko
except ImportError:
    sys.exit("paramiko not installed. Run: pip install paramiko")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "servers.csv"

CMD_CPU = (
    "top -bn1 | grep 'Cpu(s)' | "
    "sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | "
    "awk '{print 100 - $1}'"
)
CMD_RAM = "free -m | awk 'NR==2{printf \"%.2f\", $3*100/$2 }'"
CMD_GPU = (
    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits "
    "2>/dev/null | awk '{printf \"%.2f\\n\", $1*100/$3}'"
)
CMD_LOAD = "cat /proc/loadavg | awk '{print $1}'"
CMD_CORES = "nproc"


@dataclass
class NodeStatus:
    user: str
    ip: str
    reachable: bool
    cpu_pct: Optional[float] = None
    ram_pct: Optional[float] = None
    gpu_pct: Optional[float] = None         # first GPU, or max across GPUs
    gpu_count: int = 0
    load1: Optional[float] = None           # 1-min load average
    cores: Optional[int] = None             # nproc
    error: Optional[str] = None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s.strip())
    except Exception:
        return None


def probe(user: str, ip: str, password: str, timeout: float = 5.0) -> NodeStatus:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    status = NodeStatus(user=user, ip=ip, reachable=False)
    try:
        client.connect(
            hostname=ip, username=user, password=password,
            timeout=timeout, auth_timeout=timeout, banner_timeout=timeout,
            allow_agent=False, look_for_keys=False,
        )
        status.reachable = True

        def _run(cmd: str) -> str:
            _, out, _ = client.exec_command(cmd, timeout=timeout)
            return out.read().decode().strip()

        status.cpu_pct = _to_float(_run(CMD_CPU))
        status.ram_pct = _to_float(_run(CMD_RAM))
        status.load1 = _to_float(_run(CMD_LOAD))
        status.cores = int(_run(CMD_CORES)) if _run(CMD_CORES).isdigit() else None

        gpu_raw = _run(CMD_GPU)
        if gpu_raw:
            vals = [_to_float(line) for line in gpu_raw.splitlines() if line.strip()]
            vals = [v for v in vals if v is not None]
            if vals:
                status.gpu_count = len(vals)
                status.gpu_pct = max(vals)  # worst-loaded GPU on this node
    except paramiko.AuthenticationException:
        status.error = "auth_failed"
    except Exception as e:  # timeout, network, etc.
        status.error = type(e).__name__
    finally:
        client.close()
    return status


def fitness(s: NodeStatus) -> float:
    """Lower is better. Unreachable nodes sort last."""
    if not s.reachable:
        return 1e9
    cpu = s.cpu_pct or 0.0
    ram = s.ram_pct or 0.0
    load_per_core = (s.load1 or 0.0) / max(s.cores or 1, 1)
    gpu = s.gpu_pct or 0.0
    # Weight: load-per-core matters most for CPU-bound PTQ; GPU util matters when using GPU.
    return 2.0 * load_per_core * 100 + cpu + 0.3 * ram + 0.3 * gpu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--json", type=Path, default=None,
                    help="Also dump machine-readable JSON here.")
    args = ap.parse_args()

    assert args.csv.exists(), f"missing {args.csv}"

    rows = []
    with args.csv.open() as f:
        for r in csv.DictReader(f):
            rows.append((r["Username"].strip(), r["IP_Address"].strip(), r["Password"].strip()))

    statuses: list[NodeStatus] = []
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(probe, u, ip, pw, args.timeout) for (u, ip, pw) in rows]
        for fut in cf.as_completed(futures):
            statuses.append(fut.result())

    statuses.sort(key=fitness)

    # Pretty print
    header = f"{'user':<14} {'ip':<15} {'cpu%':>6} {'ram%':>6} {'load1':>6} {'cores':>5} {'gpu%':>6} {'gpus':>4} fit"
    print(header)
    print("-" * len(header))
    for s in statuses:
        if not s.reachable:
            print(f"{s.user:<14} {s.ip:<15} {'-':>6} {'-':>6} {'-':>6} {'-':>5} {'-':>6} {'-':>4} (unreachable: {s.error})")
            continue
        print(
            f"{s.user:<14} {s.ip:<15} "
            f"{(s.cpu_pct or 0):>6.1f} "
            f"{(s.ram_pct or 0):>6.1f} "
            f"{(s.load1 or 0):>6.2f} "
            f"{(s.cores or 0):>5d} "
            f"{(s.gpu_pct or 0):>6.1f} "
            f"{s.gpu_count:>4d} "
            f"{fitness(s):.1f}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            [asdict(s) for s in statuses], indent=2,
        ))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
