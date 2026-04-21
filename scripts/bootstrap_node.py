"""Bootstrap a target teaching node via paramiko.

Runs on the SOURCE node (172.18.40.119) using the adu env python. Transfers
code, dataset zips, teacher checkpoint, and tar-streamed soft labels to the
TARGET. Installs pip deps in the target's adu env.

Usage:
    /home/teaching/miniconda3/envs/adu/bin/python scripts/bootstrap_node.py <target_ip>
"""
from __future__ import annotations

import io
import os
import stat
import sys
import tarfile
import time
from pathlib import Path

import paramiko

HOME = Path("/home/teaching")
PROJECT = HOME / "dehazing-compression"

TARGET_USER = os.environ.get("TARGET_USER", "teaching")
TARGET_PASS = os.environ.get("TARGET_PASSWORD", "ds123")
REMOTE_DIR = os.environ.get("REMOTE_DIR", "dehazing-compression")

# Directories to create remotely.
REMOTE_TREE = [
    f"~/{REMOTE_DIR}",
    f"~/{REMOTE_DIR}/data/RESIDE",
    f"~/{REMOTE_DIR}/experiments/teachers/dehamer/ckpts/indoor",
    f"~/{REMOTE_DIR}/experiments/soft_labels",
    f"~/{REMOTE_DIR}/experiments/students",
    f"~/{REMOTE_DIR}/experiments/ptq",
    f"~/{REMOTE_DIR}/results",
]

# Code paths (relative to PROJECT). Excludes datasets / experiments / .env.
CODE_INCLUDE_DIRS = [
    "configs", "data", "evaluate", "models", "phase1_quantize",
    "phase2_distill", "scripts", "third_party",
]
CODE_INCLUDE_FILES = [
    "CLAUDE.md", "README.md", "RUNS.md", "Update.md",
    "requirements.txt", ".env.example", "gpu",
]
CODE_SKIP_SUBPATTERNS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}

# Heavy assets to copy via SFTP.
BIG_FILES = [
    ("data/RESIDE/ITS.zip",  "data/RESIDE/ITS.zip"),
    ("data/RESIDE/SOTS.zip", "data/RESIDE/SOTS.zip"),
    ("experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt",
     "experiments/teachers/dehamer/ckpts/indoor/PSNR3663_ssim09881.pt"),
]

# Stream as a tar since many small files kill SFTP throughput.
SOFT_LABELS_LOCAL = PROJECT / "experiments" / "soft_labels" / "dehamer_indoor"
SOFT_LABELS_REMOTE_PARENT = f"{REMOTE_DIR}/experiments/soft_labels"


def _connect(host: str) -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, username=TARGET_USER, password=TARGET_PASS, timeout=10,
              allow_agent=False, look_for_keys=False,
              banner_timeout=10, auth_timeout=10)
    return c


def _run(cli: paramiko.SSHClient, cmd: str, timeout: int = 3600) -> tuple[int, str]:
    _stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="ignore")
    err = stderr.read().decode(errors="ignore")
    rc = stdout.channel.recv_exit_status()
    return rc, out + err


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _skip(p: Path) -> bool:
    parts = set(p.parts)
    return any(sp in parts for sp in CODE_SKIP_SUBPATTERNS)


def sftp_put_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    """Create parent dirs then upload. Skips if remote size matches local."""
    parent = os.path.dirname(remote)
    if parent:
        _mkdir_p(sftp, parent)
    try:
        rstat = sftp.stat(remote)
        if rstat.st_size == local.stat().st_size:
            return
    except IOError:
        pass
    sftp.put(str(local), remote)


def _mkdir_p(sftp: paramiko.SFTPClient, path: str) -> None:
    path = path.rstrip("/")
    if not path or path == "/":
        return
    try:
        st = sftp.stat(path)
        if stat.S_ISDIR(st.st_mode):
            return
    except IOError:
        pass
    _mkdir_p(sftp, os.path.dirname(path))
    try:
        sftp.mkdir(path)
    except IOError:
        pass


def sync_code(sftp: paramiko.SFTPClient) -> None:
    remote_root = REMOTE_DIR
    _mkdir_p(sftp, remote_root)
    uploaded = 0
    for d in CODE_INCLUDE_DIRS:
        local_dir = PROJECT / d
        if not local_dir.exists():
            continue
        for lp in local_dir.rglob("*"):
            if lp.is_dir() or _skip(lp):
                continue
            rel = lp.relative_to(PROJECT)
            remote = f"{remote_root}/{rel.as_posix()}"
            sftp_put_file(sftp, lp, remote)
            uploaded += 1
    for f in CODE_INCLUDE_FILES:
        lp = PROJECT / f
        if not lp.exists():
            continue
        sftp_put_file(sftp, lp, f"{remote_root}/{f}")
        uploaded += 1
    _log(f"  uploaded {uploaded} code files")


def put_big_files(sftp: paramiko.SFTPClient) -> None:
    for local_rel, remote_rel in BIG_FILES:
        local = PROJECT / local_rel
        if not local.exists():
            _log(f"  (missing locally; skipping: {local_rel})")
            continue
        remote = f"{REMOTE_DIR}/{remote_rel}"
        try:
            rstat = sftp.stat(remote)
            if rstat.st_size == local.stat().st_size:
                _log(f"  exists same-size (skip): {remote_rel}")
                continue
        except IOError:
            pass
        size_mb = local.stat().st_size / 1e6
        _log(f"  sending {remote_rel} ({size_mb:.1f} MB)...")
        t0 = time.time()
        sftp_put_file(sftp, local, remote)
        dt = time.time() - t0
        _log(f"    done in {dt:.1f}s ({size_mb/max(dt,1e-9):.1f} MB/s)")


def put_soft_labels(cli: paramiko.SSHClient) -> None:
    if not SOFT_LABELS_LOCAL.is_dir():
        _log(f"  soft_labels dir missing locally at {SOFT_LABELS_LOCAL}; skipping.")
        return
    n_files = sum(1 for _ in SOFT_LABELS_LOCAL.iterdir())
    _log(f"  streaming tar of {n_files} soft-label PNGs...")
    _mkdir_p(cli.open_sftp(), f"{REMOTE_DIR}/experiments/soft_labels")
    # Use ssh exec to pipe tar through.
    cmd = f"cd {SOFT_LABELS_REMOTE_PARENT} && tar -xf -"
    stdin, stdout, stderr = cli.exec_command(cmd)
    with tarfile.open(fileobj=stdin, mode="w|") as tar:  # streaming tar
        tar.add(str(SOFT_LABELS_LOCAL), arcname="dehamer_indoor")
    stdin.channel.shutdown_write()
    rc = stdout.channel.recv_exit_status()
    err = stderr.read().decode(errors="ignore")
    if rc != 0:
        _log(f"  tar stream rc={rc}\n{err}")
    else:
        _log("  soft labels tar-streamed OK")


def remote_setup(cli: paramiko.SSHClient) -> None:
    script = r"""
set -e
cd ~/dehazing-compression
if [ ! -d data/RESIDE/ITS-Train ] && [ -f data/RESIDE/ITS.zip ]; then
    (cd data/RESIDE && unzip -q -n ITS.zip)
fi
if [ ! -d data/RESIDE/SOTS-Test ] && [ -f data/RESIDE/SOTS.zip ]; then
    (cd data/RESIDE && unzip -q -n SOTS.zip)
fi
PIP=/home/teaching/miniconda3/envs/adu/bin/pip
PY=/home/teaching/miniconda3/envs/adu/bin/python
"$PIP" install --quiet lmdb scipy scikit-image einops gdown opencv-python-headless 2>&1 | tail -3
"$PY" - <<'EOF'
import sys; sys.path.insert(0, '.')
from models.students.nafnet_student import build_student, count_params
for w in (16, 32):
    m = build_student(width=w)
    n, mM = count_params(m)
    print(f"student width={w}: {n:,} ({mM:.2f}M) OK")
EOF
echo BOOTSTRAP_DONE
"""
    rc, out = _run(cli, script, timeout=1800)
    print(out)
    if rc != 0:
        raise RuntimeError(f"remote setup failed rc={rc}")


def bootstrap(target: str) -> None:
    _log(f"=== bootstrap target={target} ===")
    cli = _connect(target)
    try:
        sftp = cli.open_sftp()
        _log("[1/5] create remote dir layout")
        for d in REMOTE_TREE:
            _mkdir_p(sftp, d.replace("~/", ""))
        _log("[2/5] sync code (sftp)")
        sync_code(sftp)
        _log("[3/5] send dataset zips + teacher ckpt (sftp)")
        put_big_files(sftp)
        _log("[4/5] stream soft labels (tar)")
        put_soft_labels(cli)
        _log("[5/5] remote unzip + pip install + smoke")
        remote_setup(cli)
        _log(f"=== done: {target} ===")
    finally:
        cli.close()


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: bootstrap_node.py <target_ip> [more_ips...]")
        sys.exit(1)
    for t in sys.argv[1:]:
        bootstrap(t)


if __name__ == "__main__":
    main()
