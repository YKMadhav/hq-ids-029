"""Record exactly what produced results/: git commit, package versions, thread pinning,
and SHA-256 of every input dataset. Written to results/RUN_MANIFEST.json."""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from src import determinism  # noqa: E402


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def main() -> None:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "xgboost", "pennylane", "qiskit", "qiskit-aer"):
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:
            packages[name] = "missing"
    data = {}
    for rel in ("data/real_flows.csv", "data/UNSW_NB15_testing-set.csv", "data/UNSW_NB15_training-set.csv"):
        path = os.path.join(REPO_ROOT, rel)
        data[rel] = sha256(path) if os.path.exists(path) else "absent"
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "determinism": determinism.current_configuration(),
        "datasets_sha256": data,
    }
    out = os.path.join(REPO_ROOT, "results", "RUN_MANIFEST.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
