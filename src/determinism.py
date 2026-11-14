"""
Project HQ-IDS-029: Deterministic execution controls
====================================================
Measured fact (results/thread_sensitivity.json, scripts/check_thread_sensitivity.py): with
identical seed and code, 1 vs 2 torch threads give DIFFERENT floating-point outputs (the sum of
predicted probabilities differs in the 5th significant digit) while each thread count is
bit-reproducible on its own. In the current de-duplicated protocol that did not flip a single
test prediction after 3 epochs, but an earlier, non-de-duplicated run at the same seed did change
accuracy (93.3 % vs 93.7 %), so thread count can steer a few-epoch SGD trajectory. Differences
between quantum variants here are small (0.1-1 accuracy points), so every experiment pins the
thread count to make results portable bit-for-bit.

Importing this module applies the pinned configuration. Set
``HQIDS_ALLOW_MULTITHREAD=1`` to opt out (results are then machine-dependent).
"""

import os

PINNED_THREADS = 1

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_configured = False
_blas_limits = None


def configure_determinism(num_threads: int = PINNED_THREADS) -> dict:
    """Pin BLAS / torch intra-op threads. Idempotent; returns what was applied."""
    global _configured, _blas_limits

    if os.environ.get("HQIDS_ALLOW_MULTITHREAD") == "1":
        return {"pinned": False, "reason": "HQIDS_ALLOW_MULTITHREAD=1"}
    if _configured:
        return {"pinned": True, "threads": num_threads, "already_configured": True}

    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(num_threads)

    try:  # numpy / scipy / scikit-learn BLAS pools that were loaded before this call
        from threadpoolctl import threadpool_limits

        _blas_limits = threadpool_limits(limits=num_threads)
    except Exception:  # pragma: no cover - threadpoolctl ships with scikit-learn
        _blas_limits = None

    import torch

    torch.set_num_threads(num_threads)
    try:
        torch.set_num_interop_threads(num_threads)
    except RuntimeError:  # can only be set once, before any parallel work started
        pass
    torch.use_deterministic_algorithms(True, warn_only=True)

    _configured = True
    return {"pinned": True, "threads": num_threads, "already_configured": False}


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. Call immediately BEFORE constructing a model:
    layer weights are drawn at construction, so seeding only inside ``fit`` (as the
    anomaly models do) leaves the initial weights dependent on unseeded global state."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def current_configuration() -> dict:
    """Snapshot used by the run manifest."""
    import torch

    return {
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_multithread_override": os.environ.get("HQIDS_ALLOW_MULTITHREAD") == "1",
    }


configure_determinism()
