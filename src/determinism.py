"""
Project HQ-IDS-029: Deterministic execution controls
====================================================
Measured fact (see SCIENTIFIC_STATUS.md): with an identical seed and identical
code, ``HybridQNNClassifier`` reaches a different test accuracy when torch runs on
1 thread than on 2 threads (93.3 % vs 93.7 % after 3 epochs at seed 42). Each
setting is bit-reproducible on its own, so "same seed => same result" holds only
for a fixed thread count. Intra-op parallelism changes the floating-point
reduction order inside the complex matmuls of the statevector simulator, and a
few-epoch SGD trajectory amplifies that.

Differences between quantum variants in this repository are 0.1-0.5 accuracy
points, i.e. smaller than that machine-dependent spread. Every experiment must
therefore pin the thread count so results are portable across machines.

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
