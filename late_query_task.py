"""Rank-calibrated late-query task and dataset for FridayRepo."""
from __future__ import annotations

import numpy as np

HISTORY_LEN = 18
HISTORY_CHANNELS = 2
N_CLASSES = 4
N_QUERIES = 3

_CLASS_PATTERNS = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 1, 0],
], dtype=int)
_EVENT_TIMES = np.array([2, 6, 10, 14], dtype=int)

def task_matrices():
    """Return three 4x4 target tables of rank 1, 2, and 3 plus the common cheap component.

    The component bases are orthonormal Hadamard directions.  The first component is shared
    by all queries and is also exposed (with noise) as the optional cheap shortcut.
    """
    h = np.array([
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ], dtype=float) / 2.0
    u = h[:, 1:4]
    v = h[:, [2, 3, 1]]
    weights = np.array([2.4, 1.6, 1.0])
    components = [weights[k] * np.outer(u[:, k], v[:, k]) for k in range(3)]
    mats = [components[0], components[0] + components[1], sum(components)]
    return np.stack(mats), components[0].copy()


def _history(rng, family: int, noise: float, jitter: int):
    if not 0 <= family < N_CLASSES:
        raise ValueError("family must be in 0..3")
    x = np.zeros((HISTORY_LEN, HISTORY_CHANNELS), dtype=float)
    for base_t, ch in zip(_EVENT_TIMES, _CLASS_PATTERNS[family]):
        t = int(base_t + rng.integers(-jitter, jitter + 1)) if jitter else int(base_t)
        x[t, ch] += 1.0
    if noise:
        x += rng.normal(0.0, noise, x.shape)
    return x


def make_late_query_dataset(
    seed: int = 0,
    n_per_pair: int = 10,
    history_noise: float = 0.12,
    jitter: int = 1,
    shortcut_noise: float = 0.08,
):
    """Create repeated history pairs followed by an unpredictable late query.

    For each sender/receiver history pair, the *same exact histories and shortcut observation*
    are repeated under all three queries.  The sender therefore cannot infer query identity from
    its input or from correlated sample noise.
    """
    rng = np.random.default_rng(seed)
    mats, shortcut_matrix = task_matrices()
    rows = []
    base_id = 0
    for s in range(N_CLASSES):
        for r in range(N_CLASSES):
            for _ in range(n_per_pair):
                hs = _history(rng, s, history_noise, jitter)
                hr = _history(rng, r, history_noise, jitter)
                shortcut = float(shortcut_matrix[s, r] + rng.normal(0.0, shortcut_noise))
                for q in range(N_QUERIES):
                    rows.append((base_id, hs, hr, s, r, q, shortcut, float(mats[q, s, r])))
                base_id += 1
    rng.shuffle(rows)
    return {
        "base_id": np.array([z[0] for z in rows], dtype=int),
        "sender_history": np.stack([z[1] for z in rows]),
        "receiver_history": np.stack([z[2] for z in rows]),
        "sender_class": np.array([z[3] for z in rows], dtype=int),
        "receiver_class": np.array([z[4] for z in rows], dtype=int),
        "query": np.array([z[5] for z in rows], dtype=int),
        "shortcut": np.array([z[6] for z in rows], dtype=float),
        "target": np.array([z[7] for z in rows], dtype=float),
        "present_event_time": 0.0,
    }


def _best_rank_nrmse(a: np.ndarray, d: int):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    dd = max(0, min(int(d), len(s)))
    recon = (u[:, :dd] * s[:dd]) @ vt[:dd] if dd else np.zeros_like(a)
    return float(np.sqrt(np.mean((a - recon) ** 2)) / (np.std(a) + 1e-12))


def algebraic_rank_reference(max_width: int = 4):
    mats, shortcut = task_matrices()
    joint = np.concatenate(list(mats), axis=1)
    residual = np.concatenate([m - shortcut for m in mats], axis=1)
    return {
        "query_ranks": [int(np.linalg.matrix_rank(m, tol=1e-10)) for m in mats],
        "joint_rank": int(np.linalg.matrix_rank(joint, tol=1e-10)),
        "residual_rank_with_exact_shortcut": int(np.linalg.matrix_rank(residual, tol=1e-10)),
        "without_shortcut": {d: {"nrmse": _best_rank_nrmse(joint, d)} for d in range(max_width + 1)},
        "with_exact_shortcut": {d: {"nrmse": _best_rank_nrmse(residual, d)} for d in range(max_width + 1)},
    }
