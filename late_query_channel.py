"""Late-query communication experiment for FridayRepo.

The sender emits a bounded noisy message before query identity is known. A bilinear
rank reference is compared with a generic nonlinear receiver. This is an engineering
falsifier, not a biological model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from late_query_task import (
    HISTORY_LEN, N_CLASSES, N_QUERIES, algebraic_rank_reference,
    make_late_query_dataset, task_matrices,
)
from late_query_model import Communicator, evaluate_communicator, train_communicator

def _stats(rows, key):
    x = np.array([r[key] for r in rows], dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max())}


def build_late_query_receipt(
    seeds: Iterable[int] = range(4), widths=(0, 1, 2, 3, 4), *,
    n_train_per_pair=40, n_test_per_pair=60, history_noise=0.03, jitter=1,
    channel_noise=0.12, shortcut_noise=0.08, epochs=600,
):
    seeds = [int(s) for s in seeds]
    width_rows = {True: {int(w): [] for w in widths}, False: {int(w): [] for w in widths}}
    direct_with_rows, direct_without_rows, erased_rows = [], [], []
    for seed in seeds:
        train = make_late_query_dataset(1000 + 2 * seed, n_train_per_pair, history_noise, jitter, shortcut_noise)
        test = make_late_query_dataset(1001 + 2 * seed, n_test_per_pair, history_noise, jitter, shortcut_noise)
        for use_shortcut in (False, True):
            for w in widths:
                model = train_communicator(train, int(w), seed=seed * 101 + int(w) + (10000 if use_shortcut else 0),
                                           channel_noise=channel_noise, use_shortcut=use_shortcut, epochs=epochs)
                width_rows[use_shortcut][int(w)].append(
                    evaluate_communicator(model, train, test, channel_noise=channel_noise,
                                          use_shortcut=use_shortcut, eval_seed=5000 + seed))
        direct = train_communicator(train, 0, seed=20000 + seed, channel_noise=0.0,
                                    use_shortcut=True, epochs=epochs, direct_access=True)
        direct_with_rows.append(evaluate_communicator(direct, train, test, channel_noise=0.0,
                                                      use_shortcut=True, eval_seed=7000 + seed))
        direct_no = train_communicator(train, 0, seed=21000 + seed, channel_noise=0.0,
                                       use_shortcut=False, epochs=epochs, direct_access=True)
        direct_without_rows.append(evaluate_communicator(direct_no, train, test, channel_noise=0.0,
                                                         use_shortcut=False, eval_seed=7100 + seed))
        maxw = max(int(w) for w in widths)
        erased = train_communicator(train, maxw, seed=30000 + seed, channel_noise=channel_noise,
                                    use_shortcut=True, epochs=epochs, erase_sender=True)
        erased_rows.append(evaluate_communicator(erased, train, test, channel_noise=channel_noise,
                                                 use_shortcut=True, eval_seed=9000 + seed))

    def summarize(bucket):
        out = {}
        for w, rows in bucket.items():
            out[str(w)] = {
                "mean_nrmse": _stats(rows, "nrmse")["mean"],
                "sd_nrmse": _stats(rows, "nrmse")["sd"],
                "mean_sender_shuffle_nrmse": _stats(rows, "sender_shuffle_nrmse")["mean"],
                "mean_used_dimensions": _stats(rows, "used_dimensions")["mean"],
                "mean_effective_rank": _stats(rows, "effective_rank")["mean"],
                "mean_min_class_separation_over_noise": _stats(rows, "min_class_mean_separation_over_noise")["mean"],
                "mean_principal_sender_basis_abs_correlation": _stats(rows, "principal_sender_basis_abs_correlation")["mean"],
                "success_fraction_nrmse_lt_0_25": float(np.mean([r["nrmse"] < 0.25 for r in rows])),
            }
        return out

    return {
        "question": "When a late query is unknowable at send time, what history survives a narrow noisy channel, and how does a cheap shortcut change that pressure?",
        "scope": {
            "biological_claim": False, "query_available_to_sender": False,
            "sender_receiver_are_separate": True, "generic_receiver_is_nonlinear": True,
            "rank_bound_applies_exactly_only_to_bilinear_reference": True,
            "interpretation": "generic scalar messages may beat matrix-rank dimensionality by nonlinear coding; noise and bounded tanh messages make that loophole measurable rather than forbidden",
        },
        "data": {
            "sender_classes": N_CLASSES, "receiver_classes": N_CLASSES, "late_queries": N_QUERIES,
            "history_len": HISTORY_LEN, "history_noise_sd": history_noise, "jitter_samples": jitter,
            "channel_noise_sd": channel_noise, "shortcut_noise_sd": shortcut_noise,
            "train_trials_per_seed": N_CLASSES * N_CLASSES * n_train_per_pair * N_QUERIES,
            "test_trials_per_seed": N_CLASSES * N_CLASSES * n_test_per_pair * N_QUERIES,
        },
        "rank_reference": algebraic_rank_reference(max(max(widths), 4)),
        "generic": {"without_shortcut": summarize(width_rows[False]), "with_shortcut": summarize(width_rows[True])},
        "controls": {
            "direct_access_with_shortcut": {"mean_nrmse": _stats(direct_with_rows, "nrmse")["mean"], "sd_nrmse": _stats(direct_with_rows, "nrmse")["sd"]},
            "direct_access_without_shortcut": {"mean_nrmse": _stats(direct_without_rows, "nrmse")["mean"], "sd_nrmse": _stats(direct_without_rows, "nrmse")["sd"]},
            "sender_erased_with_shortcut": {"mean_nrmse": _stats(erased_rows, "nrmse")["mean"], "sd_nrmse": _stats(erased_rows, "nrmse")["sd"]},
        },
    }


def main():
    receipt = build_late_query_receipt()
    out = Path("results/late_query_receipt.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
