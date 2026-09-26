import numpy as np

from late_query_channel import (
    task_matrices,
    make_late_query_dataset,
    algebraic_rank_reference,
    train_communicator,
    evaluate_communicator,
    build_late_query_receipt,
)


def test_task_family_has_declared_ranks_and_late_query_changes_answer():
    mats, shortcut = task_matrices()
    assert [np.linalg.matrix_rank(m, tol=1e-10) for m in mats] == [1, 2, 3]
    concat = np.concatenate(mats, axis=1)
    assert np.linalg.matrix_rank(concat, tol=1e-10) == 3
    residual = np.concatenate([m - shortcut for m in mats], axis=1)
    assert np.linalg.matrix_rank(residual, tol=1e-10) == 2
    assert np.any(np.abs(mats[0] - mats[2]) > 1e-8)


def test_dataset_reuses_same_histories_before_unpredictable_query():
    data = make_late_query_dataset(seed=3, n_per_pair=2, history_noise=0.0, jitter=0, shortcut_noise=0.0)
    groups = data["base_id"]
    for base_id in np.unique(groups):
        idx = np.flatnonzero(groups == base_id)
        assert len(idx) == 3
        assert set(data["query"][idx].tolist()) == {0, 1, 2}
        assert np.allclose(data["sender_history"][idx], data["sender_history"][idx[0]])
        assert np.allclose(data["receiver_history"][idx], data["receiver_history"][idx[0]])
        assert np.allclose(data["shortcut"][idx], data["shortcut"][idx[0]])


def test_rank_reference_requires_three_dims_without_shortcut_two_with_shortcut():
    ref = algebraic_rank_reference()
    assert ref["joint_rank"] == 3
    assert ref["residual_rank_with_exact_shortcut"] == 2
    assert ref["without_shortcut"][2]["nrmse"] > 1e-6
    assert ref["without_shortcut"][3]["nrmse"] < 1e-10
    assert ref["with_exact_shortcut"][1]["nrmse"] > 1e-6
    assert ref["with_exact_shortcut"][2]["nrmse"] < 1e-10


def test_generic_sender_cannot_see_late_query_and_benefits_from_old_history():
    train = make_late_query_dataset(seed=10, n_per_pair=40, history_noise=0.03, jitter=1, shortcut_noise=0.08)
    test = make_late_query_dataset(seed=11, n_per_pair=40, history_noise=0.03, jitter=1, shortcut_noise=0.08)
    model = train_communicator(train, channel_width=2, seed=4, channel_noise=0.12, use_shortcut=True, epochs=600)
    metrics = evaluate_communicator(model, train, test, channel_noise=0.12, use_shortcut=True, eval_seed=99)
    assert metrics["sender_query_leak_max_abs"] < 1e-12
    assert metrics["nrmse"] < 0.42
    assert metrics["sender_shuffle_nrmse"] > metrics["nrmse"] + 0.08
    assert metrics["principal_sender_basis_abs_correlation"] > 0.90


def test_receipt_contains_width_sweep_shortcut_and_controls():
    receipt = build_late_query_receipt(
        seeds=range(1), widths=(0, 1, 2, 3), n_train_per_pair=30, n_test_per_pair=30,
        history_noise=0.03, jitter=1, channel_noise=0.12, shortcut_noise=0.08, epochs=500,
    )
    assert receipt["scope"]["query_available_to_sender"] is False
    assert receipt["rank_reference"]["joint_rank"] == 3
    assert receipt["rank_reference"]["residual_rank_with_exact_shortcut"] == 2
    assert set(receipt["generic"]["with_shortcut"]) == {"0", "1", "2", "3"}
    assert set(receipt["generic"]["without_shortcut"]) == {"0", "1", "2", "3"}
    assert receipt["controls"]["direct_access_with_shortcut"]["mean_nrmse"] < 0.40
    assert "direct_access_without_shortcut" in receipt["controls"]
    assert "success_fraction_nrmse_lt_0_25" in receipt["generic"]["with_shortcut"]["2"]
    assert receipt["controls"]["sender_erased_with_shortcut"]["mean_nrmse"] > receipt["generic"]["with_shortcut"]["2"]["mean_nrmse"]
