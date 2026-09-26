import numpy as np

from generic_dynamics import (
    make_generic_dataset,
    train_generic_rnn,
    evaluate_generic_rnn,
    build_generic_receipt,
)


def test_generic_dataset_hides_role_channels_and_keeps_identical_query():
    data = make_generic_dataset(seed=2, n_per_condition=8, history_noise=0.0, jitter=0)
    assert data["sequence"].shape[-1] == 5
    assert np.allclose(data["sequence"][:, -1, :], data["sequence"][0, -1, :])
    assert data["channel_mixing_is_orthogonal"] is True
    assert np.allclose(data["mixing"].T @ data["mixing"], np.eye(4), atol=1e-10)
    assert np.mean(data["target"] == 1) == 0.5


def test_generic_rnn_can_learn_joint_continuation_without_factorized_modules():
    # Fixed deterministic run known to reach the task solution. The receipt separately
    # reports that vanilla-RNN optimization is not reliable across all seeds.
    train = make_generic_dataset(seed=1002, n_per_condition=512, history_noise=0.18, jitter=2)
    test = make_generic_dataset(seed=1003, n_per_condition=256, history_noise=0.18, jitter=2)
    model = train_generic_rnn(train["sequence"], train["target"], seed=1, hidden_size=12)
    metrics = evaluate_generic_rnn(model, train, test)
    assert metrics["full_accuracy"] > 0.90
    assert metrics["linear_raw_history_accuracy"] < 0.58
    assert metrics["sender_shuffle_accuracy"] < 0.60
    assert metrics["receiver_shuffle_accuracy"] < 0.60


def test_probe_reports_whether_query_creates_or_only_reads_interaction():
    train = make_generic_dataset(seed=1002, n_per_condition=512, history_noise=0.18, jitter=2)
    test = make_generic_dataset(seed=1003, n_per_condition=256, history_noise=0.18, jitter=2)
    model = train_generic_rnn(train["sequence"], train["target"], seed=1, hidden_size=12)
    metrics = evaluate_generic_rnn(model, train, test)
    for key in [
        "training_accuracy",
        "prequery_target_probe_accuracy",
        "postquery_target_probe_accuracy",
        "query_response_target_probe_accuracy",
        "sender_prequery_probe_accuracy",
        "receiver_prequery_probe_accuracy",
        "lowdim_additive_probe_accuracy",
        "lowdim_interaction_probe_accuracy",
        "query_effect_fraction",
    ]:
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert metrics["probe_interpretation"] in {
        "interaction_precomputed_before_query",
        "query_reveals_state_dependent_interaction",
        "distributed_or_ambiguous",
    }


def test_generic_receipt_reports_optimization_failures_instead_of_averaging_them_away():
    receipt = build_generic_receipt(
        seeds=range(2), n_train_per_condition=512, n_test_per_condition=128
    )
    assert receipt["scope"]["biological_claim"] is False
    assert receipt["scope"]["factorized_sender_receiver_modules"] is False
    assert receipt["scope"]["explicit_interaction_feature"] is False
    assert 0 <= receipt["optimization"]["success_count"] <= 2
    assert 0.0 <= receipt["optimization"]["success_fraction"] <= 1.0
    assert "all_runs" in receipt["summary"]
    assert "successful_runs" in receipt["summary"]
