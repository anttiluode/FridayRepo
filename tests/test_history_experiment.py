import numpy as np

from history_experiment import (
    make_history_dataset,
    train_history_coupler,
    evaluate_history_coupler,
    build_history_receipt,
)


def test_histories_have_same_marginal_event_counts_but_different_order():
    data = make_history_dataset(seed=3, n_per_condition=4, history_noise=0.0, jitter=0)
    sneg = data["sender_history"][data["sender_family"] == -1][0]
    spos = data["sender_history"][data["sender_family"] == +1][0]
    assert np.allclose(sneg.sum(axis=0), spos.sum(axis=0))
    assert not np.array_equal(sneg, spos)
    assert data["present_event_time"] == 0.0


def test_end_to_end_history_model_learns_joint_continuation_without_state_labels():
    train = make_history_dataset(seed=5, n_per_condition=128, history_noise=0.20, jitter=2)
    test = make_history_dataset(seed=6, n_per_condition=256, history_noise=0.20, jitter=2)
    model = train_history_coupler(
        train["sender_history"], train["receiver_history"], train["target"], seed=5
    )
    metrics = evaluate_history_coupler(model, test, attacker_train=train)
    assert metrics["full_accuracy"] > 0.90
    assert metrics["additive_raw_history_accuracy"] < 0.58
    assert metrics["sender_shuffle_accuracy"] < 0.58
    assert metrics["receiver_shuffle_accuracy"] < 0.58


def test_learning_creates_history_dependent_waveform_and_susceptibility():
    train = make_history_dataset(seed=8, n_per_condition=128, history_noise=0.20, jitter=2)
    model = train_history_coupler(
        train["sender_history"], train["receiver_history"], train["target"], seed=8
    )
    clean = make_history_dataset(seed=9, n_per_condition=1, history_noise=0.0, jitter=0)
    metrics = evaluate_history_coupler(model, clean, attacker_train=train)
    assert metrics["sender_state_decode_accuracy"] > 0.90
    assert metrics["receiver_state_decode_accuracy"] > 0.90
    assert metrics["sender_waveform_distance"] > 0.10
    assert metrics["receiver_susceptibility_distance"] > 0.10
    assert metrics["waveform_area_difference_abs"] < 1e-10
    assert metrics["waveform_peak_index_difference_abs"] == 0.0
    assert metrics["waveform_peak_height_difference_abs"] < 1e-10
    assert metrics["sender_transplant_flip_fraction"] > 0.95
    assert metrics["receiver_transplant_flip_fraction"] > 0.95


def test_history_receipt_separates_emergent_result_from_biological_claim():
    receipt = build_history_receipt(seeds=range(3), n_train_per_condition=96, n_test_per_condition=160)
    assert receipt["scope"]["biological_claim"] is False
    assert receipt["scope"]["explicit_state_labels_used_for_training"] is False
    assert receipt["summary"]["full_accuracy"]["mean"] > 0.88
    assert receipt["summary"]["additive_raw_history_accuracy"]["mean"] < 0.60
