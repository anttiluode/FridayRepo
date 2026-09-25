import numpy as np

from experiment import (
    make_waveform,
    make_susceptibility,
    interaction_response,
    run_experiment,
    learn_state_conditioned_filter,
    build_receipt,
)


def test_sender_state_changes_waveform_but_not_event_time():
    t1, w1 = make_waveform(-1)
    t2, w2 = make_waveform(+1)
    assert np.array_equal(t1, t2)
    assert not np.allclose(w1, w2)


def test_receiver_state_changes_response_to_identical_waveform():
    t, w = make_waveform(+1)
    k_neg = make_susceptibility(t, -1)
    k_pos = make_susceptibility(t, +1)
    y_neg = interaction_response(t, w, k_neg)
    y_pos = interaction_response(t, w, k_pos)
    assert np.sign(y_neg) != np.sign(y_pos)


def test_full_interaction_beats_timestamp_and_attackers():
    receipt = run_experiment(seed=7, n_per_condition=500, noise=0.035)
    assert receipt["full"]["accuracy"] > 0.90
    assert receipt["timestamp_only"]["accuracy"] < 0.60
    assert receipt["waveform_only"]["accuracy"] < 0.60
    assert receipt["clamped_waveform"]["accuracy"] < 0.60
    assert receipt["shuffled_waveform"]["accuracy"] < 0.60
    assert receipt["additive_state_model"]["accuracy"] < 0.60


def test_state_transplants_flip_the_composed_answer():
    receipt = run_experiment(seed=11, n_per_condition=300, noise=0.02)
    assert receipt["sender_transplant"]["flip_fraction"] > 0.95
    assert receipt["receiver_transplant"]["flip_fraction"] > 0.95


def test_waveform_state_is_shape_not_area_energy_or_peak_height():
    _, w_neg = make_waveform(-1)
    _, w_pos = make_waveform(+1)
    carrier = 0.5 * (w_neg + w_pos)
    peak = int(np.argmax(carrier))
    assert abs(w_neg.sum() - w_pos.sum()) < 1e-10
    assert abs(np.dot(w_neg, w_neg) - np.dot(w_pos, w_pos)) < 1e-10
    assert abs(w_neg[peak] - w_pos[peak]) < 1e-10
    assert np.argmax(w_neg) == np.argmax(w_pos) == peak
    assert abs(w_neg.max() - w_pos.max()) < 1e-10


def test_state_conditioned_susceptibility_can_be_learned_from_examples():
    learned = learn_state_conditioned_filter(
        seed=5, n_train_per_condition=200, n_test_per_condition=400, noise=0.12
    )
    assert learned["interaction_accuracy"] > 0.93
    assert learned["additive_accuracy"] < 0.58
    assert learned["kernel_alignment"] > 0.84


def test_receipt_keeps_constructed_and_learned_claims_separate():
    receipt = build_receipt(seeds=range(2), n_per_condition=100, noise=0.12)
    assert receipt["known_answer"]["summary"]["full"]["mean_accuracy"] > 0.90
    assert receipt["learned_susceptibility"]["interaction_accuracy"]["mean"] > 0.90
    assert receipt["learned_susceptibility"]["additive_accuracy"]["mean"] < 0.60
    assert receipt["scope"]["biological_claim"] is False
