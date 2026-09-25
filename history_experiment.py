"""Learned-history follow-up for FridayRepo.

This removes the explicit sender/receiver state labels from the model. Each side receives an
actual two-channel temporal history. A learned temporal encoder turns that history into a
resident scalar state. Sender state modulates a learned waveform shape; receiver state
modulates a learned temporal susceptibility. The only training signal is continuation error.

The architecture still supplies the *possibility* of state-dependent coupling. The experiment
asks whether the useful histories, emitted shape, and receiver susceptibility can all be
learned from continuation loss without handing the model the hidden +/- state labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


HISTORY_LEN = 24
HISTORY_CHANNELS = 2
WAVEFORM_SAMPLES = 96


def _noisy_history(
    rng: np.random.Generator,
    family: int,
    history_noise: float,
    jitter: int,
    history_len: int = HISTORY_LEN,
) -> np.ndarray:
    x = np.zeros((history_len, HISTORY_CHANNELS), dtype=float)
    early = 6 + int(rng.integers(-jitter, jitter + 1))
    late = 17 + int(rng.integers(-jitter, jitter + 1))
    if family == -1:
        x[early, 0] = 1.0
        x[late, 1] = 1.0
    else:
        x[early, 1] = 1.0
        x[late, 0] = 1.0
    if history_noise:
        x += rng.normal(0.0, history_noise, size=x.shape)
    return x


def make_history_dataset(
    seed: int = 0,
    n_per_condition: int = 128,
    history_noise: float = 0.20,
    jitter: int = 2,
) -> Dict[str, np.ndarray | float]:
    """Balanced four-way dataset. Hidden family labels are diagnostics only.

    Target is +1 when sender and receiver histories belong to the same temporal family,
    otherwise -1. The model never receives ``sender_family`` or ``receiver_family`` during
    fitting. The present event always occurs at t=0 after both histories have ended.
    """
    rng = np.random.default_rng(seed)
    sender, receiver, target, sfam, rfam = [], [], [], [], []
    for s in (-1, +1):
        for r in (-1, +1):
            for _ in range(n_per_condition):
                sender.append(_noisy_history(rng, s, history_noise, jitter))
                receiver.append(_noisy_history(rng, r, history_noise, jitter))
                target.append(s * r)
                sfam.append(s)
                rfam.append(r)
    order = rng.permutation(len(target))
    return {
        "sender_history": np.asarray(sender)[order],
        "receiver_history": np.asarray(receiver)[order],
        "target": np.asarray(target, dtype=int)[order],
        "sender_family": np.asarray(sfam, dtype=int)[order],
        "receiver_family": np.asarray(rfam, dtype=int)[order],
        "history_time": np.linspace(-1.0, -1.0 / HISTORY_LEN, HISTORY_LEN),
        "present_event_time": 0.0,
    }


def _carrier_and_projector(n: int = WAVEFORM_SAMPLES):
    """Common AP-like carrier and a generic shape subspace, with no planted useful mode."""
    t = np.linspace(0.0, 0.12, n, endpoint=False)
    carrier = np.exp(-0.5 * ((t - 0.030) / 0.0055) ** 2)
    carrier -= 0.28 * np.exp(-0.5 * ((t - 0.050) / 0.013) ** 2)
    peak = int(np.argmax(carrier))

    # Learned deviations cannot alter total area, overlap the common carrier, or write
    # directly onto the carrier peak. This removes easy amplitude/area/peak shortcuts while
    # leaving a large generic temporal shape space. No task-specific shape direction is given.
    constraints = [carrier, np.ones_like(carrier)]
    for idx in range(max(0, peak - 2), min(n, peak + 3)):
        axis = np.zeros_like(carrier)
        axis[idx] = 1.0
        constraints.append(axis)
    C = np.column_stack(constraints)
    projector = np.eye(n) - C @ np.linalg.pinv(C)
    return t, carrier, projector


@dataclass
class HistoryCoupler:
    sender_encoder: np.ndarray
    receiver_encoder: np.ndarray
    sender_bias: float
    receiver_bias: float
    sender_shape: np.ndarray
    receiver_kernel: np.ndarray
    output_gain: float
    output_bias: float
    carrier: np.ndarray
    waveform_time: np.ndarray

    def _flat(self, histories: np.ndarray) -> np.ndarray:
        h = np.asarray(histories, dtype=float)
        if h.ndim != 3 or h.shape[1:] != (HISTORY_LEN, HISTORY_CHANNELS):
            raise ValueError("histories must have shape (n, 24, 2)")
        return h.reshape(len(h), -1)

    def sender_state(self, histories: np.ndarray) -> np.ndarray:
        x = self._flat(histories)
        return np.tanh(x @ self.sender_encoder + self.sender_bias)

    def receiver_state(self, histories: np.ndarray) -> np.ndarray:
        x = self._flat(histories)
        return np.tanh(x @ self.receiver_encoder + self.receiver_bias)

    def waveforms(self, histories: np.ndarray) -> np.ndarray:
        h = self.sender_state(histories)
        return self.carrier[None, :] + h[:, None] * self.sender_shape[None, :]

    def susceptibilities(self, histories: np.ndarray) -> np.ndarray:
        h = self.receiver_state(histories)
        return h[:, None] * self.receiver_kernel[None, :]

    def score(self, sender_history: np.ndarray, receiver_history: np.ndarray) -> np.ndarray:
        waves = self.waveforms(sender_history)
        kernels = self.susceptibilities(receiver_history)
        # Carrier is common to every trial and constrained orthogonal to receiver_kernel.
        interaction = np.sum((waves - self.carrier[None, :]) * kernels, axis=1)
        return self.output_gain * interaction + self.output_bias

    def predict(self, sender_history: np.ndarray, receiver_history: np.ndarray) -> np.ndarray:
        return np.where(self.score(sender_history, receiver_history) >= 0.0, +1, -1)


def train_history_coupler(
    sender_history: np.ndarray,
    receiver_history: np.ndarray,
    target: np.ndarray,
    *,
    seed: int = 0,
    steps: int = 1500,
    learning_rate: float = 0.02,
    l2: float = 1e-4,
) -> HistoryCoupler:
    """Train all useful states/shapes from continuation error only.

    No hidden family/state labels enter this function. The factorization into sender history
    -> waveform and receiver history -> susceptibility is an architectural prior; its actual
    temporal encoders and temporal shapes are learned end-to-end.
    """
    xs = np.asarray(sender_history, dtype=float).reshape(len(sender_history), -1)
    xr = np.asarray(receiver_history, dtype=float).reshape(len(receiver_history), -1)
    y = np.asarray(target, dtype=float)
    if xs.shape != xr.shape or xs.shape[1] != HISTORY_LEN * HISTORY_CHANNELS:
        raise ValueError("sender/receiver histories must both have shape (n, 24, 2)")
    if set(np.unique(y)) - {-1.0, 1.0}:
        raise ValueError("target must contain only -1/+1")
    y01 = (y + 1.0) / 2.0

    rng = np.random.default_rng(seed)
    _, carrier, projector = _carrier_and_projector()
    params = {
        "sender_encoder": rng.normal(0.0, 0.15, xs.shape[1]),
        "receiver_encoder": rng.normal(0.0, 0.15, xr.shape[1]),
        "sender_bias": np.array(0.0),
        "receiver_bias": np.array(0.0),
        "sender_shape": projector @ rng.normal(0.0, 0.15, WAVEFORM_SAMPLES),
        "receiver_kernel": projector @ rng.normal(0.0, 0.15, WAVEFORM_SAMPLES),
        "output_gain": np.array(1.0),
        "output_bias": np.array(0.0),
    }
    # Keep the emitted waveform perturbation small so the common event peak and timing stay fixed.
    params["sender_shape"] *= 0.24 / np.linalg.norm(params["sender_shape"])
    params["receiver_kernel"] /= np.linalg.norm(params["receiver_kernel"])

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    for step in range(1, steps + 1):
        hs = np.tanh(xs @ params["sender_encoder"] + params["sender_bias"])
        hr = np.tanh(xr @ params["receiver_encoder"] + params["receiver_bias"])
        coupling = float(params["sender_shape"] @ params["receiver_kernel"])
        z = params["output_gain"] * hs * hr * coupling + params["output_bias"]
        prob = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        dz = (prob - y01) / len(y01)

        grad: Dict[str, np.ndarray] = {}
        grad["output_bias"] = np.array(dz.sum())
        dhs = dz * params["output_gain"] * hr * coupling
        dhr = dz * params["output_gain"] * hs * coupling
        ds_pre = dhs * (1.0 - hs * hs)
        dr_pre = dhr * (1.0 - hr * hr)
        grad["sender_encoder"] = xs.T @ ds_pre + l2 * params["sender_encoder"]
        grad["receiver_encoder"] = xr.T @ dr_pre + l2 * params["receiver_encoder"]
        grad["sender_bias"] = np.array(ds_pre.sum())
        grad["receiver_bias"] = np.array(dr_pre.sum())
        grad["output_gain"] = np.array(np.sum(dz * hs * hr * coupling))
        dc = float(np.sum(dz * params["output_gain"] * hs * hr))
        grad["sender_shape"] = projector @ (
            dc * params["receiver_kernel"] + l2 * params["sender_shape"]
        )
        grad["receiver_kernel"] = projector @ (
            dc * params["sender_shape"] + l2 * params["receiver_kernel"]
        )

        for key in params:
            m[key] = 0.9 * m[key] + 0.1 * grad[key]
            v[key] = 0.999 * v[key] + 0.001 * (grad[key] * grad[key])
            mhat = m[key] / (1.0 - 0.9**step)
            vhat = v[key] / (1.0 - 0.999**step)
            params[key] -= learning_rate * mhat / (np.sqrt(vhat) + 1e-8)

        # Keep both learned temporal objects in the scalar-cue-free shape subspace.
        params["sender_shape"] = projector @ params["sender_shape"]
        params["receiver_kernel"] = projector @ params["receiver_kernel"]
        params["sender_shape"] *= 0.24 / max(np.linalg.norm(params["sender_shape"]), 1e-12)
        params["receiver_kernel"] /= max(np.linalg.norm(params["receiver_kernel"]), 1e-12)

    t, carrier, _ = _carrier_and_projector()
    return HistoryCoupler(
        sender_encoder=params["sender_encoder"],
        receiver_encoder=params["receiver_encoder"],
        sender_bias=float(params["sender_bias"]),
        receiver_bias=float(params["receiver_bias"]),
        sender_shape=params["sender_shape"],
        receiver_kernel=params["receiver_kernel"],
        output_gain=float(params["output_gain"]),
        output_bias=float(params["output_bias"]),
        carrier=carrier,
        waveform_time=t,
    )


def _accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(target)))


def _best_sign_decode(state: np.ndarray, family: np.ndarray) -> float:
    pred = np.where(state >= 0.0, +1, -1)
    raw = _accuracy(pred, family)
    return max(raw, _accuracy(-pred, family))


def _fit_additive_raw_history(train: Dict[str, np.ndarray | float], l2: float = 1.0):
    xs = np.asarray(train["sender_history"]).reshape(len(train["target"]), -1)
    xr = np.asarray(train["receiver_history"]).reshape(len(train["target"]), -1)
    y = np.asarray(train["target"], dtype=float)
    X = np.column_stack([np.ones(len(y)), xs, xr])
    reg = l2 * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def _predict_additive_raw_history(beta: np.ndarray, data: Dict[str, np.ndarray | float]):
    xs = np.asarray(data["sender_history"]).reshape(len(data["target"]), -1)
    xr = np.asarray(data["receiver_history"]).reshape(len(data["target"]), -1)
    X = np.column_stack([np.ones(len(xs)), xs, xr])
    return np.where(X @ beta >= 0.0, +1, -1)


def evaluate_history_coupler(
    model: HistoryCoupler,
    data: Dict[str, np.ndarray | float],
    *,
    attacker_train: Dict[str, np.ndarray | float],
) -> Dict[str, float]:
    """Evaluate the learned factorization and cheap falsifiers.

    Hidden family labels are used here only for post-hoc diagnostics/transplants, never for
    fitting the HistoryCoupler.
    """
    sh = np.asarray(data["sender_history"])
    rh = np.asarray(data["receiver_history"])
    y = np.asarray(data["target"], dtype=int)
    sf = np.asarray(data["sender_family"], dtype=int)
    rf = np.asarray(data["receiver_family"], dtype=int)

    pred = model.predict(sh, rh)
    beta = _fit_additive_raw_history(attacker_train)
    add_pred = _predict_additive_raw_history(beta, data)

    rng = np.random.default_rng(880301)
    sender_shuffle = model.predict(sh[rng.permutation(len(sh))], rh)
    receiver_shuffle = model.predict(sh, rh[rng.permutation(len(rh))])

    # Clamp emitted state-dependent shape to the common carrier.
    clamped_score = np.zeros(len(y)) + model.output_bias
    clamped_pred = np.where(clamped_score >= 0.0, +1, -1)

    # Clamp receiver resident state to its training-set mean.
    train_rstate = model.receiver_state(np.asarray(attacker_train["receiver_history"]))
    fixed_hr = float(train_rstate.mean())
    waves = model.waveforms(sh)
    fixed_kernel = fixed_hr * model.receiver_kernel
    fixed_interaction = (waves - model.carrier[None, :]) @ fixed_kernel
    fixed_score = model.output_gain * fixed_interaction + model.output_bias
    fixed_pred = np.where(fixed_score >= 0.0, +1, -1)

    hs = model.sender_state(sh)
    hr = model.receiver_state(rh)
    wave = model.waveforms(sh)
    sus = model.susceptibilities(rh)

    def class_mean(arr, labels, value):
        return np.asarray(arr)[labels == value].mean(axis=0)

    wneg, wpos = class_mean(wave, sf, -1), class_mean(wave, sf, +1)
    kneg, kpos = class_mean(sus, rf, -1), class_mean(sus, rf, +1)

    # Post-hoc causal transplant on family-average histories. This does not enter training.
    sproto = {f: np.asarray(sh[sf == f]).mean(axis=0) for f in (-1, +1)}
    rproto = {f: np.asarray(rh[rf == f]).mean(axis=0) for f in (-1, +1)}
    sender_flips, receiver_flips = [], []
    for s in (-1, +1):
        for r in (-1, +1):
            a = model.predict(sproto[s][None], rproto[r][None])[0]
            b = model.predict(sproto[-s][None], rproto[r][None])[0]
            c = model.predict(sproto[s][None], rproto[-r][None])[0]
            sender_flips.append(a == -b)
            receiver_flips.append(a == -c)

    qcos = float(
        abs(model.sender_shape @ model.receiver_kernel)
        / (np.linalg.norm(model.sender_shape) * np.linalg.norm(model.receiver_kernel))
    )
    carrier_peak = int(np.argmax(model.carrier))

    return {
        "full_accuracy": _accuracy(pred, y),
        "timestamp_only_accuracy": 0.5,
        "additive_raw_history_accuracy": _accuracy(add_pred, y),
        "sender_shuffle_accuracy": _accuracy(sender_shuffle, y),
        "receiver_shuffle_accuracy": _accuracy(receiver_shuffle, y),
        "clamped_waveform_accuracy": _accuracy(clamped_pred, y),
        "fixed_receiver_accuracy": _accuracy(fixed_pred, y),
        "sender_state_decode_accuracy": _best_sign_decode(hs, sf),
        "receiver_state_decode_accuracy": _best_sign_decode(hr, rf),
        "sender_waveform_distance": float(np.linalg.norm(wpos - wneg)),
        "receiver_susceptibility_distance": float(np.linalg.norm(kpos - kneg)),
        "sender_transplant_flip_fraction": float(np.mean(sender_flips)),
        "receiver_transplant_flip_fraction": float(np.mean(receiver_flips)),
        "learned_shape_kernel_abs_cosine": qcos,
        "waveform_area_difference_abs": float(abs(wpos.sum() - wneg.sum())),
        "waveform_peak_height_difference_abs": float(abs(wpos.max() - wneg.max())),
        "waveform_peak_index_difference_abs": float(abs(np.argmax(wpos) - np.argmax(wneg))),
        "carrier_peak_sample_difference_abs": float(abs(wpos[carrier_peak] - wneg[carrier_peak])),
    }


def _stats(rows: list[Dict[str, float]], key: str) -> Dict[str, float]:
    vals = np.asarray([r[key] for r in rows], dtype=float)
    return {
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def build_history_receipt(
    seeds: Iterable[int] = range(16),
    n_train_per_condition: int = 128,
    n_test_per_condition: int = 256,
    history_noise: float = 0.20,
    jitter: int = 2,
) -> Dict[str, object]:
    seeds = [int(s) for s in seeds]
    rows = []
    for seed in seeds:
        train = make_history_dataset(
            seed=1000 + seed,
            n_per_condition=n_train_per_condition,
            history_noise=history_noise,
            jitter=jitter,
        )
        test = make_history_dataset(
            seed=2000 + seed,
            n_per_condition=n_test_per_condition,
            history_noise=history_noise,
            jitter=jitter,
        )
        model = train_history_coupler(
            np.asarray(train["sender_history"]),
            np.asarray(train["receiver_history"]),
            np.asarray(train["target"]),
            seed=seed,
        )
        rows.append(evaluate_history_coupler(model, test, attacker_train=train))

    keys = list(rows[0])
    return {
        "question": "Can waveform x susceptibility emerge from temporal histories when only continuation error is supervised?",
        "scope": {
            "biological_claim": False,
            "explicit_state_labels_used_for_training": False,
            "status": "learned-history toy",
            "architectural_prior": "sender history modulates emitted shape; receiver history modulates temporal susceptibility; their inner product drives continuation",
            "interpretation": "tests whether the factorized mechanism can learn useful latent history states and temporal coupling, not whether biology uses it or whether the factorization is uniquely necessary",
        },
        "data": {
            "history_len": HISTORY_LEN,
            "history_channels": HISTORY_CHANNELS,
            "history_noise_sd": history_noise,
            "event_jitter_samples": jitter,
            "present_event_time": 0.0,
            "train_trials_per_seed": 4 * n_train_per_condition,
            "test_trials_per_seed": 4 * n_test_per_condition,
        },
        "n_seeds": len(seeds),
        "summary": {key: _stats(rows, key) for key in keys},
    }


def main() -> None:
    receipt = build_history_receipt()
    out = Path(__file__).resolve().parent / "results" / "history_receipt.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for key in (
        "full_accuracy",
        "additive_raw_history_accuracy",
        "sender_shuffle_accuracy",
        "receiver_shuffle_accuracy",
        "sender_state_decode_accuracy",
        "receiver_state_decode_accuracy",
        "learned_shape_kernel_abs_cosine",
    ):
        s = receipt["summary"][key]
        print(f"{key:38s} {s['mean']:.4f} +/- {s['sd']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
