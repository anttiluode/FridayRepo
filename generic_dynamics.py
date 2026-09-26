"""Generic recurrent falsifier for FridayRepo: no waveform/filter factorization is supplied."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np

HISTORY_LEN = 24
INPUT_DIM = 5
MIXING_SEED = 20260926


def _history(rng, family, noise, jitter):
    x = np.zeros((HISTORY_LEN, 2), float)
    early = 6 + int(rng.integers(-jitter, jitter + 1))
    late = 17 + int(rng.integers(-jitter, jitter + 1))
    if family == -1:
        x[early, 0], x[late, 1] = 1.0, 1.0
    elif family == 1:
        x[early, 1], x[late, 0] = 1.0, 1.0
    else:
        raise ValueError("family must be -1 or +1")
    if noise:
        x += rng.normal(0.0, noise, x.shape)
    return x


def _mixing_matrix():
    rng = np.random.default_rng(MIXING_SEED)
    q, r = np.linalg.qr(rng.normal(size=(4, 4)))
    signs = np.sign(np.diag(r)); signs[signs == 0] = 1.0
    return q * signs


def _compose(sender, receiver, mixing):
    seq = np.zeros((HISTORY_LEN + 1, INPUT_DIM), float)
    seq[:-1, :4] = np.concatenate([sender, receiver], axis=1) @ mixing
    seq[-1, 4] = 1.0
    return seq


def make_generic_dataset(seed=0, n_per_condition=128, history_noise=0.18, jitter=2):
    rng, mixing, rows = np.random.default_rng(seed), _mixing_matrix(), []
    for s in (-1, 1):
        for r in (-1, 1):
            for _ in range(n_per_condition):
                rows.append((_history(rng, s, history_noise, jitter),
                             _history(rng, r, history_noise, jitter), s, r, int(s * r > 0)))
    rng.shuffle(rows)
    sender = np.stack([z[0] for z in rows]); receiver = np.stack([z[1] for z in rows])
    return {
        "sequence": np.stack([_compose(a, b, mixing) for a, b in zip(sender, receiver)]),
        "sender_history": sender, "receiver_history": receiver,
        "sender_family": np.array([z[2] for z in rows]),
        "receiver_family": np.array([z[3] for z in rows]),
        "target": np.array([z[4] for z in rows]),
        "mixing": mixing, "channel_mixing_is_orthogonal": True, "present_event_time": 0.0,
    }


@dataclass
class GenericRNN:
    Wx: np.ndarray; Wh: np.ndarray; b: np.ndarray; wo: np.ndarray; bo: float

    def hidden_trajectory(self, sequence):
        x = np.asarray(sequence, float); h = np.zeros((len(x), self.Wh.shape[0])); states = [h.copy()]
        for t in range(x.shape[1]):
            h = np.tanh(x[:, t] @ self.Wx + h @ self.Wh + self.b); states.append(h.copy())
        return np.stack(states, axis=1)

    def logits(self, sequence):
        return self.hidden_trajectory(sequence)[:, -1] @ self.wo + self.bo


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _acc(logits, y):
    return float(np.mean((logits >= 0) == (np.asarray(y) > 0)))


def _forward(x, Wx, Wh, b, wo, bo):
    h = np.zeros((len(x), Wh.shape[0])); states = [h]
    for t in range(x.shape[1]):
        h = np.tanh(x[:, t] @ Wx + h @ Wh + b); states.append(h)
    return h @ wo + bo, states


def train_generic_rnn(sequence, target, seed=0, hidden_size=12, epochs=900, lr=0.012, l2=1e-4):
    """Full-batch BPTT/Adam; only the final continuation target is supervised."""
    x, y, rng = np.asarray(sequence, float), np.asarray(target, float), np.random.default_rng(seed)
    Wx = rng.normal(0, 0.45 / np.sqrt(x.shape[2]), (x.shape[2], hidden_size))
    q, _ = np.linalg.qr(rng.normal(size=(hidden_size, hidden_size))); Wh = 0.82 * q
    b = np.zeros(hidden_size); wo = rng.normal(0, 0.35 / np.sqrt(hidden_size), hidden_size); bo = 0.0
    params = [Wx, Wh, b, wo]; m = [np.zeros_like(p) for p in params] + [0.0]; v = [np.zeros_like(p) for p in params] + [0.0]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for epoch in range(1, epochs + 1):
        logits, states = _forward(x, Wx, Wh, b, wo, bo); dlogit = (_sigmoid(logits) - y) / len(y)
        grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b), states[-1].T @ dlogit + l2 * wo]
        gbo = float(dlogit.sum()); dh = dlogit[:, None] * wo[None, :]
        for t in range(x.shape[1] - 1, -1, -1):
            dz = dh * (1.0 - states[t + 1] ** 2)
            grads[0] += x[:, t].T @ dz; grads[1] += states[t].T @ dz; grads[2] += dz.sum(0)
            dh = dz @ Wh.T
        grads[0] += l2 * Wx; grads[1] += l2 * Wh
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads) + gbo * gbo); scale = min(1.0, 5.0 / (norm + 1e-12))
        for i, (p, g) in enumerate(zip(params, grads)):
            g *= scale; m[i] = b1 * m[i] + (1 - b1) * g; v[i] = b2 * v[i] + (1 - b2) * g * g
            p -= lr * (m[i] / (1 - b1 ** epoch)) / (np.sqrt(v[i] / (1 - b2 ** epoch)) + eps)
        gbo *= scale; m[-1] = b1 * m[-1] + (1 - b1) * gbo; v[-1] = b2 * v[-1] + (1 - b2) * gbo * gbo
        bo -= lr * (m[-1] / (1 - b1 ** epoch)) / (np.sqrt(v[-1] / (1 - b2 ** epoch)) + eps)
        if epoch >= 120 and epoch % 20 == 0 and _acc(logits, y) >= 0.995:
            break
    return GenericRNN(Wx, Wh, b, wo, float(bo))


def _ridge_fit(X, y, ridge=1.0):
    X1 = np.column_stack([np.asarray(X, float), np.ones(len(X))]); reg = ridge * np.eye(X1.shape[1]); reg[-1, -1] = 0
    return np.linalg.solve(X1.T @ X1 + reg, X1.T @ np.asarray(y, float))


def _ridge_score(X, beta):
    return np.column_stack([np.asarray(X, float), np.ones(len(X))]) @ beta


def _probe(train_X, train_y, test_X, test_y, ridge=1.0):
    beta = _ridge_fit(train_X, train_y, ridge); score = _ridge_score(test_X, beta)
    return float(np.mean(np.where(score >= 0, 1, -1) == test_y)), beta, score


def _sequences(sender, receiver, mixing):
    return np.stack([_compose(a, b, mixing) for a, b in zip(sender, receiver)])


def evaluate_generic_rnn(model, train, test):
    ytr = np.where(train["target"] > 0, 1, -1); yte = np.where(test["target"] > 0, 1, -1)
    training_accuracy = _acc(model.logits(train["sequence"]), train["target"])
    full_logits = model.logits(test["sequence"]); full_accuracy = _acc(full_logits, test["target"])
    raw_tr = train["sequence"][:, :-1, :4].reshape(len(ytr), -1); raw_te = test["sequence"][:, :-1, :4].reshape(len(yte), -1)
    linear_raw, _, _ = _probe(raw_tr, ytr, raw_te, yte, 5.0)
    rng = np.random.default_rng(99173); ps, pr = rng.permutation(len(yte)), rng.permutation(len(yte))
    sender_shuffle = _acc(model.logits(_sequences(test["sender_history"][ps], test["receiver_history"], test["mixing"])), test["target"])
    receiver_shuffle = _acc(model.logits(_sequences(test["sender_history"], test["receiver_history"][pr], test["mixing"])), test["target"])

    tr = model.hidden_trajectory(train["sequence"]); te = model.hidden_trajectory(test["sequence"])
    pre_tr, pre_te, post_tr, post_te = tr[:, -2], te[:, -2], tr[:, -1], te[:, -1]
    def query_delta(pre):
        z = np.zeros((len(pre), INPUT_DIM)); q = z.copy(); q[:, 4] = 1.0
        return np.tanh(q @ model.Wx + pre @ model.Wh + model.b) - np.tanh(z @ model.Wx + pre @ model.Wh + model.b)
    delta_tr, delta_te = query_delta(pre_tr), query_delta(pre_te)
    pre_target, _, _ = _probe(pre_tr, ytr, pre_te, yte); post_target, _, _ = _probe(post_tr, ytr, post_te, yte)
    delta_target, _, _ = _probe(delta_tr, ytr, delta_te, yte)
    sender_acc, sb, sender_test = _probe(pre_tr, train["sender_family"], pre_te, test["sender_family"])
    receiver_acc, rb, receiver_test = _probe(pre_tr, train["receiver_family"], pre_te, test["receiver_family"])
    sender_train, receiver_train = _ridge_score(pre_tr, sb), _ridge_score(pre_tr, rb)
    add_tr = np.column_stack([sender_train, receiver_train]); add_te = np.column_stack([sender_test, receiver_test])
    int_tr = np.column_stack([sender_train, receiver_train, sender_train * receiver_train])
    int_te = np.column_stack([sender_test, receiver_test, sender_test * receiver_test])
    low_add, _, _ = _probe(add_tr, ytr, add_te, yte, 0.5); low_inter, _, _ = _probe(int_tr, ytr, int_te, yte, 0.5)
    no_query = test["sequence"].copy(); no_query[:, -1, 4] = 0.0; no_query_logits = model.logits(no_query)
    no_query_accuracy = _acc(no_query_logits, test["target"])
    pre_model_accuracy = _acc(pre_te @ model.wo + model.bo, test["target"])
    effect = float(np.mean(np.abs(full_logits - no_query_logits)) / (np.mean(np.abs(full_logits)) + 1e-12))
    if pre_target >= 0.90 or pre_model_accuracy >= 0.90:
        interpretation = "interaction_precomputed_before_query"
    elif pre_target < 0.70 and post_target >= 0.90 and delta_target >= 0.85:
        interpretation = "query_reveals_state_dependent_interaction"
    else:
        interpretation = "distributed_or_ambiguous"
    return {
        "training_accuracy": training_accuracy, "full_accuracy": full_accuracy,
        "linear_raw_history_accuracy": linear_raw, "sender_shuffle_accuracy": sender_shuffle,
        "receiver_shuffle_accuracy": receiver_shuffle, "no_query_accuracy": no_query_accuracy,
        "prequery_model_accuracy": pre_model_accuracy, "prequery_target_probe_accuracy": pre_target,
        "postquery_target_probe_accuracy": post_target, "query_response_target_probe_accuracy": delta_target,
        "sender_prequery_probe_accuracy": sender_acc, "receiver_prequery_probe_accuracy": receiver_acc,
        "lowdim_additive_probe_accuracy": low_add, "lowdim_interaction_probe_accuracy": low_inter,
        "query_effect_fraction": effect, "probe_interpretation": interpretation,
    }


def _stats(rows, key):
    x = np.array([r[key] for r in rows], float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "min": float(x.min()), "max": float(x.max())}


def build_generic_receipt(seeds: Iterable[int] = range(8), n_train_per_condition=512, n_test_per_condition=256, history_noise=0.18, jitter=2):
    rows = []
    for seed in map(int, seeds):
        train = make_generic_dataset(1000 + 2 * seed, n_train_per_condition, history_noise, jitter)
        test = make_generic_dataset(1001 + 2 * seed, n_test_per_condition, history_noise, jitter)
        rows.append(evaluate_generic_rnn(train_generic_rnn(train["sequence"], train["target"], seed=seed), train, test))
    keys = ["training_accuracy", "full_accuracy", "linear_raw_history_accuracy", "sender_shuffle_accuracy", "receiver_shuffle_accuracy",
            "no_query_accuracy", "prequery_model_accuracy", "prequery_target_probe_accuracy", "postquery_target_probe_accuracy",
            "query_response_target_probe_accuracy", "sender_prequery_probe_accuracy", "receiver_prequery_probe_accuracy",
            "lowdim_additive_probe_accuracy", "lowdim_interaction_probe_accuracy", "query_effect_fraction"]
    solved = [r for r in rows if r["training_accuracy"] >= 0.95 and r["full_accuracy"] >= 0.90]
    summarize = lambda group: {k: _stats(group, k) for k in keys} if group else {}
    def counts(group):
        out = {}
        for r in group: out[r["probe_interpretation"]] = out.get(r["probe_interpretation"], 0) + 1
        return out
    return {
        "question": "Does a generic recurrent dynamical system invent an equivalent factorization when only continuation loss is supplied?",
        "scope": {"biological_claim": False, "factorized_sender_receiver_modules": False, "explicit_interaction_feature": False,
                  "explicit_state_labels_used_for_training": False,
                  "model": "single vanilla tanh RNN over orthogonally mixed history channels plus one identical query pulse",
                  "interpretation": "post-hoc probes discriminate precomputation from query-revealed state dependence; they do not prove a unique internal mechanism"},
        "data": {"history_len": HISTORY_LEN, "mixed_history_channels": 4, "query_channels": 1,
                 "history_noise_sd": history_noise, "event_jitter_samples": jitter,
                 "train_trials_per_seed": 4 * n_train_per_condition, "test_trials_per_seed": 4 * n_test_per_condition},
        "n_seeds": len(rows),
        "optimization": {"success_definition": "training accuracy >= 0.95 and held-out accuracy >= 0.90",
                         "success_count": len(solved), "success_fraction": float(len(solved) / len(rows)) if rows else 0.0,
                         "note": "failed runs are retained rather than averaged away; mechanism summaries are shown separately for solved runs"},
        "summary": {"all_runs": summarize(rows), "successful_runs": summarize(solved)},
        "probe_interpretation_counts": {"all_runs": counts(rows), "successful_runs": counts(solved)},
    }


def main():
    receipt = build_generic_receipt(); out = Path("results/generic_receipt.json"); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2) + "\n"); print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
