"""Generic two-party learner for the late-query communication experiment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from late_query_task import HISTORY_CHANNELS, HISTORY_LEN, N_CLASSES, N_QUERIES

def _onehot(q, n=N_QUERIES):
    out = np.zeros((len(q), n), dtype=float)
    out[np.arange(len(q)), np.asarray(q, dtype=int)] = 1.0
    return out


def _flat_history(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 3 or a.shape[1:] != (HISTORY_LEN, HISTORY_CHANNELS):
        raise ValueError("history must have shape (n, HISTORY_LEN, 2)")
    return a.reshape(len(a), -1)


@dataclass
class Communicator:
    sender_w: np.ndarray
    sender_b: np.ndarray
    receiver_w: np.ndarray
    receiver_b: np.ndarray
    message_w: np.ndarray
    message_b: np.ndarray
    decoder_w: np.ndarray
    decoder_b: np.ndarray
    out_w: np.ndarray
    out_b: float
    channel_width: int
    direct_access: bool = False
    erase_sender: bool = False

    def sender_hidden(self, history):
        x = np.zeros_like(history) if self.erase_sender else history
        return np.tanh(_flat_history(x) @ self.sender_w + self.sender_b)

    def receiver_hidden(self, history):
        return np.tanh(_flat_history(history) @ self.receiver_w + self.receiver_b)

    def clean_message(self, history):
        hs = self.sender_hidden(history)
        if self.direct_access:
            return hs
        if self.channel_width == 0:
            return np.zeros((len(hs), 0), dtype=float)
        return np.tanh(hs @ self.message_w + self.message_b)

    def predict(self, data, *, channel_noise=0.0, use_shortcut=True, seed=0):
        msg = self.clean_message(data["sender_history"])
        if channel_noise and not self.direct_access and self.channel_width:
            msg = msg + np.random.default_rng(seed).normal(0.0, channel_noise, msg.shape)
        hr = self.receiver_hidden(data["receiver_history"])
        shortcut = data["shortcut"][:, None] if use_shortcut else np.zeros((len(hr), 1))
        z = np.concatenate([hr, msg, _onehot(data["query"]), shortcut], axis=1)
        g = np.tanh(z @ self.decoder_w + self.decoder_b)
        return g @ self.out_w + self.out_b


def train_communicator(
    data,
    channel_width: int,
    *,
    seed: int = 0,
    channel_noise: float = 0.12,
    use_shortcut: bool = True,
    epochs: int = 450,
    hidden_size: int = 6,
    decoder_size: int = 14,
    learning_rate: float = 0.012,
    l2: float = 1e-3,
    direct_access: bool = False,
    erase_sender: bool = False,
):
    """Train two separate generic temporal encoders and a late-query nonlinear receiver.

    The sender encoder sees only the old sender history. Its bounded message is produced before
    query identity exists. The receiver separately encodes its own history and receives the
    message, then the query and optional shortcut. Only the final regression target is supervised.
    """
    if channel_width < 0:
        raise ValueError("channel_width must be nonnegative")
    rng = np.random.default_rng(seed)
    xs = _flat_history(data["sender_history"])
    if erase_sender:
        xs = np.zeros_like(xs)
    xr = _flat_history(data["receiver_history"])
    y = np.asarray(data["target"], dtype=float)
    q = _onehot(data["query"])
    shortcut = data["shortcut"][:, None] if use_shortcut else np.zeros((len(y), 1))

    in_dim = HISTORY_LEN * HISTORY_CHANNELS
    sw = rng.normal(0.0, 0.20 / np.sqrt(in_dim), (in_dim, hidden_size))
    sb = np.zeros(hidden_size)
    rw = rng.normal(0.0, 0.20 / np.sqrt(in_dim), (in_dim, hidden_size))
    rb = np.zeros(hidden_size)
    msg_dim = hidden_size if direct_access else channel_width
    mw = rng.normal(0.0, 0.30 / np.sqrt(hidden_size), (hidden_size, channel_width)) if channel_width else np.zeros((hidden_size, 0))
    mb = np.zeros(channel_width)
    zdim = hidden_size + msg_dim + N_QUERIES + 1
    dw = rng.normal(0.0, 0.45 / np.sqrt(zdim), (zdim, decoder_size))
    db = np.zeros(decoder_size)
    ow = rng.normal(0.0, 0.30 / np.sqrt(decoder_size), decoder_size)
    ob = np.array(0.0)

    params = [sw, sb, rw, rb, mw, mb, dw, db, ow, ob]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    scale_y = np.std(y) + 1e-12

    for epoch in range(1, epochs + 1):
        hs = np.tanh(xs @ sw + sb)
        hr = np.tanh(xr @ rw + rb)
        if direct_access:
            msg_clean = msg = hs
        elif channel_width:
            msg_clean = np.tanh(hs @ mw + mb)
            msg = msg_clean + rng.normal(0.0, channel_noise, msg_clean.shape) if channel_noise else msg_clean
        else:
            msg_clean = msg = np.zeros((len(y), 0))
        z = np.concatenate([hr, msg, q, shortcut], axis=1)
        g = np.tanh(z @ dw + db)
        pred = g @ ow + ob
        dp = 2.0 * (pred - y) / (len(y) * scale_y * scale_y)

        gow = g.T @ dp + l2 * ow
        gob = np.array(dp.sum())
        dg = dp[:, None] * ow[None, :]
        da = dg * (1.0 - g * g)
        gdw = z.T @ da + l2 * dw
        gdb = da.sum(axis=0)
        dz = da @ dw.T
        pos = 0
        dhr = dz[:, :hidden_size]; pos += hidden_size
        dmsg = dz[:, pos:pos + msg_dim]

        grpre = dhr * (1.0 - hr * hr)
        grw = xr.T @ grpre + l2 * rw
        grb = grpre.sum(axis=0)

        if direct_access:
            dhs = dmsg
            gmw = np.zeros_like(mw); gmb = np.zeros_like(mb)
        elif channel_width:
            dmpre = dmsg * (1.0 - msg_clean * msg_clean)
            gmw = hs.T @ dmpre + l2 * mw
            gmb = dmpre.sum(axis=0)
            dhs = dmpre @ mw.T
        else:
            gmw = np.zeros_like(mw); gmb = np.zeros_like(mb); dhs = np.zeros_like(hs)

        gspre = dhs * (1.0 - hs * hs)
        gsw = xs.T @ gspre + l2 * sw
        gsb = gspre.sum(axis=0)

        grads = [gsw, gsb, grw, grb, gmw, gmb, gdw, gdb, gow, gob]
        norm = np.sqrt(sum(float(np.sum(gg * gg)) for gg in grads))
        clip = min(1.0, 5.0 / (norm + 1e-12))
        for i, (p, grad) in enumerate(zip(params, grads)):
            grad = grad * clip
            m[i] = b1 * m[i] + (1.0 - b1) * grad
            v[i] = b2 * v[i] + (1.0 - b2) * grad * grad
            p -= learning_rate * (m[i] / (1.0 - b1 ** epoch)) / (np.sqrt(v[i] / (1.0 - b2 ** epoch)) + eps)

        if epoch >= 120 and epoch % 20 == 0:
            nrmse = float(np.sqrt(np.mean((pred - y) ** 2)) / scale_y)
            if nrmse < 0.12:
                break

    return Communicator(sw, sb, rw, rb, mw, mb, dw, db, ow, float(ob), channel_width, direct_access, erase_sender)


def _nrmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)) / (np.std(y) + 1e-12))


def _message_diagnostics(model: Communicator, data, channel_noise):
    msg = model.clean_message(data["sender_history"])
    if msg.shape[1] == 0:
        return {"used_dimensions": 0, "effective_rank": 0.0, "min_class_mean_separation_over_noise": 0.0, "principal_sender_basis_component": 0, "principal_sender_basis_abs_correlation": 0.0}
    means = np.stack([msg[data["sender_class"] == c].mean(axis=0) for c in range(N_CLASSES)])
    centered = means - means.mean(axis=0, keepdims=True)
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    power = sv * sv
    used = int(np.sum(sv > (0.1 * sv[0] if sv[0] > 0 else np.inf)))
    p = power / power.sum() if power.sum() else np.zeros_like(power)
    nz = p[p > 0]
    effective = float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) else 0.0
    dists = [np.linalg.norm(means[i] - means[j]) for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES)]
    denom = channel_noise if channel_noise > 0 else 1.0

    h = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], float) / 2.0
    sender_basis = h[:, 1:4]
    principal_scores = centered @ vt[0] if len(sv) and sv[0] > 0 else np.zeros(N_CLASSES)
    cors = []
    for k in range(sender_basis.shape[1]):
        if np.std(principal_scores) == 0:
            cors.append(0.0)
        else:
            cors.append(float(np.corrcoef(principal_scores, sender_basis[:, k])[0, 1]))
    best = int(np.argmax(np.abs(cors))) if cors else 0
    return {
        "used_dimensions": used,
        "effective_rank": effective,
        "min_class_mean_separation_over_noise": float(min(dists) / denom),
        "principal_sender_basis_component": best + 1,
        "principal_sender_basis_abs_correlation": float(abs(cors[best])) if cors else 0.0,
    }


def evaluate_communicator(model, train, test, *, channel_noise=0.12, use_shortcut=True, eval_seed=0):
    pred = model.predict(test, channel_noise=channel_noise, use_shortcut=use_shortcut, seed=eval_seed)
    metrics = {"nrmse": _nrmse(pred, test["target"])}
    clean_msg = model.clean_message(test["sender_history"])
    leak = 0.0
    for bid in np.unique(test["base_id"]):
        idx = np.flatnonzero(test["base_id"] == bid)
        if clean_msg.shape[1]:
            leak = max(leak, float(np.max(np.ptp(clean_msg[idx], axis=0))))
    metrics["sender_query_leak_max_abs"] = leak
    rng = np.random.default_rng(eval_seed + 177)
    shuffled = dict(test)
    perm = rng.permutation(len(test["sender_history"]))
    shuffled["sender_history"] = test["sender_history"][perm]
    metrics["sender_shuffle_nrmse"] = _nrmse(
        model.predict(shuffled, channel_noise=channel_noise, use_shortcut=use_shortcut, seed=eval_seed + 1),
        test["target"],
    )
    metrics.update(_message_diagnostics(model, test, channel_noise))
    return metrics
