# FridayRepo

A deliberately small **known-answer engineering spike** for one question:

> Can hidden history change the correct future through **state-dependent waveform × state-dependent susceptibility** when event times are identical?

This repo was motivated by the September 2026 preprint **“Action potential waveforms are state-dependent”** (Martin-Burgos et al., bioRxiv DOI `10.64898/2026.09.15.751814`). That paper shows that within-neuron action-potential waveform variability is systematically related to recent input and surrounding state. It does **not** show the mechanism tested here. FridayRepo asks the next computational question in a toy system.

## The construction

There are two independent hidden histories:

- sender history leaves state `s ∈ {-1,+1}`;
- receiver history leaves state `r ∈ {-1,+1}`.

Every trial has the **same event time**. The sender emits an AP-like waveform

```text
waveform = common carrier + s * small shape mode
```

and the receiver has a state-dependent temporal susceptibility

```text
susceptibility = r * shape mode.
```

The downstream response is their temporal inner product. Because the common carrier is orthogonal to the shape mode, the response contains the joint term

```text
response ∝ s * r
```

so the correct continuation is an XOR/XNOR-like function of the two histories.

The waveform manipulation is intentionally prevented from cheating through easy scalar cues: the two sender states have the same event time, equal integrated area to numerical precision, equal L2 energy, the same peak location, and the same peak height. What changes is temporal shape away from the event peak.

## Receipt

`python experiment.py` reproduces `results/receipt.json` over 16 seeds with Gaussian sample noise `σ=0.12`.

| condition | mean accuracy |
|---|---:|
| full waveform × susceptibility | **97.84%** |
| timestamp only | 50.00% |
| waveform only, even with oracle sender-state decode | 50.00% |
| waveform clamped to the mean | 50.00% |
| waveform shuffled between trials | 49.86% |
| receiver susceptibility frozen | 49.98% |
| additive model given both hidden state labels | 50.00% |
| interaction oracle | 100.00% |

Changing only sender state flips the noiseless composed answer in **100%** of the four state pairs. Changing only receiver state also flips it in **100%**.

The useful result is not the 97.8% by itself. The construction is known in advance. The useful part is the attacker pattern: neither timestamp, sender waveform/state alone, receiver state alone, nor an additive combination of both histories determines the continuation. The missing operation is the **interaction term**.

## Learned susceptibility follow-up

The strongest version in this repo does not hand the matched temporal kernel to the classifier. It learns a state-conditioned kernel from noisy training examples.

The interaction learner sees

```text
receiver_state * (waveform - training_mean)
```

while the additive attacker sees the **same waveform samples and receiver state**, but without their multiplicative interaction.

Across 16 seeds:

| learned model | result |
|---|---:|
| state-conditioned interaction | **96.05% ± 0.62%** |
| additive same-observation attacker | **50.23% ± 1.60%** |
| learned-kernel alignment with planted temporal mode | **0.881 ± 0.012** cosine |

So the receiver does not need to be handed the exact waveform feature: the training data can recover the useful temporal susceptibility. What remains architecturally supplied is the fact that receiver state is allowed to modulate how the waveform is read.

## What this does and does not show

It **does show computational sufficiency** for a narrow idea:

```text
sender history
    -> emitted temporal shape
    -> receiver-history-dependent susceptibility
    -> joint response
    -> different continuation
```

with identical event times and without area/energy/peak-height cues.

It **does not show** that biological neurons use this exact code, that action-potential waveform is a general neural communication alphabet, that resonance is the mechanism, or that this is a new learning algorithm. The core joint term is bilinear. This is a clean bridge experiment, not a biological discovery.

That boundary matters. The new preprint supplies evidence for the first arrow — neuronal state and recent input can be reflected in waveform shape. FridayRepo asks what becomes possible **if a downstream system is also state-dependent**.

## Why it connects to the recent repos

The experiment collapses several recurring motifs into one tiny system:

- **SilentPing:** a small event means something only through resident state.
- **OperatorTime / HistoryCompiler:** the same present can lead to a different effective operation because history survives in state.
- **ResonaattoriAivo:** temporal shape is read by a temporal susceptibility rather than reduced immediately to a timestamp/token.
- **Sihti / residue line:** the next useful step is to let prediction error reshape the susceptibility instead of planting the mode.

The sharper sentence is:

> **The ping can be a readout of sender state while the target is itself a state-dependent filter.**

That gives a minimal physical/computational form to “pings hit moving targets”: both sides of the coupling can carry history.

## Learned-history follow-up

`history_experiment.py` now removes the supplied `s` and `r` values from training. Each side instead receives a real 24-step, two-channel temporal history. The two history families contain the same event counts on the same channels; only their **order** differs. Events are jittered by ±2 samples and Gaussian noise (`σ=0.20`) is added. The present event still occurs at the same `t=0` on every trial.

A learned temporal encoder compresses each history to a resident state. Sender state modulates a learned small waveform deviation around a common AP-like carrier. Receiver state modulates a separately learned temporal susceptibility. The only supervision is the final continuation target. Hidden history-family labels exist in the dataset generator solely for post-hoc diagnostics and are never passed to `train_history_coupler`.

Across 16 seeds (`512` training and `1024` held-out trials per seed):

| condition / diagnostic | result |
|---|---:|
| learned history → waveform × susceptibility | **95.00% ± 0.96%** |
| additive linear model on both raw histories | 50.13% ± 1.37% |
| sender history shuffled at test | 50.03% ± 1.52% |
| receiver history shuffled at test | 50.32% ± 1.17% |
| emitted waveform clamped to common carrier | 50.00% |
| receiver resident state clamped | 49.99% |
| sender history family decodable from learned resident state | **97.55% ± 0.72%** |
| receiver history family decodable from learned resident state | **97.36% ± 0.63%** |
| sender-history transplant flips composed answer | **100%** |
| receiver-history transplant flips composed answer | **100%** |

The learned emitted shapes keep the same integrated area, carrier peak location, and peak height across the two sender histories. The state-dependent difference lives away from the common event peak. The full receipt is `results/history_receipt.json`.

This is a real step beyond the first construction: continuation error alone can teach both temporal encoders and the two sides of the temporal interface. But the important limitation moved rather than disappeared. We still **architecturally supply the factorization**

```text
sender history -> emitted shape
receiver history -> susceptibility
interaction -> continuation
```

so this does not show that an unconstrained dynamical system would discover waveform-mediated coupling, and it does not establish a uniquely resonant mechanism. The near-perfect alignment of the learned sender shape and receiver kernel is also expected from this bilinear objective: only their inner product matters, so optimization rewards alignment. It is a receipt for learning the factorized mechanism, not evidence for a biological resonance code.

## Next falsifier

Remove the last thing we are still handing the system: the explicit **sender-waveform / receiver-filter factorization**. Give a small generic dynamical network the same two histories and identical present event, train only continuation error, and then *measure* whether a sender-like emitted temporal mode and receiver-like susceptibility emerge. Compare it against an equally sized ordinary recurrent/nonlinear baseline.

That is the point where “waveform × susceptibility” would have to become a discovered internal organization rather than the coordinates we chose for the learner.

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python experiment.py
python history_experiment.py
```

Both experiments use NumPy only; `pytest` is present for the TDD checks.
