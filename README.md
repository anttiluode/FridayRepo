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

## Generic dynamics falsifier

`generic_dynamics.py` removes the remaining sender-waveform / receiver-filter architecture. The model is now one ordinary 12-unit tanh RNN with one recurrent state. It sees the two noisy temporal histories only after their four source channels have been mixed by a fixed orthogonal transform, followed by one identical query pulse. There is no sender module, receiver module, waveform generator, susceptibility kernel, state label, or explicit interaction feature. All recurrent weights are trained only from the final continuation loss.

This experiment exposed an important optimization boundary, so the receipt does **not** average failed runs away. With 2,048 training trials and 1,024 held-out trials per seed, 4 of 8 deterministic runs met the declared solved criterion (train ≥95%, held-out ≥90%). The other four remain in `results/generic_receipt.json` as optimization failures.

Among the four solved runs:

| condition / post-hoc probe | result |
|---|---:|
| generic RNN held-out continuation | **97.34% ± 1.57%** |
| linear classifier on the complete raw mixed histories | 49.49% ± 1.15% |
| sender history shuffled at test | 50.56% ± 1.11% |
| receiver history shuffled at test | 49.66% ± 1.60% |
| target linearly decodable **before** the common query | **97.27% ± 1.62%** |
| target linearly decodable after the query | 97.31% ± 1.45% |
| target decodable from the query-induced state change alone | 94.19% ± 2.59% |
| sender family linearly decodable pre-query | 76.83% ± 2.75% |
| receiver family linearly decodable pre-query | 78.32% ± 1.43% |
| additive probe on those two latent scores | 55.25% ± 1.64% |
| same two scores plus their product | **95.39% ± 3.38%** |

The result is therefore **not** “a generic RNN spontaneously invented our resonator interface.” It did something more revealing. In every solved run, the joint answer was already linearly present in the recurrent state **before the identical query arrived**. The common query was not required to create the interaction. The network had compiled the two histories into a joint state in advance.

At the same time, an interaction-like coordinate can be recovered post hoc. Two imperfect linear probes for the two history families are individually only ~77–78% accurate, and their additive combination remains at chance, but adding the product of those two latent scores recovers ~95% of the target. That is compatible with a low-dimensional bilinear organization inside the learned state, but it is not proof that the RNN internally implements a sender waveform meeting a receiver susceptibility. The stronger observation is simpler:

```text
history A + history B
        -> generic recurrent dynamics
        -> joint state already containing the interaction
        -> common query mostly reads/transforms what is already there
```

This is a useful correction to the resonance story. When the future question is fixed in advance, a generic nonlinear dynamical system has no reason to wait for the query. It can compile the relevant relation into resident state early. In that sense this experiment lands closer to **HistoryCompiler** than to a literal resonant reader.

The optimization failures matter too. The factorized learned-history model reached ~95% with 512 training trials per seed; this generic RNN was given 2,048 and still solved only 4/8 initializations under this simple training setup. That is not a general sample-efficiency theorem—the models and optimizers differ—but it is evidence inside this toy that supplying the factorization is a strong inductive bias.

## Next falsifier

Make the query **unpredictable until after the histories**. Present the same resident history state with several possible late probes whose identity changes which relation must be reported. Then a single precompiled answer is insufficient. The discriminator becomes whether a generic dynamical system learns a state whose *response to the late probe* carries the computation.

That is the cleaner place to look for learned susceptibility: not “can the network represent the answer before the ping?”, but “does the same stored state respond differently to a genuinely new query, and can that response geometry generalize to held-out queries?”

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python experiment.py
python history_experiment.py
python generic_dynamics.py
```

All three experiments use NumPy only; `pytest` is present for the TDD checks.
