# Late-query channel falsifier

This is FridayRepo's next gate after `generic_dynamics.py`.

The previous generic RNN was allowed to see both histories before an identical query, and in successful runs it simply compiled the joint answer into resident state before the query arrived. That did **not** test learned susceptibility. This experiment removes that escape route.

## Question

Can a sender preserve old history in a narrow noisy event **before it knows which question will later be asked**, and will that history survive when a cheaper recent-context shortcut is also available?

The experiment has two separate subsystems:

```text
old sender history -> sender encoder -> bounded noisy message ------+
                                                                late receiver query
receiver history ---------------------> receiver encoder ------------+-> answer
recent shortcut -----------------------------------------------------+
```

The sender never receives query identity. For every physical history pair, the exact same sender history, receiver history, and shortcut sample are duplicated under all three later queries. Only the receiver sees which query was selected.

The histories contain four temporal classes. Each class has exactly the same event counts on both channels; only event order differs. The three query-specific 4x4 target tables have ranks 1, 2, and 3.

## Rank-calibrated reference

For a genuinely bilinear communication model

```text
y(s,r,q) = message(s)^T reader(r,q)
```

a width-`d` message can represent at most a rank-`d` concatenated table. Here the concatenated target rank is **3**.

The optional shortcut exposes the first rank-1 component, so the exact residual table has rank **2**.

Best possible truncated-SVD NRMSE:

| message width | no shortcut | exact shortcut residual |
|---:|---:|---:|
| 0 | 1.000 | 1.000 |
| 1 | 0.511 | 0.404 |
| 2 | 0.207 | ~0 |
| 3 | ~0 | ~0 |

That is a calibration theorem for the bilinear reference only. The generic receiver below is nonlinear, so a scalar analog message can in principle encode more than one linear coordinate.

## Generic learner

The sender and receiver are separate small trainable temporal encoders. The sender emits a bounded `tanh` message of width 0–4. Gaussian noise (`sigma=0.12`) is added **after emission, at the receiver side**. Only then does the receiver get the three-way late query. The receiver is nonlinear and is not given the rank factorization.

Four deterministic seeds were run. Per seed there are 1,920 training and 2,880 held-out trials. History noise is `sigma=0.03`, event jitter is +/-1 sample, and shortcut noise is `sigma=0.08`.

### With the cheap shortcut

| width | held-out NRMSE | sender-shuffle NRMSE | success (`NRMSE < 0.25`) | learned effective message rank |
|---:|---:|---:|---:|---:|
| 0 | 0.606 +/- 0.015 | 0.606 | 0/4 | 0 |
| 1 | **0.136 +/- 0.005** | 0.714 | **4/4** | 1.00 |
| 2 | 0.134 +/- 0.004 | 0.716 | 4/4 | 1.00 |
| 3 | 0.132 +/- 0.005 | 0.715 | 4/4 | 1.00 |
| 4 | 0.133 +/- 0.005 | 0.716 | 4/4 | 1.00 |

The sender-erased control with the same shortcut gives **0.611 +/- 0.012 NRMSE**. Direct access to both learned history encodings plus the shortcut gives **0.144 +/- 0.011**.

So old history is not discarded in favor of the shortcut. Shuffling the sender history after training destroys the good solution, while erasing sender memory returns performance to the shortcut-only regime.

More interestingly, widening the channel beyond one scalar gives essentially no benefit. The learned class-mean message stays rank ~1 even when 2–4 dimensions are available. Its principal direction aligns almost perfectly with one of the task's sender Hadamard directions (`|r| ~= 0.99985` for width 1).

This is the concrete reason the generic learner can beat the bilinear residual-rank prediction. The shortcut already supplies one sender-dependent relational bit once the receiver combines it with its own context. The one-dimensional message learns another sender bit. In this four-class finite world, those pieces are enough for a nonlinear receiver to reconstruct the sender identity needed by all three later queries. The generic model is therefore **not violating** the rank theorem; it is using a different nonlinear code.

### Without the shortcut

The generic optimization is much less reliable:

| width | mean held-out NRMSE | success (`NRMSE < 0.25`) |
|---:|---:|---:|
| 0 | 1.230 | 0/4 |
| 1 | 1.042 +/- 0.488 | 0/4 |
| 2 | 0.796 +/- 0.619 | 1/4 |
| 3 | 0.805 +/- 0.505 | 0/4 |
| 4 | 1.057 +/- 0.550 | 1/4 |

Even the direct-access learner without the shortcut is unstable (**0.575 +/- 0.459 NRMSE**), so this cannot be interpreted as a channel-capacity failure. It is an optimization result: in this toy, the cheap rank-1 cue actually **scaffolds** discovery of the remaining old-history code rather than suppressing it.

That is almost the opposite of the naive shortcut story, and it is the main finding of this gate.

## What this shows

It shows a clean separation between three statements:

1. **Bilinear rank bound:** exact and known in advance. Rank 3 is needed without the shortcut; rank 2 is needed after subtracting the exact shortcut component.
2. **Generic nonlinear communication:** one noisy scalar can be enough because the receiver can combine a nonlinear code with its own state and the shortcut.
3. **Training pressure:** the cheap shortcut does not automatically erase old-history communication. Here it makes the useful low-dimensional history code far easier to learn, and sender-history ablation proves that the learned solution still depends on old history.

It does **not** show a biological resonance code, and it does not show that brains optimize communication this way. It is an engineering result about a late-query bottleneck.

## Why this matters for the bike idea

The important condition is now present:

```text
history happens
-> message is emitted
-> only later does the query arrive
```

The sender cannot precompute the answer. What persists across the gap is not "the answer" but a reusable property of old history that can support several future questions.

That is much closer to the operational meaning of

> memory as learned susceptibility to future probes

than the previous XOR experiments were.

The surprise is that the system does **not** need a wide literal rank-matched channel once a cheap contextual cue is available. It learns to transmit only the missing coordinate.

The next serious attacker is therefore not another wider channel. It is to make the shortcut **sometimes wrong or regime-dependent** and ask whether the sender dynamically increases history-bearing bandwidth only on the trials where the shortcut is unreliable. That would test adaptive communication rather than a fixed code.
