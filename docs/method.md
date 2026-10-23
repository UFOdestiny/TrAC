# Method

## Setting

A reasoning model answers a question with one trace: a reasoning body followed by the returned
answer section. The goal is a score that ranks responses by their probability of being correct
and stays calibrated, while adding little to the cost of the generation.

The reasoning body depends on the backbone format ([`trac/backbones.py`](../trac/backbones.py)):

| Format | Backbones | Reasoning body |
|---|---|---|
| `think` | Qwen3-8B (thinking), Olmo-3-Think, R1-Distill | inside `<think> … </think>` |
| `plain` | Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410 | the response up to the line that states the first `\boxed{}` |
| `harmony` | gpt-oss | the `analysis` channel |

Every probe is built at the token level as `prompt_ids + body_ids[:t] + tok(close + cue)`. The prefix
is therefore byte-identical to what the model generated, and vLLM's prefix cache serves every probe
after a single prefill of the trace.

## Signals

**PCE: Prefix-Conditioned Elicitation (active).**
- *Completed prefix* ([`trac.pipeline.endpoint`](../trac/pipeline/endpoint.py)), all probes at `t = T`:
  - the greedy answer after cue 0 (`The final answer is \boxed{`): its agreement with the returned answer `a`, its log-likelihood, and its first-token confidence;
  - the same under two cue paraphrases;
  - 8 sampled re-answers;
  - the teacher-forced likelihood of `a`.
- *Trajectory* ([`trac.pipeline.trajectory`](../trac/pipeline/trajectory.py)): at 20 checkpoints `t_j ≈ jT/21`, each moved back to a line break, and at the end: 1 greedy and 4 sampled answers, plus the teacher-forced likelihood of every token of `a`.

**TUP: Trace Uncertainty Profile (passive).** These come from the original generation's token
log-probabilities and renormalized top-5 entropy:
- 8 binned means of each;
- level, spread, slope, and linearity statistics;
- the confidence of the answer region.

**TrAC-0 (training-free).** The share of re-elicited answers (greedy and sampled, pooled per checkpoint)
that equal `a`, averaged over a fixed grid of 20 fractions plus the end
([`trac0_score`](../trac/analysis/trajectory_features.py)).

Answers are compared by one typed equivalence ([`trac/answers.py`](../trac/answers.py)): `math_verify`
for math, exact letters for multiple choice, and normalized alias match for open QA. The same
comparison produces labels, agreement, and votes.

## Feature blocks

[`trac.analysis.features`](../trac/analysis/features.py) writes one `.npz` per cell with these blocks:

| Block | Content | Dim |
|---|---|---|
| `AGREE`, `FLL`, `FCONF` | completed-prefix agreement, answer log-likelihood (mean/min/head), first-token confidence | 1, 3, 1 |
| `CUES`, `REPEAT`, `TFORCE`, `CONTRAST` | cue paraphrases, sampled re-answers, teacher-forced likelihood of `a`, likelihood contrast | 4, 3, 3, 2 |
| `TRACE`, `ANS` | TUP profile, answer-region confidence | 27, 5 |
| `QONLY`, `SHUF` | question-only and shuffled-trace controls | 3, 3 |
| `PT`, `SD`, `BASE` | P(True), SelfDoubt inputs, mean log-prob / self-certainty / DeepConf | 1, 2, 3 |
| `SC`, `SAMP` | K = 8 vote shares and entropy; SelfCheckGPT, semantic entropy, EigV, Ecc, Deg, KLE, CISC | 3, 9 |
| `META` | output length, reasoning closed, truncated | 3 |

[`trac.analysis.trajectory_features`](../trac/analysis/trajectory_features.py) derives the trajectory blocks:
- `TRAJ` (52 dims): per grid point, greedy agreement and the sampled share; plus commitment points, switches, distinct answers, and likelihood trends.
- `TRAC0`.
- Their ablations: greedy only, 8 checkpoints, and 4 checkpoints.
- The comparators `TEXTPOS` (answer position in the text) and `TSC` (trajectory-weighted vote).

## Heads and protocol

[`trac/analysis/heads.py`](../trac/analysis/heads.py):

- **Out-of-fold scores.** Every learned method is scored with 5 shuffled stratified 5-fold partitions, and the scores are averaged over the partitions. All methods in a cell share the same folds.
- **Stacked head (TrAC).** Two base learners, a logistic regression (C = 0.1) and a histogram gradient-boosting model (depth 3, leaf 20), are fixed a priori. They produce inner 3-fold out-of-fold scores inside each outer training fold, and a logistic combiner is fit on their logits, optionally with raw anchor features. Outer test folds never influence any fit.
- **Baselines.** Learned baselines (HS-probe, Liu et al., Supervised Consensus) choose their hyperparameters by nested 3-fold CV inside the outer training folds.
- **Aggregation.** Macro averages run over the settings with at least 10 correct and 10 incorrect responses. Contrasts use a paired hierarchical bootstrap: resample settings, then responses stratified by label, 2,000 times.

## Method registry

[`trac/analysis/methods.py`](../trac/analysis/methods.py) defines every evaluated method. The names below
are the keys stored in `EXP_ROOT/scores/*.npz` and used by the paper tables.

| Name | Budget | Inputs |
|---|---|---|
| `TrAC` | one trace | PCE (completed prefix + trajectory) + TUP, stacked head |
| `TrAC-0` | one trace | training-free trajectory agreement |
| `TrAC + HS` | white-box | TrAC + hidden-state base learners (answer end, PCE cue; layer 0.75L) |
| `TrAC + CF@8` | 8 traces | TrAC + consensus anchors (vote shares, entropy, CISC, SE) |
| `TrAC + SC@2`, `TrAC + PC`, `TrAC + CF@8 + TSC@8` | | TrAC with one more signal |
| `PCE`, `TUP`, `Trajectory only`, `PCE (completed prefix only)`, `TrAC without trajectory` | one trace | components under the same head |
| `TUP + HS`, `TUP + SC@2`, `TUP + CF@8` | | the same fusions without PCE |
| `Supervised prior signals`, `Text position` | one trace | matched-supervision comparators |
| `HS-probe`, `Liu et al. (2024)`, `Supervised Consensus@8`, `SC@8 (Platt)` | | learned baselines |
| `Abl: …` | | ablation rows of the paper |

Raw (unsupervised) baselines are listed in `raw_baselines`:
- single trace: mean log-prob, self-certainty, DeepConf-bottom, P(True), verbalized confidence, SelfDoubt, and prefix consistency;
- eight samples: SC@8, CISC@8, the semantic-entropy variants, SelfCheckGPT@8, EigV@8, Deg@8, KLE@8, and the trajectory-weighted SC@8.
