# TrAC: Trace-Conditioned Answer Consistency

**Uncertainty quantification for reasoning LLMs from one completed trace.**

TrAC asks whether a model keeps re-committing to the answer it returned while it reads its own
reasoning. Instead of sampling many full trajectories, it decodes short answers from the KV cache
of a single trace and combines them with signals the original generation already provides.

| View | What it measures | Extra cost |
|---|---|---|
| **PCE**, Prefix-Conditioned Elicitation (active) | At 20 checkpoints along the trace and at the completed prefix, re-elicit a short answer after `The final answer is \boxed{` (1 greedy + 4 sampled) and score the teacher-forced likelihood of the returned answer. | short decodes over the cached prefix |
| **TUP**, Trace Uncertainty Profile (passive) | Binned token log-probability and entropy, slopes, extrema, answer-region confidence. | none |

- **TrAC-0** is the training-free score: the share of re-elicited answers that equal the returned answer, averaged along the trace.
- **TrAC** fuses PCE and TUP with a cross-fitted stacked head (logistic regression + gradient boosting).
- Correctness labels always come from deterministic verifiers (`math_verify`, option letters, alias match). No LLM judge is used anywhere.

## Results

These are macro averages over 22 settings: 3 backbones × 8 benchmarks, minus the two AIME settings that have fewer than ten correct answers. Latency is the median wall-clock relative to one generation, counting every query the scores use.

| Method | Budget | AUROC ↑ | Latency |
|---|---|---|---|
| SelfDoubt | one trace | 0.652 | ≈1.0× |
| **TrAC-0** (training-free) | one trace | **0.722** | 1.04× |
| TUP (passive only) | one trace | 0.765 | 1.00× |
| **TrAC** | one trace | **0.811** | 1.06× |
| Liu et al. (2024), hidden states | one trace, white-box | 0.823 | 1.00× |
| **TrAC + HS** | one trace, white-box | **0.842** | 1.06× |
| SC@8 | 8 traces | 0.864 | 3.85× |
| **TrAC + CF@8** | 8 traces | **0.882** | 3.91× |

Backbones: Qwen3-8B (thinking), Llama-3.1-8B-Instruct, Ministral-8B-Instruct-2410. Benchmarks:
GSM8K, MATH500, Minerva, OlympiadBench, AIME 2024–2026, GPQA-Diamond, MMLU-Pro, and TriviaQA.

## Installation

```bash
git clone <this repository> trac && cd trac
pip install -e ".[gpu,data,dev]"     # CPU analysis only: pip install -e .
cp .env.example .env                 # then set MODELS_ROOT, DATASETS_ROOT, EXP_ROOT, PAPER_DIR
python scripts/download_data.py      # benchmark mirrors -> $DATASETS_ROOT (GPQA is gated)
pytest                               # CPU unit tests
```

Every path comes from `.env`, and no code contains an absolute path. Variables already set in the
environment override `.env`.

| Variable | Meaning |
|---|---|
| `MODELS_ROOT` | one folder per model, named as in [`trac/backbones.py`](trac/backbones.py), plus `deberta-large-mnli` |
| `DATASETS_ROOT` | benchmark mirrors written by `scripts/download_data.py` |
| `EXP_ROOT` | all caches, scores, and logs ([layout](docs/artifacts.md)) |
| `PAPER_DIR` | LaTeX project that receives `table/*.tex` and `figures/*.pdf` |
| `PYTHON` | interpreter used by `scripts/*.sh` (optional) |

## Quick start

Score one benchmark with one backbone (one GPU):

```bash
python -m trac.pipeline.generate   --model qwen8b --dataset math500 --tag math500_qwen8b_k8   # trace + 7 samples
python -m trac.pipeline.endpoint   --gen-tag math500_qwen8b_k8                               # completed-prefix probes
python -m trac.pipeline.trajectory --gen-tag math500_qwen8b_k8                               # checkpoint probes
python -m trac.analysis.features   --tags math500_qwen8b_k8
python -m trac.analysis.scores     --tag math500_qwen8b_k8 --only "TrAC|TUP"
```

Each GPU stage infers the backbone from the cache tag and writes one artifact per tag under
`EXP_ROOT`. A stage skips any tag whose output already exists.

Reproduce the paper (full matrix: two H200 GPUs, about 50 GPU-hours):

```bash
bash scripts/run_queue.sh configs/queues/1_baselines.txt,configs/queues/2_trac.txt,configs/queues/3_controls.txt 0,1
bash scripts/run_stage.sh trajectory 0 <tags...>          # also sample_trajectory, prefix_consistency
bash scripts/run_analysis.sh                             # CPU: features -> scores -> tables and figures
```

[`docs/reproduction.md`](docs/reproduction.md) lists every step, its cost, and the exact cell lists.

## Repository layout

```
trac/
  config.py          paths from .env and the EXP_ROOT layout
  backbones.py       backbone registry, reasoning formats, token-exact prefixes
  prompts.py         prompt templates, answer cues, self-assessment questions
  answers.py         answer extraction, typed equivalence, deterministic verification
  datasets.py        benchmark loaders
  store.py, metrics.py
  pipeline/          GPU stages (vLLM / transformers), one module per stage + a shared job queue
  analysis/          CPU stages: features, heads, method registry, scores, controls, comparisons
  paper/             tables and figures of the paper
configs/queues/      job lists for the multi-GPU queue
scripts/             launchers (queue, per-stage, CPU analysis, SLURM), data download
tests/               CPU unit tests
docs/                method, reproduction guide, artifact layout
```

## Documentation

- [`docs/method.md`](docs/method.md): signals, features, heads, evaluation protocol, and the method registry.
- [`docs/reproduction.md`](docs/reproduction.md): end-to-end runbook for the GPU and CPU stages.
- [`docs/artifacts.md`](docs/artifacts.md): what every stage writes under `EXP_ROOT`, and the file schemas.

## License

MIT, see [`LICENSE`](LICENSE). Models and datasets keep their own licenses.
