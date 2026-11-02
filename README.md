# TrAC: Interrogating Reasoning Traces for Label-Free Correctness Estimation on New Models

TrAC (Trace-conditioned Answer Contrast) estimates whether a reasoning model's answer is correct,
including for a newly deployed model that has no labeled responses. Instead of only reading a
completed reasoning trace, TrAC asks the same frozen model questions about it at cached prefixes:

- **Intermediate answers**: which answer would it give now, at four checkpoints and at the end of the trace?
- **Answer likelihood**: how likely is its final answer *a* at those positions?
- **Rival answers**: which answer would it give if *a* were wrong, and how does *a* compare with its own alternatives?

These intermediate-answer features, together with token-probability features of the original
generation, feed a stacked logistic-regression and gradient-boosting classifier. To score a new model,
TrAC fits the classifier on other models' labeled responses and ranks every feature within the new
model's unlabeled responses, which removes backbone-specific feature scales.

Across 8 benchmarks and 4 backbones (30 eligible settings), TrAC reaches 0.813 AUROC without any label
of the target model or task, compared with 0.836 when trained in-domain. It also scores a held-out
Qwen3-32B at 0.840. All queries reuse the cached trace, adding 2% to 30% to generation latency at
moderate load.

## Repository layout

```
trac/
  config.py          paths and environment from .env
  benchmarks.py      evaluation matrix (benchmarks, backbones) and cache tags
  backbones.py       backbone registry and token-level reasoning formats
  datasets.py        benchmark loaders
  prompts.py         prompt templates and the answer cue
  answers.py         answer extraction and rule-based equivalence (correctness labels)
  metrics.py         AURC, selective risk, ECE, Brier score
  store.py           artifact readers and atomic writers
  pipeline/          GPU stages (vLLM, transformers) and the multi-GPU job queue
  analysis/
    features/        per-setting feature blocks
    evaluation/      in-domain scores, paired comparisons, convergence, rival answers
    transfer/        label-free transfer, judges, hidden-state probe, verifier, few labels, new backbone
    methods.py       feature views and the registry of evaluated methods
    heads.py         stacked classifier and the out-of-fold protocol
    run.py           the whole CPU analysis in dependency order
  paper/             tables and figures of the paper
configs/queues/      GPU job queues: the main matrix and the new backbone
scripts/             dataset download and GPU launchers
docs/                pipeline stages and artifact formats
```

## Setup

Use Python 3.10 or newer and run all commands from the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env        # then edit the paths
python -m trac.config       # prints the resolved paths
```

The GPU stages also need `torch>=2.9`, `transformers>=4.57`, and `vllm>=0.11` in a CUDA environment.
Rendering the framework figure needs `PyMuPDF>=1.24`, XeLaTeX, and the Times New Roman font.

### Configuration

All paths come from `.env`. Variables exported in the shell override it, and relative paths
resolve against the repository root.

| Variable | Default | Content |
| --- | --- | --- |
| `MODELS_ROOT` | `models` | One folder per model (see below) |
| `DATASETS_ROOT` | `data` | Benchmark mirrors written by `scripts/download_data.py` |
| `EXP_ROOT` | `outputs` | Caches, features, scores, reports, and logs ([docs/artifacts.md](docs/artifacts.md)) |
| `PAPER_DIR` | `outputs/paper` | Receives `table/*.tex` and `figures/*.pdf` |
| `PYTHON` | `python` | Interpreter used by the Bash launchers |
| `HF_HOME` | unset | Hugging Face cache for the dataset download |
| `TRAC_ENV_FILE` | `.env` | Alternative configuration file (set it in the environment) |

### Models and data

`MODELS_ROOT` holds one folder per model, named as in `trac/backbones.py`:

- Backbones: `Qwen3-8B`, `Llama-3.1-8B-Instruct`, `Ministral-8B-Instruct-2410`, `gpt-oss-20b`, and `Qwen3-32B` (the new backbone).
- Baselines: `deberta-large-mnli` (NLI), `Skywork-Reward-V2-Llama-3.1-8B` (reward model), `Qwen2.5-Math-PRM-7B` (process reward model), and `Qwen3-1.7B` (fine-tuned verifier).

Download the benchmarks (GSM8K, MATH500, Minerva-Math, OlympiadBench, AIME 2024-2026, GPQA-Diamond,
MMLU-Pro, TriviaQA). GPQA is gated, so accept its terms on the Hugging Face Hub and log in first:

```bash
python scripts/download_data.py                  # or --only math500,gsm8k
```

## Quick start: one setting

```bash
TAG=math500_qwen8b_k8
python -m trac.pipeline.generate --model qwen8b --dataset math500 --tag $TAG --k 8
for stage in endpoint trajectory contrast; do python -m trac.pipeline.$stage --gen-tag $TAG; done
python -m trac.analysis.features.build --tags $TAG
OMP_NUM_THREADS=1 python -m trac.analysis.evaluation.scores --tag $TAG --only "TrAC|Token probs."
```

`EXP_ROOT/scores/$TAG.npz` then holds the out-of-fold TrAC scores and the token-probability baseline.
Every module prints its options with `--help`.

## Reproducing the paper

**1. GPU caches.** Each line of a queue file is one setting with its GPU stages. Start one worker per GPU;
workers share the queue through lock directories under `EXP_ROOT/locks`, and every stage skips
outputs that already exist:

```bash
bash scripts/run_queue.sh configs/queues/main.txt,configs/queues/new_backbone.txt 0,1,2,3
bash scripts/queue_status.sh
```

The cost audits and the fine-tuned verifiers run separately, one GPU each:

```bash
# paired cost audit of every TrAC query, and of the minimal configuration
bash scripts/run_stage.sh throughput 0 math500_qwen8b_k8 math500_llama8b_k8 math500_ministral8b_k8 math500_gptoss20b_k8
STAGE_ARGS="--minimal --concurrency 16" bash scripts/run_stage.sh throughput 0 \
  math500_qwen8b_k8 math500_llama8b_k8 math500_ministral8b_k8 math500_gptoss20b_k8
# sampling latency (SC@k)
STAGE_ARGS="--n 64 --batch 16 --gpu-mem 0.88" bash scripts/run_stage.sh latency 0 \
  math500_qwen8b_k8 math500_llama8b_k8 math500_ministral8b_k8 olympiad_qwen8b_k8
# verifiers fine-tuned on the other backbones' other datasets (LOMDO), answer-only and full trace
for m in qwen8b llama8b ministral8b gptoss20b; do
  for d in gsm8k math500 minerva olympiad aime3 gpqa_diamond mmlupro triviaqa; do
    python -m trac.pipeline.verifier --model $m --dataset $d
    python -m trac.pipeline.verifier --model $m --dataset $d --full
  done
done
for d in gsm8k math500 minerva olympiad aime3 gpqa_diamond mmlupro triviaqa; do
  python -m trac.pipeline.verifier --model qwen32b --k 1 --dataset $d --full
done
```

**2. CPU analysis.** One command runs features, in-domain scores, label-free transfer, and the
analyses of the paper in dependency order, with single-threaded worker processes:

```bash
python -m trac.analysis.run --jobs 6
```

**3. Tables and figures.** Point `PAPER_DIR` at a LaTeX project (it needs `figures/framework.tex`
for the framework figure) and build:

```bash
OMP_NUM_THREADS=1 python -m trac.paper.build                  # or --only result,ablation,teaser
```

| Paper element | Output in `PAPER_DIR` | Build step | Produced from |
| --- | --- | --- | --- |
| Main results (Table 1) | `table/result.tex` | `result` | `evaluation.scores`, `pipeline.throughput`, `pipeline.latency` |
| Ablations | `table/ablation.tex` | `ablation` | `evaluation.scores` |
| Label-free transfer | `table/transfer_free.tex` | `transfer_free` | `transfer.evaluate`, `.judge`, `.probe`, `.verifier` |
| Transfer protocols | `table/transfer.tex` | `transfer` | `transfer.evaluate` |
| Target labels | `table/few.tex` | `few` | `transfer.few_labels` |
| New backbone | `table/new_backbone.tex` | `new_backbone` | `transfer.new_backbone` |
| Cost audit | `table/cost.tex` | `cost` | `pipeline.throughput` |
| Paired comparisons | `table/app_comparisons.tex` | `app_comparisons` | `evaluation.pairs` |
| Other appendix tables | `table/app_*.tex`, `table/label_efficiency.tex` | `app_*`, `label_efficiency` | `evaluation.scores`, `features.auxiliary` |
| Framework | `figures/framework.pdf` | `framework` | `figures/framework.tex` |
| Teaser panels | `figures/teaser-*.pdf` | `teaser` | trajectory caches, `transfer.few_labels` |
| Insight panels | `figures/insight-*.pdf` | `insight` | `transfer.evaluate`, `transfer.new_backbone`, latency audits |
| Rank against raw features | `figures/scale.pdf` | `scale` | `transfer.evaluate` |
| Agreement along the trace | `figures/trajectory.pdf` | `trajectory` | trajectory caches |

Numbers quoted in the text without a table come from `EXP_ROOT/reports/`: `convergence.json`
(convergence points, answers already written in the trace, traces of the same question),
`rivals.json` (what the rival answers reveal), and `transfer_diagnostics.json` (reference sets,
ranking-regret bound, sign reversals).

## Compute

The GPU stages ran on NVIDIA H200 GPUs (141 GB): about 100 GPU-hours in total, most of it for the eight
traces per question and the sampling, hidden-state, and prefix-consistency baselines. The CPU analysis
takes a few hours on 6 cores; base-learner predictions are cached under `EXP_ROOT/base_cache`, so later
runs are much faster. Run scikit-learn single-threaded per process (`OMP_NUM_THREADS=1`): OpenMP
oversubscription slows the boosted trees by orders of magnitude on shared machines.

## License

Code: [MIT](LICENSE). Models and datasets keep their own licenses and access terms.
