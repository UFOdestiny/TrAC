# Reproduction guide

All commands run from the repository root with `.env` configured (see the README). GPU stages
use vLLM and are resumable: a stage skips any cache tag whose output already exists, and
generation resumes at the first missing chunk of 1,024 questions.

## Environment used for the paper

| Item | Value |
|---|---|
| Software | Python 3.11, vLLM 0.31.0, Transformers 5.10.4, PyTorch 2.13, scikit-learn 1.9.1, math-verify 0.9.0 |
| Hardware | 2 × H200 (141 GB); the CPU analysis runs on any machine |
| Compute | about 50 GPU-hours for the full matrix |

## Cells

A *cell* is one (benchmark, backbone) pair. Its cache tag is `<dataset>_<backbone>_k8`, for
example `math500_qwen8b_k8`, and every artifact of the cell is stored under that tag.

- **Main matrix (24 cells).** Datasets: `gsm8k math500 minerva olympiad aime3 gpqa_diamond mmlupro triviaqa`. Backbones: `qwen8b llama8b ministral8b`. AIME for Llama and Ministral has fewer than 10 correct answers, so those two cells are reported but excluded from the aggregates.
- **Decoding controls.** `dm_{g,t02,t10}_{math500,minerva}_<backbone>` vary the original-trace temperature. `<cell>_pt0.7` re-elicits at T = 0.7.
- **Seed controls.** `s{2,3}_{math500,olympiad}_<backbone>` use other generation seeds.

## 1. GPU stages

### 1a. Queued stages: generation, endpoint probes, hidden states, NLI, endpoint latency

```bash
bash scripts/run_queue.sh configs/queues/1_baselines.txt,configs/queues/2_trac.txt,configs/queues/3_controls.txt 0,1
bash scripts/queue_status.sh            # progress, active jobs, errors, disk
```

| Queue | Per cell |
|---|---|
| `1_baselines` | `generate` (K = 8, T = 0.7, top-p 0.95; 32,768 new tokens for Qwen3, 16,384 for the instruct models), endpoint `ptrue,verbconf`, `hidden_states`, `nli` |
| `2_trac` | endpoint `pce,qonly,shuffled,tforce,repeat`, merged into the stage-1 file (`--add`) |
| `3_controls` | decoding-matrix and seed cells (original trace + PCE), then the endpoint latency audits |

- **Locking.** Both workers pull from the same queue through atomic lock directories under `$EXP_ROOT/locks`, so the two GPUs balance automatically.
- **Retrying.** To retry a failed job, delete `$EXP_ROOT/locks/<job>` (and `locks/gen__<tag>` if its generation failed).
- **Queue format.** The line format is documented in [`trac/pipeline/queue.py`](../trac/pipeline/queue.py).

### 1b. Per-cell stages

```bash
MAIN="gsm8k_qwen8b_k8 gsm8k_llama8b_k8 ..."                         # the 24 main cells
bash scripts/run_stage.sh trajectory 0 $MAIN                         # 20 checkpoints x (1 greedy + 4 sampled)
bash scripts/run_stage.sh trajectory 1 dm_g_math500_qwen8b ...       # decoding and seed controls too (TrAC-0 columns)
bash scripts/run_stage.sh sample_trajectory 0 $MAIN                  # 8-checkpoint trajectories of the 7 extra samples
bash scripts/run_stage.sh prefix_consistency 1 $MAIN                 # Iwase et al. (2026) baseline
STAGE_ARGS="--mode trajectory" bash scripts/run_stage.sh latency 0 \
    math500_qwen8b_k8 olympiad_qwen8b_k8 math500_llama8b_k8 math500_ministral8b_k8
```

On SLURM, one stage over the 24 main cells can run on two GPUs:
`sbatch --account=<acct> --partition=<part> scripts/slurm/gpu_stage.sbatch prefix_consistency`.

### Practical notes

- **Stage isolation.** Run one vLLM engine per GPU. Each stage starts its engine at 85–88% memory utilization.
- **Chaining jobs.** To chain a job behind a running one, wait on its verified PID (`while kill -0 <pid>; do sleep 30; done`). Do not use `pgrep -f`: the pattern also matches the waiting shell itself.
- **Long traces.** Under the 32k-token budget, a fraction of AIME traces are truncated. Truncated traces are labeled incorrect and flagged in the `META` block.
- **Ministral.** Ministral loads in Mistral format. Prompts are always built as chat-template token ids, because re-encoding a rendered template is unsafe for the tekken tokenizer.

## 2. CPU analysis

```bash
bash scripts/run_analysis.sh 6      # 6 parallel single-threaded processes
```

This runs, in order:

| Step | Module | Output under `EXP_ROOT` |
|---|---|---|
| features | `trac.analysis.features` | `features/` |
| auxiliary statistics | `trac.analysis.auxiliary` | `auxiliary/` |
| out-of-fold scores (per cell) and their aggregate | `trac.analysis.scores` | `scores/`, `scores.json` |
| decoding / seed / latency controls | `trac.analysis.controls` | `controls.json` |
| prefix consistency, within-question AUROC, weighted votes | `trac.analysis.pc_comparison` | `prefix_consistency.json` |
| within-fold AUROC | `trac.analysis.within_fold` | `within_fold/` |
| text trajectory and latent commitment | `trac.analysis.text_trajectory` | `feature_cache/text_trajectory__*` |
| tables and figures | `trac.paper.build` | `$PAPER_DIR/table`, `$PAPER_DIR/figures`, `paper_summary.json` |

Scoring all methods for the 24 cells is the slowest step. Keep `OMP_NUM_THREADS=1`: OpenMP
spin-waiting under a shared CPU load made multi-threaded sklearn about 100× slower. Base-learner
predictions are cached under `base_cache/`, so later runs and `--only` re-scoring are cheap.

To rebuild individual tables after a change:

```bash
OMP_NUM_THREADS=1 python -m trac.paper.build --only table_main,table_ablation,figs
```

The available steps are `table_main`, `table_ablation`, `table_decmatrix`, `table_robustness`, `table_transfer`,
`table_label_eff`, `table_efficiency`, `app_matrix`, `app_blindspot`, `app_fusion`, `app_leakage_length`,
`app_taxonomy`, `app_dataset`, `app_efficiency`, `app_calibration`, `figures`, `fig_trajectory`, and `figs`.

## 3. Diagnostics

- `python -m trac.analysis.audit` reports per-cell accuracy, truncation, unparsable answers, length statistics, probe coverage, and the sanity of the hidden-state and NLI caches.
- `python -m trac.pipeline.check_logprobs --gen-tag <tag>` (GPU) checks the stored vLLM log-probabilities against a transformers recomputation of the same tokens.
