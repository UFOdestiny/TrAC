# Rewind and Ask: Cross-Model Confidence Estimation from Reasoning Traces

REWIND estimates whether a reasoning model's answer is correct, including for a new model that has no
labeled responses. It rewinds a completed reasoning trace and asks the same frozen model about it at
cached prefixes: which answer it would give at intermediate checkpoints (intermediate answers), how
likely its final answer *a* is there (answer likelihood), and which answer it would give if *a* were
wrong (runner-up answers). These query features, with token-probability features of the original
generation, feed a stacked logistic-regression and gradient-boosting classifier. To score a new model
without labels, the classifier is fitted on other models' labeled responses and every feature is replaced
by its rank within the new model's unlabeled responses (rank features). All queries reuse the cached
trace. REWIND-Lite keeps a minimal set of queries; REWIND@8 adds the consensus of eight sampled traces.

## Requirements

Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

The GPU stages also need `torch>=2.9`, `transformers>=4.57`, and `vllm>=0.11` in a CUDA environment.

## Configuration

All paths come from `.env`; copy `.env.example` and edit it. Variables exported in the shell override
`.env`, relative paths resolve against the repository root, and `REWIND_ENV_FILE` selects another file.

```bash
cp .env.example .env
python -m rewind.config        # prints the resolved paths
```

| Variable | Default | Content |
| --- | --- | --- |
| `MODELS_ROOT` | `models` | One folder per model, named as in `rewind/backbones.py` |
| `DATASETS_ROOT` | `data` | Benchmark mirrors written by `scripts/download_data.py` |
| `EXP_ROOT` | `outputs` | Generations, query caches, features, scores, and logs |
| `PYTHON` | `python` | Interpreter used by the Bash launchers |
| `HF_HOME` | unset | Hugging Face cache for the dataset download |

## Models and data

Place the models in `MODELS_ROOT`: `Qwen3-8B`, `Llama-3.1-8B-Instruct`, `Ministral-8B-Instruct-2410`,
`gpt-oss-20b`, and `Qwen3-32B` (the held-out new model). Download the benchmarks (GSM8K, MATH500,
Minerva-Math, OlympiadBench, AIME 2024-2026, GPQA-Diamond, MMLU-Pro, TriviaQA); GPQA is gated, so accept
its terms on the Hugging Face Hub and log in first:

```bash
python scripts/download_data.py                  # or --only math500,gsm8k
```

## One setting

A setting is a benchmark and a model, with the cache tag `{dataset}_{model}_k{K}`:

```bash
TAG=math500_qwen8b_k8
python -m rewind.pipeline.generate --model qwen8b --dataset math500 --tag $TAG --k 8
for stage in endpoint intermediate_answers runner_up; do python -m rewind.pipeline.$stage --gen-tag $TAG; done
python -m rewind.features.build --tags $TAG
OMP_NUM_THREADS=1 python -m rewind.score --tag $TAG                 # in-domain, out of fold
OMP_NUM_THREADS=1 python -m rewind.score --tag $TAG --label-free    # needs the other settings' features
```

`EXP_ROOT/rewind_scores/$TAG.npz` then holds the labels and the scores of REWIND, REWIND-Lite, and
REWIND@8. Every module prints its options with `--help`.

## Full matrix

The job queue builds its jobs from the benchmark and backbone registries (`rewind/benchmarks.py`): the
main matrix (8 benchmarks x 4 models, K = 8 traces per question) and the held-out new model (one trace
per question). Start one worker per GPU; workers share the queue through lock directories under
`EXP_ROOT/locks`, and every stage skips outputs that already exist:

```bash
bash scripts/run_queue.sh main,new_backbone 0,1,2,3
bash scripts/queue_status.sh
python -m rewind.pipeline.queue main --list      # the jobs and their state
bash scripts/run_stage.sh runner_up 0 math500_qwen8b_k8   # one stage outside the queue
```

Then build the features of every setting and score them (single-threaded scikit-learn per process;
OpenMP oversubscription slows the boosted trees considerably):

```bash
python -m rewind.features.build
for tag in $(ls outputs/features | sed 's/\.npz$//'); do
  OMP_NUM_THREADS=1 python -m rewind.score --tag $tag
  OMP_NUM_THREADS=1 python -m rewind.score --tag $tag --label-free
done
```

Base-learner predictions are cached under `EXP_ROOT/base_cache` by content hash, so later runs are faster.

## Repository layout

```
rewind/
  config.py, benchmarks.py, backbones.py, datasets.py, prompts.py, answers.py, metrics.py, store.py
  pipeline/    generate, endpoint, intermediate_answers, runner_up (vLLM), queries, and the job queue
  features/    feature blocks: token probabilities (build), intermediate answers, runner-up answers
  views.py     feature views of REWIND, REWIND-Lite, and REWIND@8
  classifier.py  stacked LR + gradient-boosting classifier and the out-of-fold protocol
  transfer.py  source settings and rank features for label-free transfer
  score.py     in-domain and label-free scoring
scripts/       dataset download and GPU launchers
```

## License

Code: [MIT](LICENSE). Models and datasets keep their own licenses and access terms.
