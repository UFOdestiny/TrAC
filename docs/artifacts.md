# Artifacts under `EXP_ROOT`

Every stage writes one file per cache tag `{dataset}_{model}_k{K}` (for example `math500_qwen8b_k8`;
the new backbone uses `_k1`). Files are written atomically, and every stage skips outputs that
already exist, so interrupted runs resume. `trac/store.py` reads and writes them.

## GPU stages (`trac.pipeline`)

| Directory | Stage | Content |
| --- | --- | --- |
| `gen/{tag}.json.gz` | `generate` | One record per question: `id`, `question`, `gold` (+ `gold_raw`, `aliases`, `type`, `n_choices`, `category`), `prompt`, the original trace (`raw`, answer section `final`, extracted `answer`, label `correct`, `finish`, `reason_closed`, `n_tokens`), and the K-1 `samples` (`raw`, `final`, `answer`, `correct`, `meanlp`, `minlp`, `n_tokens`, `finish`) |
| `gen/{tag}.tok.npz` | `generate` | Token arrays of the original traces, concatenated with lengths: `prompt_ids`, `prompt_len`, `out_ids`, `out_len`, chosen-token log-probability `chosen_lp`, top-k entropy `ent`, mean top-k log-probability `topk_mean` |
| `endpoint/{tag}.json.gz` | `endpoint` | `end`: the greedy intermediate answer at the end of the trace (`text`, `answer`, `agree`, `ntok`, answer-token log-probabilities `tok_lps`); `ptrue`; `verbconf` (`text`, `value`) |
| `trajectory/{tag}.json.gz` | `trajectory` | `T` (reasoning tokens), `ck`: per checkpoint `k`, the greedy answer (`ans`, `surf`, `agree`, `tok_lps`), sampled answers `samp` (`ans`, `agree`), and the token log-likelihoods of *a* `tf`; `end`: the same at the end of the trace |
| `contrast/{tag}.json.gz` | `contrast` | `pos` (positions), `cands` (candidate answers with their source `a`, `traj`, or `elic`), `tf` (token log-likelihoods per candidate and position), `elic` (rival elicited at each position: `k`, `surf`, `ans`, `eq_a`, `tok_lps`) |
| `sample_trajectory/{tag}.json.gz` | `sample_trajectory` | Per question, for each extra sample `k`: `T`, `closed`, greedy intermediate `answers` at 8 checkpoints and the end, and their agreement with the sample's own answer `agree_own` |
| `prefix_consistency/{tag}.json.gz` | `prefix_consistency` | Per question and trace `k`: prefix consistency `pc`, agreement of each regeneration `same`, `regen_tokens`, `prefix_tokens` |
| `hidden/{tag}.npz` | `hidden_states` | `H` (questions x 4 positions x 3 layers x hidden size, float16), `ids`, `positions`, `layers` |
| `nli/{tag}.npz` | `nli` | Entailment `E` and contradiction `C` probabilities between the K answers, SelfCheckGPT score `selfcheck`, `ids` |
| `topk/{tag}.json.gz` | `topk` | Verbalized top-3 answers: `text`, `items` (`surf`, `ans`, `eq_a`, `conf`) |
| `reward/{tag}.npz` | `reward` | `ids`, reward-model score `reward`, last-token embedding `emb` (float16) |
| `prm/{tag}.npz` | `prm` | `ids`, minimum and last step reward `prm_min`, `prm_last` (mathematical benchmarks) |
| `verifier/{answer,full}/{tag}.npz` | `verifier` | `ids`, verifier logit `score` of the held-out setting |
| `latency/throughput_{name}.json` | `throughput` | Paired cost audit: per run, seconds of the original generation and of every TrAC query, warm and after a cache reset, and of SC@8; token and query counts (`_minimal`: the minimal configuration) |
| `latency/sampling_{name}.json` | `latency` | Seconds of the original generation, SC@k (k = 2, 4, 8), and the end-of-trace answer with and without prefix caching |

## CPU analysis (`trac.analysis`)

| Path | Module | Content |
| --- | --- | --- |
| `features/{tag}.npz` | `features.build` | Feature blocks aligned to `ids` and the labels `y` (see the module docstring) |
| `feature_cache/{trajectory,contrast,text_trajectory}__{tag}.npz` | `features.trajectory`, `.contrast`, `.text` | Derived blocks, rebuilt when the source cache or the module is newer |
| `auxiliary/{tag}.npz` | `features.auxiliary` | SC@2 vote share, error taxonomy, answer stated in the trace, output length |
| `scores/{tag}.npz` | `evaluation.scores` | `y` and the out-of-fold scores of every learned method in `trac.analysis.methods.METHODS` |
| `base_cache/*.npz` | `heads` | Content-hashed base-learner predictions; safe to delete, rebuilt on demand |
| `transfer/{protocol}_{norm}/{tag}.npz` | `transfer.evaluate` | `y` and the label-free scores of each feature view (`lomdo`, `lomfo`, `lomo`, `lodo`; `rank` or raw `none` features) |
| `transfer/lomdo_judge/{tag}.npz` | `transfer.judge` | Judge classifier on the reward-model embedding, and TrAC with it |
| `transfer/lomdo_probe/{tag}.npz` | `transfer.probe` | Hidden-state probe aligned across backbones by canonical correlation |
| `transfer/few_labels/{tag}.json` | `transfer.few_labels` | TrAC trained on n target labels against the label-free classifier |
| `transfer/diagnostics/{tag}.json` | `transfer.diagnostics` | Label-free AUROC with different reference sets for the ranks |
| `reports/*.json` | aggregates | `scores`, `transfer`, `judge_transfer`, `probe_transfer`, `verifier_transfer`, `few_labels`, `transfer_diagnostics`, `new_backbone_qwen32b`, `pairs`, `convergence`, `rivals` |

## Job queue

`locks/{tag}__{stage}/` marks a claimed job (`done` or `failed` inside it), and `logs/` holds one log
per job and per worker. Remove a failed job's lock directory to retry it.
