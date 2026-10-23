# Artifacts under `EXP_ROOT`

Each stage writes one file per cache tag. Directory names are defined once in
[`trac/config.py`](../trac/config.py).

```
EXP_ROOT/
  gen/{tag}.json.gz, {tag}.tok.npz   trac.pipeline.generate
  endpoint/{tag}.json.gz             trac.pipeline.endpoint
  trajectory/{tag}.json.gz           trac.pipeline.trajectory
  sample_trajectory/{tag}.json.gz    trac.pipeline.sample_trajectory
  prefix_consistency/{tag}.json.gz   trac.pipeline.prefix_consistency
  hidden/{tag}.npz                   trac.pipeline.hidden_states
  nli/{tag}.npz                      trac.pipeline.nli
  latency/{endpoint,trajectory}_{name}.json   trac.pipeline.latency
  features/{tag}.npz                 trac.analysis.features
  feature_cache/                     trajectory / text-position / TSC / text-trajectory blocks (rebuilt when stale)
  auxiliary/{tag}.npz                trac.analysis.auxiliary
  scores/{tag}.npz                   trac.analysis.scores (out-of-fold scores, one array per method + y)
  within_fold/{tag}.json             trac.analysis.within_fold
  base_cache/                        base-learner predictions, keyed by content hash
  scores.json, controls.json, prefix_consistency.json, paper_summary.json
  locks/, logs/                      queue state and per-job logs
```

## Generation (`gen/`)

`{tag}.json.gz` holds one record per question:

```
id, question, gold, gold_raw, aliases, type (math | mc | qa), n_choices, category, prompt,
raw          original trace, special tokens kept
final        returned answer section
answer       extracted answer
correct      deterministic label
finish, reason_closed, n_tokens
samples      K-1 records {raw, final, answer, correct, meanlp, minlp, n_tokens, finish}
```

`{tag}.tok.npz` holds the original traces' token arrays, concatenated across questions with length
offsets: `prompt_ids, prompt_len, out_ids, out_len, chosen_lp, ent, topk_mean`. Use
`trac.store.load_generation(tag)`, which attaches `_pids` / `_oids` to each record.

## Probes

**`endpoint/{tag}.json.gz`**, per question:
- `pce`: 3 cue records `{text, answer, agree, ntok, tok_lps}`;
- `qonly`, `shuffled` (+ `src_id`): the control probes;
- `tforce`: `{ans, ans_ids, tok_lps}`;
- `repeat`: 8 records `{text, answer, agree, lp0}`;
- `ptrue`: a float;
- `verbconf`: `{text, value}`.

**`trajectory/{tag}.json.gz`**, per question: `{id, T, closed, ck, end}`.
- `ck` holds one record per checkpoint: `{k, ans, surf, agree, tok_lps, samp: {ans, agree}, tf}`.
  - `k` is the checkpoint position in body tokens;
  - `tf` is the teacher-forced log-probability of each token of `a}`.
- `end` has the same fields at the completed prefix.

**`sample_trajectory/{tag}.json.gz`**, per question: `samples`, one record per extra sample
`{k, T, closed, agree_own, answers}`, covering 8 checkpoints plus the end.

**`prefix_consistency/{tag}.json.gz`**, per question: `traces["0".."7"]`, each
`{pc, same, regen_tokens, prefix_tokens}`. Trace `"0"` is the original trace.

**`hidden/{tag}.npz`**: `H` with shape (N, 4 positions, 3 layers, d), float16.
- Positions: `end_reason, answer_end, mean_reason, cue`.
- Layers: 0.5L, 0.75L, L.

**`nli/{tag}.npz`**:
- `E`, `C`: entailment and contradiction probabilities among the K answers, shape (N, K, K);
- `selfcheck`: the SelfCheckGPT-NLI score.

## Analysis

- **`features/{tag}.npz`**: `ids`, `y`, and the feature blocks listed in [method.md](method.md#feature-blocks).
- **`scores/{tag}.npz`**: `y` and one out-of-fold score array per learned method, keyed by the registry names in [method.md](method.md#method-registry).
