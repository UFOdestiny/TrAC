"""GPU stages. Each module is a `python -m trac.pipeline.<stage>` entry point that reads
cached generations and writes one artifact per cache tag under EXP_ROOT.

  generate            original trace + K-1 samples, token-exact            -> gen/
  endpoint            probes at the completed reasoning prefix             -> endpoint/
  trajectory          re-elicitation at checkpoints along the trace        -> trajectory/
  sample_trajectory   greedy checkpoint answers of the K-1 extra samples   -> sample_trajectory/
  prefix_consistency  prefix-consistency baseline (Iwase et al., 2026)     -> prefix_consistency/
  hidden_states       residual stream for the white-box baselines          -> hidden/
  nli                 NLI evidence for the sampling baselines              -> nli/
  latency             wall-clock audits                                    -> latency/
  queue               shared multi-GPU job queue over the stages above
  check_logprobs      vLLM vs. transformers log-prob sanity check
"""
