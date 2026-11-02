"""GPU stages. Each module is a `python -m trac.pipeline.<stage>` entry point that writes one artifact
per cache tag under EXP_ROOT; `queue` runs them over the settings of a queue file.

  generate            original trace + K-1 samples, token-exact              -> gen/
  endpoint            end-of-trace answer, P(True), verbalized confidence    -> endpoint/
  trajectory          intermediate answers and likelihood of a at checkpoints -> trajectory/
  contrast            rival answers and their likelihoods along the trace    -> contrast/
  sample_trajectory   intermediate answers of the K-1 extra samples          -> sample_trajectory/
  prefix_consistency  prefix-consistency baseline (Iwase et al., 2026)       -> prefix_consistency/
  hidden_states       residual stream for the white-box baselines            -> hidden/
  nli                 NLI evidence for the sampling baselines                -> nli/
  topk                verbalized top-3 answers                               -> topk/
  reward              reward model score and response embedding              -> reward/
  prm                 process reward model (mathematical benchmarks)         -> prm/
  verifier            fine-tuned text verifier (per held-out setting)        -> verifier/
  throughput          paired cost audit of every TrAC query                  -> latency/
  latency             sampling latency audit (SC@k)                          -> latency/
"""
