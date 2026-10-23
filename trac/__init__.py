"""TrAC: Trace-Conditioned Answer Consistency for uncertainty quantification in reasoning LLMs.

TrAC scores one completed reasoning trace with two views under a shared stacked head:

  PCE  Prefix-Conditioned Elicitation (active): re-elicit a short answer after the
       completed reasoning prefix and at checkpoints along the trace, and measure how
       consistently the model reproduces the answer it returned.
  TUP  Trace Uncertainty Profile (passive): token-level uncertainty already available
       from the original generation.

TrAC-0 is the training-free share of re-elicited answers that equal the returned answer.
Correctness labels come from deterministic verifiers (math_verify, option letters,
alias match), never from an LLM judge.

Package layout
  trac.config, trac.backbones, trac.prompts, trac.answers, trac.datasets, trac.store,
  trac.metrics             shared building blocks
  trac.pipeline            GPU stages: generation and probes (vLLM / transformers)
  trac.analysis            CPU stages: features, heads, evaluation, controls
  trac.paper               tables and figures of the paper
"""

__version__ = "1.0.0"
