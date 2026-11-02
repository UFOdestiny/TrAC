"""TrAC: Trace-conditioned Answer Contrast, label-free correctness estimation for reasoning LLMs.

TrAC interrogates a completed reasoning trace with the same frozen model. At cached prefixes it asks
which answer the model would give now (intermediate answers), how likely its final answer a is (answer
likelihood), and which answer it would give if a were wrong (rival answers). These intermediate-answer
features, with token-probability features of the original generation, feed a stacked classifier. For a
new model without labels, the classifier is fitted on other models' labeled responses and every feature
is ranked within the new model's unlabeled responses.

Correctness labels come from deterministic verifiers (math_verify, option letters, alias match).

  trac.config, trac.benchmarks, trac.backbones, trac.datasets, trac.prompts, trac.answers,
  trac.metrics, trac.store   shared building blocks
  trac.pipeline              GPU stages: generation and probes (vLLM, transformers)
  trac.analysis              CPU stages: features, classifiers, evaluation, label-free transfer
  trac.paper                 tables and figures of the paper
"""
