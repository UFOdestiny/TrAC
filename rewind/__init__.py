"""REWIND: cross-model confidence estimation from reasoning traces.

REWIND rewinds a completed reasoning trace and asks the same frozen model about it at cached prefixes:
which answer it would give now (intermediate answers), how likely its final answer a is (answer
likelihood), and which answer it would give if a were wrong (runner-up answers). These features, with
token-probability features of the original generation, feed a stacked logistic-regression and
gradient-boosting classifier. For a new model without labels, the classifier is fitted on other models'
labeled responses and every feature is replaced by its rank within the new model's unlabeled responses.

Correctness labels come from deterministic answer checkers (math_verify, option letters, alias match).

  rewind.config, rewind.benchmarks, rewind.backbones, rewind.datasets, rewind.prompts, rewind.answers,
  rewind.metrics, rewind.store    shared building blocks
  rewind.pipeline                 GPU stages: generation and the queries (vLLM) and the job queue
  rewind.features                 CPU: feature blocks of a setting
  rewind.views, rewind.classifier, rewind.transfer, rewind.score
                                  feature views, the stacked classifier, rank features, and scoring
"""
