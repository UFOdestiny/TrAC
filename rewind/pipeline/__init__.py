"""GPU stages of REWIND. Each module is a `python -m rewind.pipeline.<stage>` entry point that writes one
artifact per cache tag under EXP_ROOT; `queue` runs them over the settings of the evaluation matrix.

  generate              original trace + K-1 samples, token-exact                     -> gen/
  endpoint              intermediate answer at the end of the trace                   -> endpoint/
  intermediate_answers  intermediate answers and likelihood of a at the checkpoints   -> intermediate_answers/
  runner_up             runner-up answers and the candidates' likelihoods             -> runner_up/
  queries               helpers shared by the query stages
  queue                 multi-GPU job queue over the main matrix and the new backbone
"""
