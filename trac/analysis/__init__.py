"""CPU stages: features, heads, scores, and the analyses behind the paper.

  features             per-cell feature blocks from the generation and endpoint caches
  trajectory_features  trajectory blocks, TrAC-0, text-position and trajectory-vote baselines
  cells                load every block of a cell (+ hidden states)
  heads, stats         out-of-fold heads (logistic, nested, stacked) and the bootstrap
  methods              registry of learned methods and raw baselines
  scores               out-of-fold scores per cell and their macro evaluation
  auxiliary            per-response statistics used by the paper tables
  controls             decoding matrix, generation seeds, latency summary
  pc_comparison        prefix consistency, within-question AUROC, weighted votes
  within_fold          within-fold AUROC robustness check
  text_trajectory      model-free text trajectory and latent commitment
  audit                data-quality audit of generation cells
"""
