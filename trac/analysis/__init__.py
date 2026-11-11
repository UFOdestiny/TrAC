"""CPU analysis: every stage reads GPU caches or earlier analysis outputs under EXP_ROOT.

features     per-setting feature blocks (generation, endpoint, trajectory, rival answers, text)
evaluation   in-domain out-of-fold scores and the analyses of the paper's results
transfer     label-free transfer to a new backbone
cells, methods, heads, stats   feature access, the method registry, classifiers, and the bootstrap
run          the whole analysis in dependency order
"""
