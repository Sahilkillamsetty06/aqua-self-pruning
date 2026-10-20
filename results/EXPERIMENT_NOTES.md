# Experiment notes

- Dataset: locally generated two-class moons, 1,200 examples, fixed seed per run; 80/20 train/test split.
- Model: MLP widths `[2, 32, 32, 2]`, 1,152 weight connections, excluding biases from sparsity accounting.
- Optimizer: Adam, learning rate 0.01.
- Pareto sweep: 50 epochs; seeds 7, 17, 27; target sparsities 0%, 50%, 75%, 90%, 95%.
- Saliency score: `abs(weight * full_training_set_gradient)` at pruning events.
- Schedule: cubic ramp from epoch 5 through epoch 37; remaining epochs allow recovery.
- At 90% target: 115 active connections / 1,152 total = 90.017% achieved sparsity. Both magnitude and saliency averaged 99.72% test accuracy across three seeds; population standard deviation 0.20 percentage points.
- At 95% target: magnitude averaged 99.44% and saliency averaged 99.17% test accuracy.
- Sparse-aware inference visits only active connections and reduces the counted weight MACs by the achieved sparsity. The current Python-loop implementation is slower in wall-clock time than dense NumPy matmul for this small network; no latency speedup is claimed.
