# AQUA: Self-Pruning Neural Network

A small reverse-mode autodiff engine, MLP, Adam optimizer, and gradual global pruning implementation using Python 3.10+ and NumPy. Matplotlib is used for plots. **Scikit-learn is not used at all** (therefore it is not used for training, models, gradients, or even dataset loading). The classification dataset is generated locally in `train/data.py`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all gradient checks and mask correctness tests:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```

Reproduce the dense Part 2 baseline and learning curve data:

```bash
PYTHONPATH=. python -m train.baseline --epochs 80 --seed 7 --out results/baseline_history.json
```

Reproduce an 80%-sparsity self-pruning run:

```bash
PYTHONPATH=. python -m train.self_prune --epochs 80 --sparsity 0.8 --criterion saliency --seed 7 --out results/self_prune.json
```

Reproduce the Part 4 Pareto sweep (3 seeds per target; runtime depends on hardware):

```bash
PYTHONPATH=. python -m train.sweep --epochs 80 --seeds 7,17,27 --sparsities 0,.5,.75,.9,.95 --outdir results --plotdir plots
```

The sweep writes raw rows to `results/pareto_sweep.csv` and the Pareto plot to `plots/pareto.png`. Run the commands from the repository root. All random generators are explicitly seeded.

Generate the Part 2 loss/accuracy learning curve plot after running the baseline command:

```bash
PYTHONPATH=. python -m train.plot_learning_curve --input results/baseline_history.json --output plots/baseline_learning_curve.png
```

## Current experiment snapshot

The committed sweep used 50 epochs and seeds 7, 17, and 27. At 90% target sparsity, both methods retained 115/1,152 weights (90.017% achieved sparsity) and averaged 99.72% test accuracy across the three seeds. The observed saliency-minus-magnitude accuracy difference was 0.00 percentage points; this small experiment does not establish superiority for either criterion. At 95%, accuracy was 99.17% for saliency and 99.44% for magnitude on average. See `DESIGN.md` for the falsifiable claim and `results/pareto_sweep.csv` for raw rows.

**Cost caveat:** the operation count drops by approximately the achieved sparsity because the sparse-aware inference path visits active connections only. However, the measured Python-loop sparse path is slower than dense NumPy matrix multiplication on this tiny model. The dense timing baseline is a dense matrix multiply after applying the mask (i.e. dense-times-zero), not a sparse speedup. These measurements intentionally distinguish fewer connection MACs from lower wall-clock latency.

## What is measured

- `achieved_sparsity`: fraction of masked weight connections, excluding biases.
- `active_weights`: count of true entries in the binary masks.
- `dense_weight_macs_per_sample`: dense weight matrix multiply-accumulate count, equal to the number of weight entries across layers for one sample.
- `sparse_weight_macs_per_sample`: active connection count for the sparse-aware forward path.
- `dense_batch_ms` and `sparse_batch_ms`: measured batch latency on the current machine; these are noisy, implementation- and hardware-dependent, and are not a guarantee that sparse execution is faster.

The training path uses masked dense matrix multiplication for simplicity. The inference path `MLP.predict_sparse` loops over output units and multiplies only active incoming connections. This is a real sparse-aware path, but the Python loop overhead can outweigh arithmetic savings for small networks. The operation-count reduction is therefore reported separately from wall-clock timing.

## Limitations

This is a deliberately small reference implementation for correctness and experimentation, not a production inference runtime. The engine currently targets 2-D minibatch matrix multiplication, and the synthetic moons task is intentionally small. Results on this dataset should not be generalized to image datasets or large production models.
