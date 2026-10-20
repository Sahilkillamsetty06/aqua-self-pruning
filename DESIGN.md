# Design notes and mathematical defense

## Importance criterion

For a currently active scalar weight `w`, consider deleting it by replacing it with zero. The perturbation is `delta_w = -w`. A first-order Taylor expansion of the loss gives

`L(w + delta_w) - L(w) ~= (dL/dw) * delta_w = -w * dL/dw`.

The signed expression estimates whether deletion locally increases or decreases loss. To rank connections by the magnitude of the predicted local disturbance, this implementation uses `|w * dL/dw|`. It is a first-order saliency proxy, not an exact loss increase: it ignores second-order interactions and is noisy when gradients are estimated from a minibatch. Magnitude pruning uses `|w|` as the baseline. The sweep compares them at the same requested sparsity levels and across multiple fixed seeds.

## What is the gradient of a masked weight?

The effective weight is `w_eff = m * w`, where `m` is a non-trainable binary mask. By the chain rule, `dL/dw = m * dL/dw_eff`. Thus a masked connection (`m = 0`) has zero gradient with respect to its underlying weight while the hard mask is in force. It contributes zero to the forward pass and cannot be updated by ordinary backpropagation. This is deliberate: a removed connection should not silently move because of stale gradients or optimizer momentum.

The mask is a discrete control variable, not a differentiable trainable parameter in this implementation. A regrowth policy must explicitly decide to set `m` back to one; it cannot expect ordinary gradients through the zero mask to discover a dormant connection. If a connection is revived, `Adam.on_mask_change` clears its first and second moments. Otherwise, momentum accumulated before pruning could create an unrelated update immediately after revival. The masked-weight test verifies zero masked gradient, exact zero forward parameter, zeroed moments on revival, and subsequent trainability.

## Pruning schedule

Pruning follows a cubic ramp from initial sparsity `s_i` to final sparsity `s_f`:

`s(t) = s_f + (s_i - s_f) * (1 - t/T)^3`.

Early training retains capacity while features are forming; pruning accelerates later, then leaves a recovery interval before the end of training. At each pruning event, a global ranking removes the lowest-scoring currently active connections until the target active count is reached. This creates a global budget rather than imposing an arbitrary equal sparsity per layer. The current implementation does not perform regrowth during the training algorithm; the optimizer's mask-transition API and tests cover safe revival semantics.

## Initialization and numerical stability

Hidden ReLU layers use He normal initialization with standard deviation `sqrt(2 / fan_in)`. The output layer uses Xavier normal initialization with standard deviation `sqrt(2 / (fan_in + fan_out))`. Biases start at zero. Softmax cross entropy subtracts each row's maximum logit before exponentiation, preventing overflow from large positive logits. Sigmoid uses separate positive/negative branches. Tests compare analytic gradients with centered finite differences.

## Autodiff bottlenecks and possible improvements

This engine builds Python objects for each operation and recursively discovers a topological order on each backward call. It stores NumPy arrays per node and does not fuse operations, so Python dispatch, graph traversal, and temporary allocations dominate for small workloads. Improvements include caching the topological order while the graph is immutable, operation fusion, in-place gradient accumulation where safe, vectorized batch operations, graph lifetime management, and using a compiled sparse kernel for inference. Any cached graph must be invalidated when operation topology changes.

## Serving in a multi-tenant inference service

A production service should separate model artifacts from tenant request state, validate shapes and model versions at admission, batch compatible requests, and enforce per-tenant concurrency, memory, and latency budgets. Model loading should verify checksums and schemas; sparse masks and compressed indices should be immutable after publication. A compiled CSR/CSC or block-sparse kernel is preferable to Python loops for high request rates. Benchmark p50/p95/p99 latency, throughput, CPU utilization, memory, cold-start cost, and accuracy under realistic batch sizes. Route requests by model version and hardware capability, use bounded queues and backpressure, isolate failures, and expose metrics without leaking tenant data. Sparse parameter count alone is not evidence of lower latency; the actual sparse kernel and target hardware must be benchmarked.

## Falsifiable claim supported by the committed sweep

With the current committed 50-epoch sweep, three seeds (7, 17, 27), and the synthetic moons task, at the 90% target both methods retained 115 of 1,152 weights (90.017% achieved sparsity). Mean test accuracy was 99.72% for magnitude pruning and 99.72% for saliency pruning; both had population standard deviation 0.20 percentage points. The measured difference is 0.00 percentage points across these three seeds. Therefore, this experiment does **not** support a claim that saliency is better than magnitude at 90% sparsity. It supports the narrower claim that both retained the same observed accuracy on this easy synthetic task under this configuration. The result is falsifiable by rerunning `train.sweep`; it should not be generalized to other datasets.
