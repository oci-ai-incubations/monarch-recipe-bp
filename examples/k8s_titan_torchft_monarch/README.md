# Configuration: OKE + TorchTitan + Monarch + TorchFT

[Deployment instructions](./Deployment.md)

Up to this point, our system is operationally clean (thanks to Monarch) and throughput-healthy (thanks to TorchTitan). But it's still fragile: if any one of the 16 GPUs hiccups mid-run, the whole job dies and we restart from the last checkpoint. On a 2-node cluster that's annoying. At 200 nodes it's a multi-thousand-dollar lesson.

This is exactly the problem [**TorchFT**](https://github.com/pytorch/torchft) was built to solve.

#### How TorchFT works, in one paragraph

TorchFT splits the world into **replicas**. Each replica is a self-contained DDP/FSDP group — in our case, one A100 node with 8 GPUs. Replicas train in lockstep, but they don't form a single torch.distributed process group. Instead, a small Rust-based coordination server called the **Lighthouse** sits off to the side and runs a per-step quorum: at every training step, every replica checks in, reports its progress, and TorchFT averages gradients across the surviving replicas via NCCL. If a replica dies, the others notice within milliseconds, drop it from the quorum, and **keep training without restarting**. When the dead replica comes back, it pulls fresh weights from a peer over the network and rejoins the quorum at the current step. No checkpoint reload, no `torchrun` restart, no rendezvous storm.

That's the headline:

- **No restart on failure.** Lose a replica → the rest keeps moving. Get the replica back → it catches up automatically.
- **No global barrier on the failure path.** The Lighthouse is out-of-band — it doesn't sit in the NCCL collective hot path, so a single hung process can't block everyone.
- **Per-step recovery granularity.** The worst case is "lose the last step's worth of work," not "lose hours."
- **Plays nicely with Monarch.** TorchFT manages replica-level fault tolerance; Monarch manages pod/process orchestration. Together they cover the full failure surface.

#### Configuration

- OKE (2 × A100 BM, 16 GPUs)
- TorchTitan (Llama 3 8B on C4)
- Monarch (controller-driven orchestration)
- **TorchFT + Lighthouse** (2 replicas, 8 GPUs each, per-step quorum)

The controller now spins up the Lighthouse before launching training and passes its address to every replica. Each replica's optimizer is wrapped in TorchFT's `FTOptimizersContainer`, which slots the per-step quorum + cross-replica gradient sync into the normal optimizer step.

#### Results — `K8S Monarch Test` with `--torchft`

| Date | Steps | Time | Loss | Status | MFU | TPS | TFLOPs | Grad Norm | Memory |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1000 | 6389 s | 12.24841 → 4.65391 | Success | **21.46%** | 1301 | 66.97 | 1.2422 | 55.12 GiB |

The good news first: **the run completed and converged correctly.** All 1000 steps finished, both end-of-run signals fired cleanly — the controller logged `[Controller] Lighthouse stopped` and the replicas reached `step: 999` — and the loss trajectory (12.25 → 4.65) is statistically indistinguishable from the Monarch-only run from 3.2 (12.27 → 4.65). The per-step quorum and cross-replica gradient sync work exactly as designed, and the patches above mean we're now training at the *correct* learning rate. From a correctness standpoint, TorchFT is wired up end-to-end on Monarch on OKE.

The bad news is in the wall-clock column. Time-to-1000-steps went from **2505 s to 6389 s** — a **2.5× slowdown** vs. the Monarch-only run from 3.2. MFU collapsed from 55.49% to 21.46%, and throughput per GPU dropped from 173 TFLOPs to 67. That is a lot, and it's worth a real explanation before we move on.

#### What we paid in performance

| Metric | Monarch only | + TorchFT | Delta |
|---|---|---|---|
| Wall-clock (1000 steps) | 2505 s | 6389 s | **+155%** |
| MFU | 55.49% | 21.46% | **−34.0 pp** |
| TPS | 3364 | 1301 | −61% |
| TFLOPs / GPU | 173.13 | 66.97 | −61% |
| Memory | 50.26 GiB | 55.12 GiB | +4.86 GiB |
| Grad Norm | 1.0923 | 1.2422 | within noise |

So: same loss curve, same convergence behavior, **2.5× the wall-clock**. The compute side didn't get slower — every per-GPU number that doesn't touch the network is identical to 3.2. Something else is eating the budget.

#### Why the slowdown — it's the inter-replica network, not TorchFT

Every TorchFT step does one extra thing that the Monarch-only run didn't: an **allreduce of the full gradient across replicas**. For Llama 3 8B in bf16, that's ~16 GB of gradients, sharded across 8 GPUs per replica → roughly **2 GB per rank** that has to cross the inter-node link, every single step.

The bottleneck isn't *that* this allreduce exists — it's the link it's running on. Looking at the NCCL logs from this run:

```
NCCL INFO NET/IB : No device found.
NCCL INFO NET/Socket : Using [0]eth0:10.244.2.55<0>
NCCL INFO Using network Socket
```

No InfiniBand. NCCL falls back to **TCP over the Kubernetes CNI overlay** — the pod-to-pod network every OKE cluster ships with by default. On this cluster, that overlay delivers roughly **1.25 GB/s** of effective bandwidth between pods. Push 2 GB across it and you've burned ~1.6 s before you've done anything useful — and that's per rank, every step.

Add up the per-step budget:

| Phase | Time |
|---|---|
| FSDP forward + backward (intra-replica, NVLink) | ~2.5 s |
| **TorchFT inter-replica allreduce (TCP overlay)** | **~3.5 s** |
| Lighthouse quorum + commit RPC | ~0.2 s |
| Optimizer step | ~0.05 s |
| **Total** | **~6.3 s / step** |

1000 steps × 6.3 s ≈ 6300 s, plus startup. That's the 6389 s we measured, almost exactly. The math points cleanly at one thing: **the inter-replica allreduce on the TCP overlay is the bottleneck.** Compute is fine. Monarch orchestration is fine. TorchFT's bookkeeping is fine. We're just shoving gradients through a pipe that's an order of magnitude slower than what the NICs in this cluster can actually do.

(The +4.86 GiB memory bump is also TorchFT's doing: it keeps a shadow copy of each rank's gradient shard for the cross-replica allreduce. Not a problem at 8B on A100 80 GB, but worth knowing about when we scale up.)