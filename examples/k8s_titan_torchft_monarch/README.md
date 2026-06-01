# Configuration: OKE + TorchTitan + Monarch + TorchFT + RDMA

[Deployment instructions](./Deployment.md)

[The previous configuration test](https://github.com/oci-ai-incubations/monarch-recipe-bp/blob/cfg3_oke_torchtitan_monarch_torchft/examples/k8s_titan_torchft_monarch/README.md) ended with a clear diagnosis: TorchFT's correctness is fine, but the per-step inter-replica allreduce is running over the Kubernetes CNI overlay at ~1.25 GB/s. The fix is obvious — put that allreduce on the RDMA fabric the cluster already has. The work is on the OKE side, not the training side: a DOCA-OFED userspace inside the image, [SR-IOV](https://en.wikipedia.org/wiki/Single-root_input/output_virtualization) VF injection on the pods, and the [NVIDIA Network Operator](https://docs.nvidia.com/networking/display/cokan10/network+operator) wiring up the NICs so NCCL can see them. Once that's done, every rank binds to all 16 `mlx5` VFs over RoCE — no Socket fallback in the NCCL logs, no overlay in the hot path.

**Configuration:**

- OKE (2 × A100 BM, 16 GPUs)
- TorchTitan (Llama 3 8B on C4)
- Monarch (controller-driven orchestration)
- TorchFT + Lighthouse (2 replicas, 8 GPUs each, per-step quorum)
- **RDMA** (DOCA-OFED + SR-IOV VFs, NCCL over RoCE)

**Results — `K8S Monarch Test` with `--torchft` and RDMA:**

| Date | Steps | Time | Loss | Status | MFU | TPS | TFLOPs | Grad Norm | Memory |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1000 | 2566 s | 12.23993 → 4.30832 | Success | **54.25%** | 3288 | 169.25 | 0.9664 | 55.12 GiB |

The NCCL logs confirm we got what we paid for: every rank binds to all 16 `mlx5` VFs over RoCE, with zero `NET/Socket` fallback. The overlay is out of the hot path.

#### What we got back

| Metric | + TorchFT (TCP overlay) | + TorchFT + RDMA | Delta |
|---|---|---|---|
| Wall-clock (1000 steps) | 6389 s | 2566 s | **−60% (2.49× speedup)** |
| MFU | 21.46% | 54.25% | **+32.8 pp** |
| TPS | 1301 | 3288 | +153% |
| TFLOPs / GPU | 66.97 | 169.25 | +153% |
| Memory | 55.12 GiB | 55.12 GiB | identical |
| Grad Norm | 1.2422 | 0.9664 | within noise |

And the more interesting comparison — RDMA-enabled TorchFT vs. the Monarch-only run from 3.2, which had *no* inter-replica traffic at all:

| Metric | Monarch only (3.2) | + TorchFT + RDMA (3.4) | Delta |
|---|---|---|---|
| Wall-clock (1000 steps) | 2505 s | 2566 s | +61 s (+2.4%) |
| MFU | 55.49% | 54.25% | −1.24 pp |
| TPS | 3364 | 3288 | −2.3% |
| TFLOPs / GPU | 173.13 | 169.25 | −2.2% |

This is the headline result of the whole section: **RDMA effectively erases the TorchFT overhead.** The 2.5× slowdown from 3.3 wasn't TorchFT's fault — it was the network. Move the same allreduce onto the RDMA fabric and we're back within ~2% of a run that doesn't do cross-replica sync at all. Per-step fault tolerance is now nearly free.

A few notes on what the numbers do and don't say:

- **The residual ~2% gap is real, not noise.** TorchFT still does *some* extra work every step: a Lighthouse quorum round-trip (~0.2 s/step) and the cross-replica allreduce itself, which is now bandwidth-bound by the RDMA fabric instead of the overlay. At ~2 GB per rank over RoCE that's well under a tenth of a second, and the quorum RPC accounts for the rest. We'll take it.
- **Memory is unchanged at 55.12 GiB.** RDMA changes the transport, not what TorchFT keeps in memory — the shadow gradient copy from 3.3 is still there. Expected.
- **Loss and grad-norm trajectories continue to match the non-FT baseline.** 12.24 → 4.31, grad-norm settling around 0.97. The training math hasn't changed; we've only sped up the bytes between replicas.
- **The TorchFT-off RDMA runs are flat vs. 3.2, as expected.** We also ran `--torchft`-off configurations with RDMA on (2499 s Monarch, 2505 s non-Monarch) — basically identical to 3.2's 2505 s. That's the right answer: when there's no cross-replica traffic, turning RDMA on can't help. The win only shows up the moment TorchFT puts gradients on the wire.

With this in place, the system finally has what we set out to build: Monarch's single-controller orchestration, TorchTitan's training stack, TorchFT's per-step fault tolerance, **and** a network that doesn't punish us for using any of it.