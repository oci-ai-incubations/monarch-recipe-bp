# Configuration: OKE + TorchTitan + Monarch

[Deployment instructions](./Deployment.md)

Now we drop in **Monarch** to the [basline configuration](../k8s_titan_torchft_non_monarch/README.md). The training code doesn't change at all. What *does* change is how we launch it: instead of `torchrun` + manually-managed pods, we write a single Python controller that says "give me 2 hosts × 8 GPUs, run this code on each rank, and tell me when it's done." Monarch's Kubernetes operator turns that into pods, schedules them, and streams logs back to the controller.

**Configuration:**

- OKE (2 × A100 BM, 16 GPUs)
- TorchTitan (Llama 3 8B on C4)
- **Monarch** (controller-driven orchestration)

**Results — `K8S Monarch Test (no built-in failures)`:**

| Date | Steps | Time | Loss | Status | MFU | TPS | TFLOPs | Grad Norm | Memory |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1000 | 2505 s | 12.26616 → 4.64640 | Success | **55.49%** | 3364 | 173.13 | 1.0923 | 50.26 GiB |

#### What we gained operationally

This is the part that's hard to put in a metrics table but matters enormously day-to-day:

- **Single controller script** describes the whole job — host count, GPUs per host, image, command, environment. No more juggling manifests across nodes.
- **Trivial scaling.** Change `num_hosts=2` to `num_hosts=8` and Monarch handles the rest. Same script.
- **Centralized failure surface.** When something goes wrong, there's *one* place to look. (And as we'll see in later sections, this is the hook that TorchFT plugs into.)
- **Portability with minimal porting effort.** Moving the same controller to Slurm only takes swapping the K8s pod-spec helper for a Slurm allocator helper at the top of the script. The training loop, actor definitions, and TorchFT wiring stay byte-for-byte the same.

#### What we paid in performance

This is the question everyone asks: *does the extra orchestration layer slow training down?*

The honest answer from our data: **no, not measurably.**

| Metric | `torchrun` baseline | With Monarch | Delta |
|---|---|---|---|
| MFU | 55.34% | 55.49% | +0.15 pp |
| TPS | 3355 | 3364 | +0.3% |
| TFLOPs / GPU | 172.68 | 173.13 | +0.3% |
| Wall-clock (1000 steps) | 2473 s | 2505 s | +32 s (+1.3%) |
| Grad Norm | 1.0655 | 1.0923 | within noise |
| Memory | 50.26 GiB | 50.26 GiB | identical |

The two runs are statistically indistinguishable. That's a great result: we get all of Monarch's operational benefits **for free**. The orchestration layer adds no overhead to the hot path — once training starts, the workers don't know or care that Monarch launched them.

#### A few honest notes on the numbers

- **MFU at 55% leaves some headroom.** A100 can push 60–65% on Llama with aggressive global batches, activation-checkpointing tuning, or FlashAttention-3. We're not there yet, but we have ~30 GiB of memory headroom to spend if we want to chase it.
- **RDMA is OFF in both runs.** Our cluster supports RDMA, but these specific runs went over TCP/IP for inter-node collectives. At 16 GPUs with a compute-bound workload it didn't bite us — that's why MFU stayed at 55%. But it'll matter as we scale past two nodes, and it's the next thing on our list.
- **Monarch vs. `torchrun` parity is the real headline.** Anytime you add an orchestration layer, the worry is "great, but at what throughput cost?" In this case the answer is: none. That parity is what makes the rest of the system — TorchFT, topology-aware scheduling, recoverable replicas — viable to build on top.