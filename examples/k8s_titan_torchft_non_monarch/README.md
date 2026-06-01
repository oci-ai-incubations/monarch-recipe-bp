# Configuration: OKE + TorchTitan + `torchrun` + TorchFT + RDMA

[Deployment instructions](./Deployment.md)

The simplest thing that could possibly work: TorchTitan launched directly via `torchrun` across our two A100 nodes, with no Monarch and no TorchFT in the picture. Pods are provisioned by a plain `provision.yaml`, `torchrun` is the rendezvous mechanism, and that's it.

**Configuration:**

- OKE (2 × A100 BM, 16 GPUs)
- TorchTitan (Llama 3 8B on C4)
- `torchrun` launcher

**Results — `K8S Non-Monarch Test`:**

| Date | Steps | Time | Loss | Status | MFU | TPS | TFLOPs | Grad Norm | Memory |
|---|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1000 | 2473 s | 12.24577 → 4.64120 | Success | **55.34%** | 3355 | 172.68 | 1.0655 | 50.26 GiB |

A few quick takeaways:

- **MFU of 55.3% is excellent for A100.** Dense BF16 peak on A100 is ~312 TFLOPs; we're sustaining ~173 TFLOPs per GPU. Typical TorchTitan Llama runs on A100 land in the 40–55% range, so we're at the top of the band right out of the gate.
- **Loss curve looks healthy** — going from ~12.25 down to ~4.64 over 1000 steps is exactly what you'd expect from a fresh init.
- **Grad norm settles around 1.07,** comfortably in the 1–2 range that indicates stable training: no explosion, no collapse.
- **Memory at 50 GiB / 80 GiB** leaves real headroom. If we wanted to push MFU further, larger global batches are on the table.
- **RDMA is OFF here.** Inter-node traffic falls back to the K8s pod overlay, but at this scale + this workload the compute-bound steps hide it. That changes the moment we introduce per-step cross-replica collectives — keep this in mind, it becomes the headline story in 3.3.

So the baseline is healthy. But operationally it's painful — every config change means hand-editing manifests, restarting pods, and re-running rendezvous. Let's fix that.