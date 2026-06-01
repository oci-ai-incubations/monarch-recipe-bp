# Configuration: OKE + TorchTitan + Monarch + TorchFT + RDMA + Kueue

[Deployment instructions](./Deployment.md)

[The previously configured system](https://github.com/oci-ai-incubations/monarch-recipe-bp/blob/cfg4_oke_torchtitan_monarch_torchft_rdma/examples/k8s_titan_torchft_monarch/README.md) is fast and fault-tolerant. What it isn't yet is *well-behaved on a shared cluster*. Two scheduling failure modes still bite us:

1. **Partial admission.** Submit a 16-GPU job to a cluster with 14 free GPUs and Kubernetes happily places the 8 GPUs it can. The other pod stalls in `Pending`. We've now claimed half the resources for a job that **cannot make any forward progress**, while blocking a smaller job that *could* run.
2. **Topology-blind placement.** The default scheduler will happily put our two replicas on nodes that are several RDMA hops apart even when topologically-adjacent nodes are free. On the 2-node configuration in this article that doesn't yet matter, but at any real scale the cross-block hops show up as a steady tax on every TorchFT allreduce.

This section is the final touch on the configuration — we add [**Kueue**](https://kueue.sigs.k8s.io/) to fix both at once.

**Configuration:**

- OKE
- TorchTitan
- Monarch
- TorchFT
- RDMA
- **Kueue** (gang scheduling + RDMA topology-aware scheduling)

#### What Kueue buys us

**Gang scheduling.** Kueue treats the whole multi-replica job as a single *Workload* and admits it atomically. Either all 16 GPUs are available and every replica pod starts together, or **no** pods are scheduled and the workload sits in the queue. There is no in-between state where half a job is sitting on the cluster doing nothing while holding GPUs hostage. The replicas that do get admitted are guaranteed to find each other and start training; the ones that don't simply wait their turn.

**RDMA topology-aware scheduling (TAS).** OKE labels each bare-metal node with its RDMA topology coordinates — HPC island, network block, local block, and hostname:

```yaml
levels:
- nodeLabel: "oci.oraclecloud.com/rdma.hpc_island_id"
- nodeLabel: "oci.oraclecloud.com/rdma.network_block_id"
- nodeLabel: "oci.oraclecloud.com/rdma.local_block_id"
- nodeLabel: "kubernetes.io/hostname"
```

Kueue reads those labels via the `Topology` CRD and, when admitting a workload, picks node combinations that minimize the depth at which replicas share a parent in the topology tree. Two pods on the same local block (sharing a leaf switch) beats two pods in the same network block (one hop up) which beats two pods in different islands (worst case). For a TorchFT allreduce that's the difference between a few microseconds of switch latency and a multi-hop round trip on every step.

Both behaviors fall out of a single mechanism: pods are labeled with the Kueue queue name, and Kueue admits them through a `ClusterQueue` whose `ResourceFlavor` references the RDMA `Topology`. Same wiring, two wins.

#### Steps we ran for the gang-scheduling test

```bash
# 1. Apply the dummy → 2 of 16 GPUs admitted in Kueue
kubectl apply -f examples/k8s_ddp_cnn/deployment_files/nvidia/dummy-kueue-consume-2-gpu.yaml

# 2. Confirm it's Admitted and consuming quota
kubectl get workloads -n monarch-tests
kubectl get clusterqueue a100-cluster-queue -o jsonpath='{.status.flavorsUsage}' | jq

# 3. Launch the titan_torchft_monarch training (in another shell)
kubectl exec -it monarch-controller -n monarch-tests -- \
  python /tmp/controller.py --namespace monarch-tests \
    --training-steps 100 \
    --image ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-rdma-01 \
    --replica-count 2 --gpus-per-host 8 \
    --kueue user-queue

# 4. Watch — replica pods should NOT run while only 14/16 GPUs are free
kubectl get pods -n monarch-tests

# 5. Free the 2 GPUs → Kueue admits the Monarch job, replicas come up together
kubectl delete -f examples/k8s_ddp_cnn/deployment_files/nvidia/dummy-kueue-consume-2-gpu.yaml
```

#### Gang scheduling in action

While the dummy job holds 2 GPUs and only 14 of the cluster's 16 GPUs are free, the training workload's replica pods sit in `SchedulingGated` exactly as designed — Kueue refuses to admit the workload until all 16 GPUs can be claimed at once:

```bash
kubectl get pods -n monarch-tests
NAME                              READY   STATUS            RESTARTS   AGE
dummy-kueue-consume-2-gpu-7s9tk   1/1     Running           0          7m2s
monarch-controller                1/1     Running           0          7m55s
replica0-0                        0/1     SchedulingGated   0          6m26s
replica1-0                        0/1     SchedulingGated   0          6m26s
```

Both replica pods stay gated — no IP, no node, no half-claimed GPUs. As soon as we delete the dummy job and the cluster has the full 16 GPUs available, Kueue admits the workload and both replicas come up together. **That's the contract:** the training either has every GPU it asked for, or it has none of them — but it never holds GPUs it can't use.

The training begins right after deleting the dummy job that was holding 2 GPUs:

```bash
kubectl get pods -n monarch-tests
NAME                              READY   STATUS            RESTARTS   AGE
monarch-controller                1/1     Running           0          11m66s
replica0-0                        1/1     Running           0          10m37s
replica1-0                        1/1     Running           0          10m37s
```

## Where this leaves us

This is the final touch on the configuration. Nothing in this section changed the training math or the per-step wall-clock — Kueue is invisible during steady-state training. What it changes is how the system behaves at the *edges*: how jobs land on the cluster, how they share it with other jobs, and where on the RDMA fabric the replicas land relative to one another. On a 2-node cluster the topology placement is degenerate (there's only one valid placement), but the same configuration carries straight over to larger clusters where the topology decisions actually start to matter.

Combined with everything we built up to 3.4, the system now has:

- a single Python controller (3.2 — Monarch),
- per-step fault tolerance with no restart cost (3.3 — TorchFT),
- inter-replica gradient sync that runs at line rate on the RDMA fabric (3.4 — RDMA),
- and atomic, topology-aware admission so jobs don't strand resources or hop across the fabric unnecessarily (3.5 — Kueue).

That's the system we set out to build.