# Non Monarch Titan TorchFT Example Installation on OKE Cluster

## Prerequisites

- An OKE cluster with at least 2 NVIDIA GPU bare-metal nodes
  (`BM.GPU.H100.8` or `BM.GPU.A100-v2.8`).
- `kubectl` configured for that cluster.
- The image baked into [`provision.yaml`](./deployment_files/nvidia/provision.yaml)
  (`ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-rdma-01`) must be
  reachable from the cluster. It contains `torchtitan` (with `experiments/ft`),
  `torchft`, and the Llama3 128k-vocab tokenizer at
  `/opt/torchtitan/tokenizers/llama3` — required for the non-debug Llama3
  flavor configured via `MODEL_FLAVOR` (currently "8B") in
  [`deployment_files/nvidia/train_titan_torchft.py`](./deployment_files/nvidia/train_titan_torchft.py).
  Build/push instructions: [`docker/README.md`](./docker/README.md).

> **No Monarch operator required.** This flow is intentionally
> Monarch-operator-free, so you do not need to `helm install monarch-operator`
> for it to work. The `monarch-tests` namespace is reused only for naming
> consistency with the rest of this repo.

## Part 1: Provision the pods

```bash
cd examples/k8s_titan_torchft_non_monarch/deployment_files/nvidia
kubectl apply -f provision.yaml

# Wait for both replicas + lighthouse to be Running
kubectl get pods -n monarch-tests -w
# NAME                  READY   STATUS    RESTARTS   AGE
# torchft-lighthouse    1/1     Running   0          30s
# vanilla-replica-0     1/1     Running   0          30s
# vanilla-replica-1     1/1     Running   0          30s
```

All three pods sleep — they do not start training automatically.

## Part 2: Copy the Python scripts into the pods

Same pattern as [`../k8s_titan_torchft_monarch/README.md`](../k8s_titan_torchft_monarch/README.md) uses for `controller.py`:

```bash
# Lighthouse pod
kubectl cp lighthouse.py monarch-tests/torchft-lighthouse:/tmp/lighthouse.py

# Replica pods
kubectl cp train_titan_torchft.py monarch-tests/vanilla-replica-0:/tmp/train_titan_torchft.py
kubectl cp train_titan_torchft.py monarch-tests/vanilla-replica-1:/tmp/train_titan_torchft.py
```

## Part 3: Start the Lighthouse

In a dedicated terminal (it stays in the foreground for the lifetime of the
job):

```bash
kubectl exec -it torchft-lighthouse -n monarch-tests -- \
  python /tmp/lighthouse.py --bind '[::]:29510' --min-replicas 1 --join-timeout-ms 60000
```

The Service `torchft-lighthouse` (defined in `provision.yaml`) routes
`http://torchft-lighthouse.monarch-tests.svc.cluster.local:29510` to this pod.
Both replica pods already have `TORCHFT_LIGHTHOUSE=http://torchft-lighthouse.monarch-tests.svc.cluster.local:29510`
set in their env (see `provision.yaml`), so they will discover the lighthouse
automatically.

## Part 4: Launch torchrun on each replica

In two more terminals — one per replica. Each call is a single-node torchrun
(`--nnodes=1`), but `--master_addr` **must** be the pod IP (`$POD_IP`,
exposed by `provision.yaml` via the K8s downward API), not `127.0.0.1`.

Why: torchrun's rendezvous TCPStore binds on `--master_addr`, and torchft
advertises that same `MASTER_ADDR:MASTER_PORT` to the Lighthouse as the
`store_address` the *other* replica must reach during cross-replica state
recovery. With `--master_addr=127.0.0.1`, the store binds only on loopback
and replica 1's `store.get('0')` hangs for 60 s, fails NCCL bootstrap, and
torchft refuses to commit step 1 — training stays stuck at the pre-failure
step.

Each command is wrapped in `sh -c '…'` so `$POD_IP` is expanded inside the
pod's shell, not on your laptop.

### Replica 0

```bash
kubectl exec -it vanilla-replica-0 -n monarch-tests -- sh -c '
  torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr="$POD_IP" \
    --master_port=29500 \
    /tmp/train_titan_torchft.py \
      --replica-id 0 \
      --replica-count 2 \
      --gpus-per-host 8 \
      --training-steps 100
'
```

### Replica 1

```bash
kubectl exec -it vanilla-replica-1 -n monarch-tests -- sh -c '
  torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --node_rank=0 \
    --master_addr="$POD_IP" \
    --master_port=29500 \
    /tmp/train_titan_torchft.py \
      --replica-id 1 \
      --replica-count 2 \
      --gpus-per-host 8 \
      --training-steps 100
'
```

## CLI flags (`train_titan_torchft.py`)

| Flag | Required | Default | Notes |
|---|---|---|---|
| `--replica-id` | yes | — | 0 .. `replica-count − 1`. Must be unique per torchrun job. Sets `fault_tolerance.replica_id`. |
| `--replica-count` | no | `2` | Total replicas joining the lighthouse. |
| `--gpus-per-host` | no | `8` | Used as `fault_tolerance.group_size` (matches the Monarch controller). |
| `--training-steps` | no | `10000` | Total training steps. |
| `--tokenizer-path` | no | `/opt/torchtitan/tests/assets/tokenizer` | Must exist inside the container image. |
| `--dataset-path` | no | _unset_ | If unset, downloads C4 from HuggingFace. If set, uses the local `c4_test` loader against this directory. |

## Debug & Monitor

```bash
kubectl get pods -n monarch-tests -o wide
kubectl logs torchft-lighthouse -n monarch-tests
kubectl logs vanilla-replica-0  -n monarch-tests
kubectl logs vanilla-replica-1  -n monarch-tests

kubectl exec -it vanilla-replica-0 -n monarch-tests -- /bin/bash
kubectl exec -it vanilla-replica-1 -n monarch-tests -- /bin/bash
```

`titan_torchft` logs contain ANSI color codes — strip them before grepping
for `step:` / `loss:`:

```bash
kubectl logs vanilla-replica-0 -n monarch-tests | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'step:|loss:'
```

## Cleanup

```bash
kubectl delete -f provision.yaml
```

This removes the namespace and every object created above. There is no Helm
release or CRD instance to clean up — `provision.yaml` is fully
self-contained.
