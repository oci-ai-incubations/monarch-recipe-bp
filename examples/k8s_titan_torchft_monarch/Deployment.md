# Monarch Titan TorchFT Example Installation on OKE Cluster

## Install the MonarchMesh CRD and operator using Helm

```bash
helm repo add monarch-operator https://meta-pytorch.github.io/monarch-kubernetes

helm repo update

helm install monarch-operator monarch-operator/monarch-operator \
  --namespace monarch-system \
  --create-namespace
```

## Install the Titan TorchFT Example Controller

```bash
# Check GPU Shapes
kubectl get nodes -o custom-columns=NAME:.metadata.name,SHAPE:.metadata.labels.'beta\.kubernetes\.io/instance-type',SHAPE_NEW:.metadata.labels.'node\.kubernetes\.io/instance-type',OCI_SHAPE:.metadata.labels.'oci\.oraclecloud\.com/shape'

# Deploy Titan TorchFT Example Controller infra

cd examples/k8s_titan_torchft_monarch/deployment_files/nvidia

kubectl apply -f provision.yaml

# Deploy Titan TorchFT Example Controller app

kubectl cp controller.py monarch-tests/monarch-controller:/tmp/controller.py
```

## (Optional) Install Kueue, Topology and Apply Resource Quota

There are 2 purposes of Kueue:

1. Gang Scheduling — training pods are admitted only when all replicas can run together
2. RDMA Topology Aware Scheduling

For more details about Kueue and TAS, check [TAS Testing document](../../tas).

For Kueue installation instructions check [Kueue document](../../tas/kueue.md).

**Provision Topology**

```bash
kubectl apply -f deployment_files/nvidia/topology.yaml
```

**Provision Resource quota**

```bash
kubectl apply -f deployment_files/nvidia/kueue_quota.yaml
```

## Launch training via Titan TorchFT Example Controller

Add `--kueue user-queue` to the command below if you applied `kueue_quota.yaml` in the previous section. The flag labels worker pods so Kueue admits them through the `user-queue` LocalQueue; without it, pods bypass Kueue entirely.

**Gang Scheduling Test (optional)**

```bash
# 1. Apply the dummy → 2 of 16 GPUs admitted in Kueue
kubectl apply -f examples/k8s_ddp_cnn/deployment_files/nvidia/dummy-kueue-consume-2-gpu.yaml

# 2. Confirm it's Admitted and consuming quota
kubectl get workloads -n monarch-tests
kubectl get clusterqueue a100-cluster-queue -o jsonpath='{.status.flavorsUsage}' | jq

# 3. Now launch the training (full command is documented below)
# kubectl exec -it monarch-controller -n monarch-tests -- \
#    python /tmp/controller.py --namespace monarch-tests \
#      --training-steps 100 \
#      --image ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-rdma-01 \
#      --replica-count 2 --gpus-per-host 8 \
#      --kueue user-queue

# 4. In other 2 shells, watch — replica pods should NOT run:
kubectl get workloads -n monarch-tests -w
kubectl get pods -n monarch-tests -o wide -w

# Expected output: 2 pods stay in SchedulingGated state (Kueue holds the
# 16-GPU Workload because only 14 GPUs are free).
kubectl get pods -n monarch-tests -o wide
NAME                              READY   STATUS            RESTARTS   AGE     IP             NODE         NOMINATED NODE   READINESS GATES
dummy-kueue-consume-2-gpu-7s9tk   1/1     Running           0          7m2s    10.244.2.105   10.0.66.66   <none>           <none>
monarch-controller                1/1     Running           0          7m55s   10.244.2.104   10.0.66.66   <none>           <none>
replica0-0                        0/1     SchedulingGated   0          6m26s   <none>         <none>       <none>           <none>

# 5. Free the 2 GPUs → Kueue admits the Monarch job, both replicas come up together
kubectl delete -f examples/k8s_ddp_cnn/deployment_files/nvidia/dummy-kueue-consume-2-gpu.yaml

# Expected output:
kubectl get pods -n monarch-tests
NAME                 READY   STATUS    RESTARTS   AGE
monarch-controller   1/1     Running   0          18m
replica0-0           1/1     Running   0          2m6s
replica1-0           1/1     Running   0          2m6s
```

**Launch (without Kueue):**

```bash
kubectl exec -it monarch-controller -n monarch-tests -- \
  python /tmp/controller.py --namespace monarch-tests \
    --training-steps 100 \
    --image ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-rdma-01 \
    --replica-count 2 --gpus-per-host 8
```

**Launch (with Kueue gang scheduling + TAS):**

```bash
kubectl exec -it monarch-controller -n monarch-tests -- \
  python /tmp/controller.py --namespace monarch-tests \
    --training-steps 100 \
    --image ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-rdma-01 \
    --replica-count 2 --gpus-per-host 8 \
    --kueue user-queue
```

## Debug and Monitor

```bash
kubectl get pods -n monarch-system -o wide
kubectl get pods -n monarch-tests -o wide

kubectl logs monarch-controller -n monarch-tests
kubectl logs replica0-0 -n monarch-tests

kubectl describe pods monarch-controller -n monarch-tests
kubectl describe pods replica0-0 -n monarch-tests

kubectl exec -it monarch-controller -n monarch-tests -- /bin/bash
kubectl exec -it replica0-0 -n monarch-tests -- /bin/bash

kubectl cp monarch-tests/monarch-controller:/tmp/root/monarch_log.log ~/Downloads/monarch_log.log
kubectl cp monarch-tests/replica0-0:/tmp/root/monarch_log.log ~/Downloads/replica_log.log
```

## Cleanup

```bash
# If you applied Kueue, delete those resources first
kubectl delete -f deployment_files/nvidia/kueue_quota.yaml
kubectl delete -f deployment_files/nvidia/topology.yaml

kubectl delete -f provision.yaml
kubectl delete namespace monarch-tests
helm uninstall monarch-operator monarch-operator/monarch-operator --namespace monarch-system

# If you installed Kueue plugin, uninstall it
helm uninstall kueue --namespace kueue-system
```
