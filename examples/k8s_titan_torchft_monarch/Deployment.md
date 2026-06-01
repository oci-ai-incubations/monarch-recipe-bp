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

## Launch training via Titan TorchFT Example Controller

```bash
kubectl exec -it monarch-controller -n monarch-tests -- \
  python /tmp/controller.py --namespace monarch-tests \
    --training-steps 100 \
    --image ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-01 \
    --replica-count 2 --gpus-per-host 8
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
kubectl delete -f provision.yaml
kubectl delete namespace monarch-tests
helm uninstall monarch-operator monarch-operator/monarch-operator --namespace monarch-system
```
