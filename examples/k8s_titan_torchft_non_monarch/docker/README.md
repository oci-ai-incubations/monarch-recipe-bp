# Building Docker Images for OCI Example

## Building Nvidia Docker Image (with RDMA support)

```bash
cd examples/k8s_titan_torchft_non_monarch/docker

podman machine start
podman build -f Dockerfile-nvidia-8B -t my_monarch:titan-torchft-nvidia-8b-01 .
podman tag my_monarch:titan-torchft-nvidia-8b-01 ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-01

podman login ghcr.io
podman push ghcr.io/dochakov-oci/monarch-oci:titan-torchft-nvidia-8b-01
```