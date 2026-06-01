# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vanilla (non-Monarch) TorchTitan + TorchFT training entrypoint.

Mirrors the FaultTolerantTrainer.Config built by
``examples/k8s_titan_torchft_monarch/deployment_files/nvidia/controller.py``, but is
launched directly via ``torchrun`` instead of being orchestrated by Monarch.

Layout: one torchrun job per replica. Each torchrun forms its own torch
distributed process group (sharded data-parallel across the replica's GPUs).
Replicas DO NOT join the same torch dist group — they coordinate only through
the TorchFT Lighthouse identified by the ``TORCHFT_LIGHTHOUSE`` env var.

Required env vars (set by torchrun):
    RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
Required env var (set by the user, points at lighthouse.py):
    TORCHFT_LIGHTHOUSE   e.g. http://<node-1-ip>:29510

CLI flags select the replica id and forward training-step / dataset / tokenizer
overrides.
"""

import argparse
import os

import torch
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    TrainingConfig,
)
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.llama3 import model_registry
from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import ProfilingConfig


# === Model selection — comment / uncomment ONE line ===
# debugmodel: ~6M params, vocab 2048 (uses the small test tokenizer baked into
# the image at /opt/torchtitan/tests/assets/tokenizer).
# Any non-debug flavor uses the real Llama3 tokenizer (vocab 128256), which
# must be present at /opt/torchtitan/tokenizers/llama3 — install it via
# docker/Dockerfile-nvidia-tokenizer.
MODEL_FLAVOR = "8B"
# MODEL_FLAVOR = "debugmodel"

_DEFAULT_TOKENIZER_PATH = {
    "debugmodel": "/opt/torchtitan/tests/assets/tokenizer",
    "1B": "/opt/torchtitan/tokenizers/llama3",
    "3B": "/opt/torchtitan/tokenizers/llama3",
    "8B": "/opt/torchtitan/tokenizers/llama3",
    "70B": "/opt/torchtitan/tokenizers/llama3",
}[MODEL_FLAVOR]

# debugmodel is a tiny scratchpad — high LR is fine. Real Llama3 flavors need
# something closer to the Llama3 pretraining LR (~3e-4) or the loss diverges.
_DEFAULT_LR = {
    "debugmodel": 8e-4,
    "1B": 3e-4,
    "3B": 3e-4,
    "8B": 3e-4,
    "70B": 1.5e-4,
}[MODEL_FLAVOR]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vanilla torchrun launcher for TorchTitan + TorchFT"
    )
    parser.add_argument(
        "--replica-id",
        type=int,
        required=True,
        help="0-indexed replica id (0 .. replica-count - 1). Must be unique per torchrun job.",
    )
    parser.add_argument(
        "--replica-count",
        type=int,
        default=2,
        help="Total number of replicas joining the lighthouse (default: 2)",
    )
    parser.add_argument(
        "--gpus-per-host",
        type=int,
        default=8,
        help="GPUs per host — used as fault_tolerance.group_size (default: 8)",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10000,
        help="Number of training steps (default: 10000)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=_DEFAULT_TOKENIZER_PATH,
        help=(
            f"Path to tokenizer directory (must exist on every node). "
            f"Default for {MODEL_FLAVOR!r}: {_DEFAULT_TOKENIZER_PATH}"
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Absolute path to a local dataset directory. If unset, downloads c4 from HuggingFace.",
    )
    parser.add_argument(
        "--torchft",
        action="store_true",
        help=(
            "Enable TorchFT replica-level fault tolerance (default: disabled). "
            "When omitted, TORCHFT_LIGHTHOUSE is not required and each replica "
            "trains independently (no cross-replica gradient sync). Useful for "
            "measuring the TorchFT overhead."
        ),
    )
    return parser.parse_args()


def build_trainer_config(args: argparse.Namespace) -> FaultTolerantTrainer.Config:
    data_parallel_shard_degree = args.gpus_per_host

    # FTOptimizersContainer dereferences ft_manager.manager unconditionally in
    # its __init__, which asserts when FaultTolerance.enable is False. The
    # FaultTolerantTrainer.__init__ already branches on the optimizer Config
    # type — pass the plain OptimizersContainer.Config when TorchFT is off.
    optimizer_config = (
        FTOptimizersContainer.Config(lr=_DEFAULT_LR)
        if args.torchft
        else OptimizersContainer.Config(lr=_DEFAULT_LR)
    )

    return FaultTolerantTrainer.Config(
        hf_assets_path=args.tokenizer_path,
        profiling=ProfilingConfig(),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        model_spec=model_registry(MODEL_FLAVOR),
        optimizer=optimizer_config,
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=200,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=4096,
            steps=args.training_steps,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4" if args.dataset_path is None else "c4_test",
            dataset_path=args.dataset_path,
        ),
        checkpoint=CheckpointManager.Config(),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        comm=CommConfig(train_timeout_seconds=300),
        fault_tolerance=FaultTolerance(
            enable=args.torchft,
            replica_id=args.replica_id,
            group_size=data_parallel_shard_degree,
            process_group="nccl",
            process_group_timeout_ms=180000,
        ),
    )


def main() -> None:
    init_logger()
    args = parse_args()

    if args.torchft and "TORCHFT_LIGHTHOUSE" not in os.environ:
        raise RuntimeError(
            "--torchft requires TORCHFT_LIGHTHOUSE. Start lighthouse.py on one "
            "node and export TORCHFT_LIGHTHOUSE=http://<that-node-ip>:29510 on "
            "every node. (Or omit --torchft to run the replica independently "
            "without TorchFT replica-sync.)"
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    lighthouse_desc = (
        os.environ["TORCHFT_LIGHTHOUSE"] if args.torchft else "<disabled: --torchft not passed>"
    )
    logger.info(
        f"[replica_{args.replica_id}_trainer_{rank}] starting "
        f"(local_rank={local_rank}, world_size={world_size}, "
        f"lighthouse={lighthouse_desc})"
    )

    trainer_config = build_trainer_config(args)
    trainer = trainer_config.build()
    logger.info(
        f"[replica_{args.replica_id}_trainer_{rank}] initialized successfully on {os.getpid()}"
    )

    try:
        logger.info(f"[replica_{args.replica_id}_trainer_{rank}] starting training")
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info(f"[replica_{args.replica_id}_trainer_{rank}] trainer cleaned up")


if __name__ == "__main__":
    main()
