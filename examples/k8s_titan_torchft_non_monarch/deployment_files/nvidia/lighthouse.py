# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone TorchFT Lighthouse server.

Mirrors what the Monarch controller does inline:

    self.lighthouse = LighthouseServer(
        bind="[::]:0", min_replicas=1, join_timeout_ms=60000
    )

For the vanilla / non-Monarch flow we pin the bind port (default 29510, same
default as the slurm runner in torchft) so each replica's torchrun job can
discover the lighthouse via a fixed ``TORCHFT_LIGHTHOUSE=http://<host>:<port>``
URL. Run this on one of the nodes before launching the per-replica torchrun
jobs.
"""

import argparse
import os
import signal
import socket
import threading

# Quiet the torchft Rust loggers (per-step "Quorum status: ..." chatter). Must
# be set BEFORE the torchft import — env_logger / tracing-subscriber read
# RUST_LOG once at load time. Override with RUST_LOG=info (or =debug) before
# launching for failure-injection debugging.
os.environ.setdefault("RUST_LOG", "warn")

from torchft.coordination import LighthouseServer
from torchtitan.tools.logging import init_logger, logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TorchFT Lighthouse server")
    parser.add_argument(
        "--bind",
        type=str,
        default="[::]:29510",
        help="Bind address for the lighthouse (default: [::]:29510)",
    )
    parser.add_argument(
        "--min-replicas",
        type=int,
        default=1,
        help="Minimum replicas required to form a quorum (default: 1)",
    )
    parser.add_argument(
        "--join-timeout-ms",
        type=int,
        default=60000,
        help="Replica join timeout in ms (default: 60000)",
    )
    return parser.parse_args()


def main() -> None:
    init_logger()
    args = parse_args()

    lighthouse = LighthouseServer(
        bind=args.bind,
        min_replicas=args.min_replicas,
        join_timeout_ms=args.join_timeout_ms,
    )

    addr = lighthouse.address()
    fqdn = socket.getfqdn()
    short = socket.gethostname()
    public_addr = addr.replace(short, fqdn)
    logger.info(f"[Lighthouse] bound at {addr}")
    logger.info(f"[Lighthouse] export TORCHFT_LIGHTHOUSE={public_addr}")

    stop = threading.Event()

    def _shutdown(signum, frame):
        logger.info(f"[Lighthouse] received signal {signum}, shutting down")
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        stop.wait()
    finally:
        lighthouse.shutdown()
        logger.info("[Lighthouse] stopped")


if __name__ == "__main__":
    main()
