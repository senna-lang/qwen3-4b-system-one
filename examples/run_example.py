"""Run fictional example requests through the adapter and print per-branch probabilities.

    python examples/run_example.py --backend torch
    python examples/run_example.py --backend mlx --adapter /path/to/local/adapter
"""
from __future__ import annotations

import argparse
from pathlib import Path

from system_one.runner import load_backend, read_requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--adapter", help="local adapter directory (default: download from the Hub)")
    parser.add_argument("--input", default=Path(__file__).with_name("requests.jsonl"), type=Path)
    args = parser.parse_args()

    model, tokenizer, predict_request = load_backend(args.backend, args.adapter)
    for request_id, request in read_requests(args.input):
        print(f"# {request_id}")
        for branch, probabilities in zip(request.branches, predict_request(model, tokenizer, request)):
            labels = branch.options or ["No", "Yes"]
            best = max(range(len(labels)), key=probabilities.__getitem__)
            dist = ", ".join(f"{label}={p:.3f}" for label, p in zip(labels, probabilities))
            print(f"  [{branch.kind}] {branch.instructions}\n    -> {labels[best]}  ({dist})")


if __name__ == "__main__":
    main()
