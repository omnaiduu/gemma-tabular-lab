"""Local data preparation and experiment entrypoints."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", required=True, type=Path)
    prepare.add_argument("--output", default="data/processed", type=Path)
    prepare.add_argument("--train-limit", default=10000, type=int)
    prepare.add_argument("--holdout-limit", default=1000, type=int)
    prepare.add_argument("--seed", default=42, type=int)
    baseline = commands.add_parser("baselines")
    baseline.add_argument("--data", default="data/processed", type=Path)
    baseline.add_argument("--output", default="artifacts/baselines", type=Path)
    baseline.add_argument("--seed", default=42, type=int)
    gpu = commands.add_parser("train")
    gpu.add_argument("--data", default="data/processed", type=Path)
    gpu.add_argument("--output", required=True, type=Path)
    gpu.add_argument("--head", choices=("digits", "regression"), required=True)
    gpu.add_argument("--adaptation", choices=("frozen", "lora"), default="lora")
    gpu.add_argument("--model-id", default="google/gemma-4-E4B")
    gpu.add_argument("--revision", default="main")
    gpu.add_argument("--steps", default=300, type=int)
    gpu.add_argument("--batch-size", default=1, type=int)
    gpu.add_argument("--max-length", default=256, type=int)
    gpu.add_argument("--max-seconds", default=3000, type=int)
    gpu.add_argument("--seed", default=42, type=int)
    gpu.add_argument("--numerical-weight", default=0.0, type=float)
    gpu.add_argument("--precision", choices=("auto", "fp16", "bf16"), default="auto")
    gpu.add_argument("--eval-batch-size", default=8, type=int)
    gpu.add_argument("--accumulation-steps", default=1, type=int)
    gpu.add_argument("--validation-limit", default=256, type=int)
    gpu.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare as run
        result = run(args.source, args.output, args.train_limit, args.holdout_limit, args.seed)
    elif args.command == "baselines":
        from .baselines import run_baselines
        result = run_baselines(args.data, args.output, args.seed)
    else:
        from .train import train
        result = train(args.data, args.output, args.head, args.adaptation, args.model_id, args.revision,
                       args.seed, args.steps, args.max_length, args.numerical_weight, args.max_seconds, args.batch_size,
                       precision=args.precision, eval_batch_size=args.eval_batch_size,
                       accumulation_steps=args.accumulation_steps, validation_limit=args.validation_limit,
                       evaluate_test=not args.skip_test)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
