"""Bounded head-only or LoRA training. Imports heavy dependencies only on use."""

import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from .codec import decode, encode
from .data import read_split
from .metrics import price_metrics


def training_batches(rows, batch_size, seed):
    """Shuffle each epoch independently while visiting every row once per epoch."""
    if not rows or batch_size <= 0:
        raise ValueError("Non-empty rows and a positive batch size are required")
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    while True:
        rng.shuffle(indices)
        for offset in range(0, len(indices), batch_size):
            yield [rows[index] for index in indices[offset:offset + batch_size]]


def train(data_dir, output_dir, head="digits", adaptation="lora", model_id="google/gemma-4-E4B",
          revision="main", seed=42, steps=300, max_length=256, numerical_weight=0.0,
          max_seconds=3000, batch_size=1):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoModel, AutoTokenizer
    from .heads import NumericalModel, digit_loss

    if head not in ("digits", "regression") or adaptation not in ("frozen", "lora"):
        raise ValueError("Unsupported experiment variant")
    if steps <= 0 or max_length < 32 or max_seconds < 60 or numerical_weight < 0 or batch_size <= 0:
        raise ValueError("Invalid training bounds")
    if not torch.cuda.is_available():
        raise RuntimeError("The Gemma experiment requires a GPU")
    start = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows = {split: read_split(data_dir, split) for split in ("train", "validation", "test")}
    log_targets = np.log1p([row["target"] for row in rows["train"]])
    target_mean, target_std = float(log_targets.mean()), max(float(log_targets.std()), 1e-6)
    print(json.dumps({"event": "loading_model", "model_id": model_id, "revision": revision}), flush=True)
    load_start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    root = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                     device_map={"": "cuda"}, attn_implementation="sdpa")
    resolved_revision = getattr(root.config, "_commit_hash", None)
    backbone = getattr(root, "language_model", None)
    if backbone is None:
        raise RuntimeError("Checkpoint did not expose the expected text backbone; inspect the model before training")
    del root
    torch.cuda.empty_cache()
    model_load_seconds = time.monotonic() - load_start
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    if adaptation == "lora":
        from peft import LoraConfig, get_peft_model
        backbone = get_peft_model(backbone, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hidden_size = backbone.config.hidden_size
    model = NumericalModel(backbone, hidden_size, head).to("cuda")
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": 5e-4},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 2e-4},
    ], weight_decay=0.01)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(json.dumps({"event": "model_ready", "model_load_seconds": model_load_seconds,
                      "trainable_parameters": trainable, "gpu": torch.cuda.get_device_name()}), flush=True)

    class PredictionDeadline(RuntimeError):
        pass

    def batch(examples):
        tokens = tokenizer([row["text"] for row in examples], padding=True, truncation=True,
                           max_length=max_length, return_tensors="pt")
        return {key: tokens[key].to("cuda") for key in ("input_ids", "attention_mask")}

    def predict(examples):
        values, slot_predictions = [], []
        model.eval()
        with torch.no_grad():
            for offset in range(0, len(examples), 8):
                if time.monotonic() - start > max_seconds - 60:
                    raise PredictionDeadline("Insufficient bounded time for complete evaluation")
                scores = model(**batch(examples[offset:offset+8]))
                if head == "digits":
                    choices = torch.stack([s.argmax(-1) for s in scores], dim=1).cpu().tolist()
                    # Marketplace prices are non-negative; sign constraint is part of this task's schema.
                    for choice in choices:
                        choice[0] = 0
                        values.append(decode(choice))
                    slot_predictions.extend(choices)
                else:
                    transformed = scores.float().cpu().numpy() * target_std + target_mean
                    values.extend(np.maximum(0, np.expm1(np.clip(transformed, -20, 230))).tolist())
        return values, slot_predictions

    validation = random.Random(seed).sample(rows["validation"], min(256, len(rows["validation"])))
    history, best, completed_steps, examples_processed = [], float("inf"), 0, 0
    seen_training_rows = set()
    best_path = output / "head.pt"
    batches = training_batches(rows["train"], batch_size, seed)
    training_start = time.monotonic()
    for step in range(steps):
        # Leave time for checkpoint saving and reporting; Modal also enforces an outer timeout.
        if time.monotonic() - start > max_seconds - 180:
            break
        model.train()
        if adaptation == "frozen":
            model.backbone.eval()
        examples = next(batches)
        seen_training_rows.update(example["row_id"] for example in examples)
        scores = model(**batch(examples))
        target = torch.tensor([example["target"] for example in examples], device="cuda", dtype=torch.float32)
        if head == "digits":
            categorical_target = torch.tensor([encode(example["target"]) for example in examples], device="cuda", dtype=torch.long)
            loss = digit_loss(scores, categorical_target, target, numerical_weight)
        else:
            scaled = (torch.log1p(target) - target_mean) / target_std
            loss = F.mse_loss(scores.float(), scaled)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        completed_steps = step + 1
        examples_processed += len(examples)
        if completed_steps % 100 == 0 or completed_steps == steps:
            try:
                predicted, _ = predict(validation)
            except PredictionDeadline:
                break
            metrics = price_metrics([row["target"] for row in validation], predicted)
            history.append({"step": completed_steps, "loss": float(loss.detach()), "validation": metrics})
            print(json.dumps(history[-1]), flush=True)
            if metrics["rmsle"] < best:
                best = metrics["rmsle"]
                torch.save(model.head.state_dict(), best_path)
                if adaptation == "lora":
                    model.backbone.save_pretrained(output / "adapter")
    if not best_path.exists():
        # A bounded run with fewer than 100 steps still has a checkpoint and clearly records its coverage.
        torch.save(model.head.state_dict(), best_path)
        if adaptation == "lora":
            model.backbone.save_pretrained(output / "adapter")
    model.head.load_state_dict(torch.load(best_path, weights_only=True))
    if adaptation == "lora":
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model.backbone, load_file(str(output / "adapter" / "adapter_model.safetensors")))
    report = {
        "model_id": model_id, "requested_revision": revision, "resolved_revision": resolved_revision,
        "head": head, "adaptation": adaptation, "seed": seed, "completed_steps": completed_steps,
        "unique_training_rows_seen": len(seen_training_rows), "available_training_rows": len(rows["train"]),
        "training_batch_size": batch_size, "training_examples_processed": examples_processed,
        "requested_steps": steps, "numerical_weight": numerical_weight,
        "target_log_mean": target_mean, "target_log_std": target_std,
        "max_length": max_length, "trainable_parameters": trainable,
        "model_load_seconds": model_load_seconds,
        "training_and_validation_seconds": time.monotonic() - training_start,
        "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
        "head_parameters": sum(p.numel() for p in model.head.parameters()),
        "gpu": torch.cuda.get_device_name(), "history": history,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
    }
    # Always save the training record before potentially costly final evaluation.
    (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    if time.monotonic() - start < max_seconds - 120:
        torch.cuda.synchronize()
        evaluation_start = time.perf_counter()
        try:
            predicted, choices = predict(rows["test"])
        except PredictionDeadline:
            report["test_status"] = "not completed: evaluation deadline reached"
            report["total_seconds"] = time.monotonic() - start
            report["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        torch.cuda.synchronize()
        evaluation_seconds = time.perf_counter() - evaluation_start
        report["test"] = price_metrics([row["target"] for row in rows["test"]], predicted)
        report["evaluation_batch_size"] = 8
        report["evaluation_rows_per_second"] = len(predicted) / max(evaluation_seconds, 1e-9)
        report["encoding_quantization_metrics"] = price_metrics([row["target"] for row in rows["test"]],
                                                               [decode(encode(row["target"])) for row in rows["test"]])
        if choices:
            actual_choices = [encode(row["target"]) for row in rows["test"]]
            report["test_slot_accuracy"] = [sum(a[i] == p[i] for a, p in zip(actual_choices, choices)) / len(choices) for i in range(8)]
        with (output / "predictions.jsonl").open("w") as handle:
            for row, value in zip(rows["test"], predicted):
                handle.write(json.dumps({"row_id": row["row_id"], "actual": row["target"], "predicted": value,
                                         "category": row["features"]["category_name"]}) + "\n")
        latency = []
        for example in rows["test"][:20]:
            if time.monotonic() - start > max_seconds - 65:
                break
            torch.cuda.synchronize()
            begin = time.perf_counter()
            predict([example])
            torch.cuda.synchronize()
            latency.append((time.perf_counter() - begin) * 1000)
        if latency:
            report["warm_batch1_p50_ms"] = float(np.percentile(latency, 50))
            report["warm_batch1_p95_ms"] = float(np.percentile(latency, 95))
    else:
        report["test_status"] = "not run: bounded job has insufficient remaining time"
    report["total_seconds"] = time.monotonic() - start
    report["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
