"""Resumable 101-class ContextMaskGIT V5 training on Food-101 tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from context_maskgit_v5 import initialize_context_maskgit_v5_from_v4, load_context_maskgit_v5
from models import erase_packets


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE22_CONTEXTUAL_TOKCOM_PROTOCOL.json"
CACHE = ROOT / "data/stage20_food101/token_cache_stage21"
OUT = ROOT / "runs/stage22_contextual_tokcom"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_history(path: Path, completed_step: int) -> None:
    if not path.exists():
        return
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    unique = {int(row["step"]): row for row in rows if int(row["step"]) <= completed_step}
    kept = [unique[index] for index in sorted(unique)]
    if [int(row["step"]) for row in kept] != list(range(1, completed_step + 1)):
        raise ValueError("Stage-22 history does not exactly cover the recovery checkpoint")
    path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")


def balanced_indices(records: list[dict], per_class: int, seed: int, classes: int = 101) -> torch.Tensor:
    selected = []
    for label in range(classes):
        candidates = [
            (index, row)
            for index, row in enumerate(records)
            if int(row["label"]) == label
        ]
        ranked = sorted(
            candidates,
            key=lambda item: hashlib.sha256(f"{seed}:{item[1]['image']}".encode()).digest(),
        )
        if len(ranked) < per_class:
            raise ValueError(f"Class {label} lacks enough development records")
        selected.extend(index for index, _ in ranked[:per_class])
    return torch.tensor(selected, dtype=torch.long)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def validate_inputs(protocol: dict) -> tuple[dict, dict, torch.Tensor, list[str]]:
    if protocol.get("status") != "frozen" or protocol.get("test_accessed") or protocol.get("test_pixels_extracted"):
        raise ValueError("Stage-22 protocol is not safely frozen")
    for name, expected in protocol["input_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen Stage-22 input changed: {name}")
    initialization = ROOT / protocol["initialization"]["checkpoint"]
    if sha256_file(initialization) != protocol["initialization"]["checkpoint_sha256"]:
        raise ValueError("Frozen Stage-22 initialization checkpoint changed")
    cache_summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
    if not cache_summary.get("completed") or cache_summary.get("test_accessed"):
        raise ValueError("Stage-21 token cache did not pass final audit")
    train = torch.load(CACHE / "train.pt", map_location="cpu", weights_only=True)
    dev = torch.load(CACHE / "dev.pt", map_location="cpu", weights_only=True)
    if train["identity"] != dev["identity"] or train["identity"]["tokenizer_sha256"] != protocol["tokenizer_sha256"]:
        raise ValueError("Stage-22 tokenizer identity mismatch")
    if train["ids"].shape != (70700, 256) or dev["ids"].shape != (5050, 256):
        raise ValueError("Stage-22 aggregate token count mismatch")
    records = read_jsonl(ROOT / "data/stage20_food101/dev.jsonl")
    indices = balanced_indices(
        records,
        int(protocol["validation"]["images_per_class"]),
        int(protocol["validation"]["subset_seed"]),
    )
    record_labels = torch.tensor([row["label"] for row in records], dtype=torch.int16)
    if not torch.equal(dev["labels"], record_labels):
        raise ValueError("Stage-22 development labels do not align with the frozen JSONL")
    embedding_payload = torch.load(ROOT / protocol["model"]["class_embeddings"], map_location="cpu", weights_only=True)
    return train, dev, indices, embedding_payload["class_names"]


@torch.inference_mode()
def evaluate(model, ids: torch.Tensor, labels: torch.Tensor, protocol: dict, seed: int) -> list[dict]:
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        missing = erase_packets(ids, float(protocol["validation"]["per"]), int(protocol["training"]["packet_tokens"]))
    conditions = [("correct", labels), ("wrong", (labels + 1) % model.classes), ("none", None)]
    results = []
    batch_size = int(protocol["validation"]["batch_size"])
    for mode, used_labels in conditions:
        total_ce = 0.0
        total_correct = 0
        total_missing = 0
        for start in range(0, len(ids), batch_size):
            target = ids[start : start + batch_size].long()
            mask = missing[start : start + batch_size]
            batch_labels = None if used_labels is None else used_labels[start : start + batch_size].long()
            received = target.masked_fill(mask, model.codes)
            logits = model(received, batch_labels)
            total_ce += float(F.cross_entropy(logits[mask], target[mask], reduction="sum"))
            total_correct += int((logits[mask].argmax(-1) == target[mask]).sum())
            total_missing += int(mask.sum())
        results.append({
            "mode": mode,
            "masked_ce": total_ce / total_missing,
            "one_pass_accuracy": total_correct / total_missing,
            "missing_tokens": total_missing,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.max_seconds < 0 or args.threads <= 0:
        parser.error("max-seconds must be nonnegative and threads positive")
    torch.set_num_threads(args.threads)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    train, dev, validation_indices, class_names = validate_inputs(protocol)
    embeddings_payload = torch.load(ROOT / protocol["model"]["class_embeddings"], map_location="cpu", weights_only=True)
    if class_names != embeddings_payload["class_names"]:
        raise ValueError("Stage-22 class names changed between validation and model construction")
    training = protocol["training"]
    torch.manual_seed(int(training["seed"]))
    base = torch.load(ROOT / protocol["initialization"]["checkpoint"], map_location="cpu", weights_only=True)
    model = initialize_context_maskgit_v5_from_v4(base, embeddings_payload["embeddings"], class_names).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(training["steps"]),
        eta_min=float(training["minimum_learning_rate"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    recovery_path = OUT / "recovery.pt"
    history_path = OUT / "history.jsonl"
    protocol_hash = sha256_file(PROTOCOL_PATH)
    start_step = best_step = 0
    best_ce = float("inf")
    best_accuracy = -1.0
    elapsed_before = 0.0
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=True)
        if recovery["protocol_sha256"] != protocol_hash:
            raise ValueError("Stage-22 recovery checkpoint belongs to another protocol")
        model.load_state_dict(recovery["state"])
        optimizer.load_state_dict(recovery["optimizer"])
        scheduler.load_state_dict(recovery["scheduler"])
        torch.random.set_rng_state(recovery["rng"])
        start_step = int(recovery["step"])
        best_step = int(recovery["best_step"])
        best_ce = float(recovery["best_ce"])
        best_accuracy = float(recovery["best_accuracy"])
        elapsed_before = float(recovery["elapsed_seconds"])
    read_history(history_path, start_step)
    atomic_json({
        "protocol_sha256": protocol_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "training_images": len(train["ids"]),
        "validation_images": len(validation_indices),
        "test_accessed": False,
    }, OUT / "config.json")
    started = time.monotonic()

    def checkpoint(step: int) -> dict:
        return {
            "model": "context_maskgit_v5",
            "config": model.config,
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.random.get_rng_state(),
            "step": step,
            "best_step": best_step,
            "best_ce": best_ce,
            "best_accuracy": best_accuracy,
            "elapsed_seconds": elapsed_before + time.monotonic() - started,
            "protocol_sha256": protocol_hash,
            "identity": train["identity"],
            "test_accessed": False,
        }

    validation_ids = dev["ids"][validation_indices].long()
    validation_labels = dev["labels"][validation_indices].long()
    per_class = int(protocol["validation"]["images_per_class"])
    chunk_indices = validation_indices.view(101, per_class)[:, : int(protocol["validation"]["chunk_images_per_class"])].flatten()
    chunk_validation_ids = dev["ids"][chunk_indices].long()
    chunk_validation_labels = dev["labels"][chunk_indices].long()
    model.train()
    with history_path.open("a", encoding="utf-8", buffering=1) as stream:
        for step in range(start_step + 1, int(training["steps"]) + 1):
            indices = torch.randint(len(train["ids"]), (int(training["batch_size"]),))
            ids = train["ids"][indices].long()
            labels = train["labels"][indices].long()
            per = float(training["per_choices"][int(torch.randint(len(training["per_choices"]), (1,)))])
            loss, missing, used_labels, parts = model.loss_components(ids, labels, int(training["packet_tokens"]), per)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Stage-22 loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "masked_ce": float(parts["masked_ce"].detach()),
                "context_prior_ce": float(parts["context_prior_ce"].detach()),
                "per": per,
                "masked_tokens": int(missing.sum()),
                "context_dropped": int((used_labels == model.no_context_id).sum()),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
            }
            if step % int(training["validation_every"]) == 0 or step == int(training["steps"]):
                seed_rows = [
                    {"seed": seed, "conditions": evaluate(model, validation_ids, validation_labels, protocol, int(seed))}
                    for seed in protocol["validation"]["seeds"]
                ]
                correct = [entry["conditions"][0] for entry in seed_rows]
                mean_ce = sum(entry["masked_ce"] for entry in correct) / len(correct)
                mean_accuracy = sum(entry["one_pass_accuracy"] for entry in correct) / len(correct)
                row["validation"] = {
                    "correct_context_mean_ce": mean_ce,
                    "correct_context_mean_accuracy": mean_accuracy,
                    "seeds": seed_rows,
                }
                if mean_ce < best_ce or (mean_ce == best_ce and mean_accuracy > best_accuracy):
                    best_step, best_ce, best_accuracy = step, mean_ce, mean_accuracy
                    atomic_save(checkpoint(step), OUT / "best.pt")
                print(json.dumps(row), flush=True)
                model.train()
            limit_reached = args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds and step < int(training["steps"])
            if limit_reached and "validation" not in row:
                seed = int(protocol["validation"]["chunk_seed"])
                row["chunk_boundary_validation"] = {
                    "seed": seed,
                    "images": len(chunk_validation_ids),
                    "conditions": evaluate(model, chunk_validation_ids, chunk_validation_labels, protocol, seed),
                }
                print(json.dumps(row), flush=True)
                model.train()
            stream.write(json.dumps(row) + "\n")
            if step % int(training["checkpoint_every"]) == 0 or step == int(training["steps"]):
                atomic_save(checkpoint(step), recovery_path)
                atomic_save(checkpoint(step), OUT / "last.pt")
            if limit_reached:
                atomic_save(checkpoint(step), recovery_path)
                atomic_save(checkpoint(step), OUT / "last.pt")
                print(json.dumps({"event": "TRAIN_CHUNK_COMPLETE", "step": step}), flush=True)
                raise SystemExit(75)

    selected = load_context_maskgit_v5(torch.load(OUT / "best.pt", map_location="cpu", weights_only=True)).eval()
    final_validation = [
        {"seed": seed, "conditions": evaluate(selected, validation_ids, validation_labels, protocol, int(seed))}
        for seed in protocol["validation"]["seeds"]
    ]
    atomic_json({
        "completed": True,
        "best_step": best_step,
        "best_correct_context_mean_ce": best_ce,
        "best_correct_context_mean_accuracy": best_accuracy,
        "final_validation": final_validation,
        "test_accessed": False,
    }, OUT / "results.json")
    atomic_json({"completed": True, "best_step": best_step, "test_accessed": False}, OUT / "status.json")


if __name__ == "__main__":
    main()
