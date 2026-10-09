"""Resumable Stage-15 training for the context-conditioned TokCom receiver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from context_maskgit_v3 import (
    ContextMaskGITv3,
    initialize_context_maskgit_v3_from_v2,
    load_context_maskgit_v3,
)
from models import erase_packets


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE15_CONTEXTUAL_TOKCOM_PROTOCOL.json"
DATA = ROOT / "data/imagenette_stage10/token_cache"
OUT = ROOT / "runs/stage15_contextual_tokcom"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_history(path: Path, completed_step: int) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    unique = {int(row["step"]): row for row in rows if int(row["step"]) <= completed_step}
    kept = [unique[index] for index in sorted(unique)]
    if [int(row["step"]) for row in kept] != list(range(1, completed_step + 1)):
        raise ValueError("History does not exactly cover the recoverable checkpoint")
    path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    return kept


def build_model(protocol: dict) -> ContextMaskGITv3:
    torch.manual_seed(int(protocol["training"]["seed"]))
    base = torch.load(ROOT / protocol["initialization"]["checkpoint"], weights_only=True)
    return initialize_context_maskgit_v3_from_v2(
        base,
        classes=int(protocol["model"]["classes"]),
        context_dropout=float(protocol["model"]["context_dropout"]),
    )


@torch.inference_mode()
def evaluate(
    model: ContextMaskGITv3,
    data: dict,
    per: float,
    seed: int,
    recovery_steps: int = 8,
    guidance_scale: float = 1.0,
    batch_size: int = 24,
) -> list[dict]:
    model.eval()
    ids = data["ids"]
    labels = data["labels"].long()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        missing = erase_packets(ids, per, 4)
    conditions = [
        ("correct", labels),
        ("wrong", (labels + 1) % model.classes),
        ("none", None),
    ]
    rows = []
    for mode, used_labels in conditions:
        total_loss = 0.0
        total_correct = 0
        total_missing = 0
        for start in range(0, len(ids), batch_size):
            target = ids[start : start + batch_size]
            mask = missing[start : start + batch_size]
            batch_labels = None if used_labels is None else used_labels[start : start + batch_size]
            received = target.masked_fill(mask, model.codes)
            logits = model.guided_logits(received, batch_labels, guidance_scale)
            total_loss += float(F.cross_entropy(logits[mask], target[mask], reduction="sum"))
            recovered = model.recover(
                received,
                mask,
                batch_labels,
                steps=recovery_steps,
                guidance_scale=guidance_scale,
            )
            total_correct += int((recovered[mask] == target[mask]).sum())
            total_missing += int(mask.sum())
        rows.append(
            {
                "mode": mode,
                "ce": total_loss / total_missing,
                "accuracy": total_correct / total_missing,
                "missing_tokens": total_missing,
                "guidance_scale": guidance_scale,
            }
        )
    return rows


def validate_inputs(protocol: dict) -> tuple[dict, dict]:
    for name, expected in protocol["data"]["hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    checkpoint_path = ROOT / protocol["initialization"]["checkpoint"]
    if sha256(checkpoint_path) != protocol["initialization"]["checkpoint_sha256"]:
        raise ValueError("Frozen V2 initialization checkpoint changed")
    train = torch.load(DATA / "train.pt", weights_only=True)
    validation = torch.load(DATA / "val.pt", weights_only=True)
    if train["identity"] != validation["identity"]:
        raise ValueError("Training and validation tokenizers differ")
    if train["identity"]["tokenizer_sha256"] != protocol["tokenizer_sha256"]:
        raise ValueError("Frozen tokenizer identity mismatch")
    if len(train["ids"]) != protocol["data"]["training_images"] or len(validation["ids"]) != protocol["data"]["validation_images"]:
        raise ValueError("Frozen sample count mismatch")
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    train, validation = validate_inputs(protocol)
    training = protocol["training"]
    model = build_model(protocol)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(training["steps"]),
        eta_min=float(training["minimum_learning_rate"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    recovery_path = OUT / "recovery.pt"
    history_path = OUT / "history.jsonl"
    protocol_hash = sha256(PROTOCOL_PATH)
    start_step = 0
    best_step = 0
    best_ce = float("inf")
    best_accuracy = -1.0
    elapsed_before = 0.0
    if recovery_path.exists():
        recovery = torch.load(recovery_path, weights_only=True)
        if recovery["protocol_sha256"] != protocol_hash:
            raise ValueError("Recovery checkpoint belongs to another protocol")
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
    atomic_json(
        {
            "protocol_sha256": protocol_hash,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "initialization_checkpoint_sha256": protocol["initialization"]["checkpoint_sha256"],
            "test_accessed": False,
            "stage14_holdout_used": False,
        },
        OUT / "config.json",
    )
    started = time.monotonic()

    def checkpoint(step: int) -> dict:
        return {
            "model": "context_maskgit_v3",
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
        }

    model.train()
    with history_path.open("a", encoding="utf-8", buffering=1) as stream:
        for step in range(start_step + 1, int(training["steps"]) + 1):
            indices = torch.randint(len(train["ids"]), (int(training["batch_size"]),))
            ids = train["ids"][indices]
            labels = train["labels"][indices].long()
            per_index = int(torch.randint(len(training["per_choices"]), (1,)))
            per = float(training["per_choices"][per_index])
            loss, missing, used_labels = model.loss(ids, labels, int(training["packet_tokens"]), per)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite contextual training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "per": per,
                "masked_tokens": int(missing.sum()),
                "context_dropped": int((used_labels == model.no_context_id).sum()),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
            }
            if step % int(training["validation_every"]) == 0 or step == int(training["steps"]):
                seed_rows = []
                for seed in protocol["validation"]["seeds"]:
                    seed_rows.append(
                        {
                            "seed": seed,
                            "conditions": evaluate(
                                model,
                                validation,
                                float(protocol["validation"]["per"]),
                                int(seed),
                                int(protocol["validation"]["recovery_steps"]),
                                float(protocol["validation"]["guidance_scale"]),
                            ),
                        }
                    )
                correct = [entry["conditions"][0] for entry in seed_rows]
                mean_ce = sum(entry["ce"] for entry in correct) / len(correct)
                mean_accuracy = sum(entry["accuracy"] for entry in correct) / len(correct)
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
            stream.write(json.dumps(row) + "\n")
            if step % int(training["checkpoint_every"]) == 0 or step == int(training["steps"]):
                atomic_save(checkpoint(step), recovery_path)
                atomic_save(checkpoint(step), OUT / "last.pt")
            if args.max_seconds > 0 and time.monotonic() - started >= args.max_seconds and step < int(training["steps"]):
                atomic_save(checkpoint(step), recovery_path)
                atomic_save(checkpoint(step), OUT / "last.pt")
                print(json.dumps({"event": "TRAIN_CHUNK_COMPLETE", "step": step}), flush=True)
                raise SystemExit(75)

    selected = load_context_maskgit_v3(torch.load(OUT / "best.pt", weights_only=True)).eval()
    sweeps = []
    for guidance in protocol["validation"]["guidance_sweep"]:
        for seed in protocol["validation"]["seeds"]:
            sweeps.append(
                {
                    "guidance_scale": guidance,
                    "seed": seed,
                    "conditions": evaluate(
                        selected,
                        validation,
                        float(protocol["validation"]["per"]),
                        int(seed),
                        int(protocol["validation"]["recovery_steps"]),
                        float(guidance),
                    ),
                }
            )
    atomic_json(
        {
            "completed": True,
            "best_step": best_step,
            "best_correct_context_mean_ce": best_ce,
            "best_correct_context_mean_accuracy": best_accuracy,
            "guidance_sweep": sweeps,
            "test_accessed": False,
            "stage14_holdout_used": False,
        },
        OUT / "results.json",
    )
    atomic_json(
        {
            "completed": True,
            "best_step": best_step,
            "test_accessed": False,
            "stage14_holdout_used": False,
        },
        OUT / "status.json",
    )


if __name__ == "__main__":
    main()
