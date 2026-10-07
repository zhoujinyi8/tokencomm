"""Resumable large-data Stage-13 ContextMaskGIT development training."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from context_maskgit_v2 import ContextMaskGITv2, load_context_maskgit_v2
from models import erase_packets
from run_pretrained import spatial_fill


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE13_TOKCOM_PROTOCOL.json"
DATA = ROOT / "data" / "imagenette_stage10" / "token_cache"
OUT = ROOT / "runs" / "stage13_tokcom_large"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def read_history(path: Path, completed_step: int) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    unique = {}
    for row in rows:
        if int(row["step"]) <= completed_step:
            unique[int(row["step"])] = row
    kept = [unique[index] for index in sorted(unique)]
    expected = list(range(1, completed_step + 1))
    if [int(row["step"]) for row in kept] != expected:
        raise ValueError("History does not exactly cover the recoverable checkpoint")
    path.write_text("".join(json.dumps(row) + "\n" for row in kept), encoding="utf-8")
    return kept


@torch.inference_mode()
def evaluate(model, data, per: float, seed: int, recovery_steps: int = 8, controls: bool = False, batch_size: int = 24):
    model.eval()
    ids = data["ids"]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        missing = erase_packets(ids, per, 4)
    donors = torch.tensor([int((data["labels"] == (int(label) + 1) % 10).nonzero()[0]) for label in data["labels"]])
    modes = ["correct", "wrong", "none"] if controls else ["correct"]
    rows = []
    for mode in modes:
        source = ids[donors] if mode == "wrong" else ids
        active = torch.ones_like(missing) if mode == "none" else missing
        total_loss = 0.0
        total_correct = 0
        total_missing = 0
        for start in range(0, len(ids), batch_size):
            target = ids[start : start + batch_size]
            mask = missing[start : start + batch_size]
            used = active[start : start + batch_size]
            received = source[start : start + batch_size].masked_fill(used, model.codes)
            logits = model(received)
            total_loss += float(F.cross_entropy(logits[mask], target[mask], reduction="sum"))
            recovered = model.recover(received, used, steps=recovery_steps)
            total_correct += int((recovered[mask] == target[mask]).sum())
            total_missing += int(mask.sum())
        rows.append({"mode": mode, "ce": total_loss / total_missing, "accuracy": total_correct / total_missing, "missing_tokens": total_missing})
    if controls:
        baseline = torch.from_numpy(spatial_fill(ids.masked_fill(missing, 0).numpy(), missing.numpy()))
        rows.append({"mode": "spatial", "accuracy": float((baseline[missing] == ids[missing]).float().mean()), "missing_tokens": int(missing.sum())})
    return rows


def main() -> None:
    torch.set_num_threads(4)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    for name, expected in (("manifest.json", protocol["data"]["manifest_sha256"]), ("train.jsonl", protocol["data"]["train_jsonl_sha256"]), ("val.jsonl", protocol["data"]["val_jsonl_sha256"])):
        if sha256(ROOT / "data" / "imagenette_stage10" / name) != expected:
            raise ValueError(f"Frozen data hash mismatch: {name}")
    train = torch.load(DATA / "train.pt", weights_only=True)
    val = torch.load(DATA / "val.pt", weights_only=True)
    if train["identity"] != val["identity"] or train["identity"]["tokenizer_sha256"] != protocol["tokenizer"]["sha256"]:
        raise ValueError("Token cache identity mismatch")
    if len(train["ids"]) != protocol["data"]["training_images"] or len(val["ids"]) != protocol["data"]["validation_images"]:
        raise ValueError("Frozen sample count mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    recovery_path = OUT / "recovery.pt"
    history_path = OUT / "history.jsonl"
    m = protocol["model"]
    model = ContextMaskGITv2(m["width"], m["transformer_layers"], m["local_blocks"])
    t = protocol["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t["steps"], eta_min=t["minimum_learning_rate"])
    start_step = 0
    best_step = 0
    best_ce = float("inf")
    best_accuracy = -1.0
    elapsed_before = 0.0
    torch.manual_seed(t["seed"])
    if recovery_path.exists():
        checkpoint = torch.load(recovery_path, weights_only=True)
        if checkpoint["protocol_sha256"] != sha256(PROTOCOL_PATH):
            raise ValueError("Recovery protocol mismatch")
        model.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        torch.random.set_rng_state(checkpoint["rng"])
        start_step = int(checkpoint["step"])
        best_step = int(checkpoint["best_step"])
        best_ce = float(checkpoint["best_ce"])
        best_accuracy = float(checkpoint["best_accuracy"])
        elapsed_before = float(checkpoint["elapsed_seconds"])
    read_history(history_path, start_step)
    config = {"protocol_sha256": sha256(PROTOCOL_PATH), "parameters": sum(x.numel() for x in model.parameters()), "test_accessed": False, "used_for_training": False}
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()

    def checkpoint(step: int):
        return {"model": "context_maskgit_v2", "config": model.config, "state": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": torch.random.get_rng_state(), "step": step, "best_step": best_step, "best_ce": best_ce, "best_accuracy": best_accuracy, "elapsed_seconds": elapsed_before + time.monotonic() - started, "protocol_sha256": config["protocol_sha256"], "identity": train["identity"]}

    with history_path.open("a", encoding="utf-8", buffering=1) as stream:
        for step in range(start_step + 1, t["steps"] + 1):
            indices = torch.randint(len(train["ids"]), (t["batch_size"],))
            ids = train["ids"][indices]
            per = float(t["per_choices"][int(torch.randint(len(t["per_choices"]), (1,)))])
            loss, missing = model.loss(ids, packet_tokens=t["packet_tokens"], per=per)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            row = {"step": step, "loss": float(loss.detach()), "per": per, "masked_tokens": int(missing.sum()), "lr": scheduler.get_last_lr()[0], "elapsed_seconds": elapsed_before + time.monotonic() - started}
            if step % t["validation_every"] == 0 or step == t["steps"]:
                scores = [evaluate(model, val, protocol["validation"]["per"], seed, protocol["validation"]["recovery_steps"])[0] for seed in protocol["validation"]["seeds"]]
                mean_ce = sum(x["ce"] for x in scores) / len(scores)
                mean_accuracy = sum(x["accuracy"] for x in scores) / len(scores)
                row["validation"] = {"mean_ce": mean_ce, "mean_accuracy": mean_accuracy, "seeds": scores}
                if mean_ce < best_ce or (mean_ce == best_ce and mean_accuracy > best_accuracy):
                    best_step, best_ce, best_accuracy = step, mean_ce, mean_accuracy
                    atomic_save(checkpoint(step), OUT / "best.pt")
                print(json.dumps(row), flush=True)
            stream.write(json.dumps(row) + "\n")
            if step % t["checkpoint_every"] == 0 or step == t["steps"]:
                atomic_save(checkpoint(step), recovery_path)
                atomic_save(checkpoint(step), OUT / "last.pt")

    best = torch.load(OUT / "best.pt", weights_only=True)
    selected = load_context_maskgit_v2(best)
    sweep = []
    for per in t["per_choices"]:
        for seed in protocol["validation"]["seeds"]:
            sweep.append({"per": per, "seed": seed, **evaluate(selected, val, per, seed, protocol["validation"]["recovery_steps"])[0]})
    controls = evaluate(selected, val, protocol["validation"]["per"], 13199, protocol["validation"]["recovery_steps"], controls=True)
    result = {"completed": True, "best_step": best_step, "best_mean_ce": best_ce, "best_mean_accuracy": best_accuracy, "sweep": sweep, "controls": controls, "test_accessed": False, "used_for_training": False}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (OUT / "status.json").write_text(json.dumps({"completed": True, "best_step": best_step, "test_accessed": False}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
