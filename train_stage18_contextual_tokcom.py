"""Resumable Stage-18 training for stronger class-conditioned TokCom."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from context_maskgit_v4 import initialize_context_maskgit_v4_from_v3, load_context_maskgit_v4
from train_stage15_contextual_tokcom import atomic_json, atomic_save, evaluate, read_history, sha256


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "STAGE18_STRONG_CMI_PROTOCOL.json"
DATA = ROOT / "data/imagenette_stage10/token_cache"
OUT = ROOT / "runs/stage18_strong_cmi"


def validate_inputs(protocol: dict) -> tuple[dict, dict]:
    for name, expected in protocol["data"]["hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen input hash mismatch: {name}")
    for name, expected in protocol["implementation_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen implementation changed: {name}")
    checkpoint_path = ROOT / protocol["initialization"]["checkpoint"]
    if sha256(checkpoint_path) != protocol["initialization"]["checkpoint_sha256"]:
        raise ValueError("Frozen V3 initialization checkpoint changed")
    stage17 = json.loads((ROOT / "STAGE17_AUDIT.json").read_text(encoding="utf-8"))
    if stage17.get("status") != "passed" or stage17.get("test_accessed") or stage17.get("stage14_holdout_used"):
        raise ValueError("Stage-17 completion or isolation audit failed")
    train = torch.load(DATA / "train.pt", weights_only=True)
    validation = torch.load(DATA / "val.pt", weights_only=True)
    if train["identity"] != validation["identity"] or train["identity"]["tokenizer_sha256"] != protocol["tokenizer_sha256"]:
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
    torch.manual_seed(int(training["seed"]))
    base = torch.load(ROOT / protocol["initialization"]["checkpoint"], weights_only=True)
    model = initialize_context_maskgit_v4_from_v3(base, float(protocol["model"]["context_prior_weight"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(training["steps"]), eta_min=float(training["minimum_learning_rate"])
    )
    OUT.mkdir(parents=True, exist_ok=True)
    recovery_path = OUT / "recovery.pt"
    history_path = OUT / "history.jsonl"
    protocol_hash = sha256(PROTOCOL_PATH)
    start_step = best_step = 0
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
            "model": "context_maskgit_v4",
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
            per = float(training["per_choices"][int(torch.randint(len(training["per_choices"]), (1,)))])
            loss, missing, used_labels, parts = model.loss_components(ids, labels, int(training["packet_tokens"]), per)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Stage-18 loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                    for seed in protocol["validation"]["seeds"]
                ]
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

    selected = load_context_maskgit_v4(torch.load(OUT / "best.pt", weights_only=True)).eval()
    sweeps = [
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
        for guidance in protocol["validation"]["guidance_sweep"]
        for seed in protocol["validation"]["seeds"]
    ]
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
        {"completed": True, "best_step": best_step, "test_accessed": False, "stage14_holdout_used": False},
        OUT / "status.json",
    )


if __name__ == "__main__":
    main()
