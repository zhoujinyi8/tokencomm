"""Freeze the public 101-class scale-transfer data plan."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "STAGE20_FOOD101_SCALE_PLAN.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Stage-20 Food-101 plan already exists")
    stage18 = json.loads((ROOT / "STAGE18_AUDIT.json").read_text(encoding="utf-8"))
    if stage18.get("status") != "passed" or stage18.get("test_accessed") or stage18.get("stage14_holdout_used"):
        raise ValueError("Stage-18 isolation prerequisite failed")
    payload = {
        "stage": 20,
        "status": "frozen",
        "purpose": "101-class/7-bit reliable-CMI scale transfer using a directly downloadable public dataset",
        "fidelity": "approximate scale transfer; Food-101 is not the paper's ImageNet100",
        "source": {
            "official_page": "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/",
            "url": "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
            "content_length": 4996278331,
            "etag": "\"129cd283b-4fdc4f19aaaa0\"",
            "last_modified": "Wed, 09 Jul 2014 16:10:15 GMT",
            "classes": 101,
            "images": 101000,
            "official_train_per_class": 750,
            "official_test_per_class": 250,
        },
        "split": {
            "seed": 20261020,
            "dev_per_class": 50,
            "train_per_class": 700,
            "train_images": 70700,
            "dev_images": 5050,
            "sealed_official_test_images": 25250,
            "rule": "rank each official training filename by SHA256(seed:name); first 50 per class are development",
        },
        "side_information": {"classes": 101, "fixed_label_bits": 7, "reliable": True},
        "test_policy": "official test pixels remain compressed and unavailable to training/model selection until a later final-test protocol",
        "implementation_sha256": {"prepare_stage20_food101.py": sha256(ROOT / "prepare_stage20_food101.py")},
        "stage14_holdout_used": False,
        "test_accessed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256(OUTPUT), "train": 70700, "dev": 5050, "sealed_test": 25250}))


if __name__ == "__main__":
    main()
