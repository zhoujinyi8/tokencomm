"""Download and verify the public checkpoints linked by the TokCom authors.

The files remain local and are deliberately excluded from result archives.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import requests


REPO = "llvictorll/Maskgit-pytorch"
FILES = {
    "vqgan": ("pretrained_maskgit/VQGAN/last.ckpt", 957954257,
              "e38a3aed8c86e8d47436f611325b3a23888cab70a0cc56c13cd84e8d4aab6e71"),
    "maskgit256": ("pretrained_maskgit/MaskGIT/MaskGIT_ImageNet_256.pth", 2090306069,
                   "74b38fa16c92da3db79f9aac3c9d0d2ef5cb6ea283c48908213f10d786304462"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name, directory):
    remote, size, expected = FILES[name]
    target = directory / Path(remote).name
    partial = target.with_name(target.name + ".part")
    directory.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size == size and digest(target) == expected:
            print(json.dumps({"name": name, "status": "verified", "bytes": size}), flush=True)
            return
        raise ValueError(f"Existing target failed verification: {target}")
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Partial file exceeds expected size: {partial}")
        if offset == size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            url = f"https://huggingface.co/{REPO}/resolve/main/{remote}"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise ValueError("Server did not honor resume range")
                with partial.open("ab" if offset else "wb") as stream:
                    last_report = time.monotonic()
                    for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if block:
                            stream.write(block)
                        if time.monotonic() - last_report > 30:
                            print(json.dumps({"name": name, "bytes": stream.tell(), "expected": size}), flush=True)
                            last_report = time.monotonic()
        except (requests.RequestException, TimeoutError) as error:
            print(json.dumps({"name": name, "retry": attempt + 1, "error": str(error)}), flush=True)
            time.sleep(min(2 ** attempt, 30))
    if not partial.exists() or partial.stat().st_size != size:
        raise ValueError(f"Incomplete download: {partial}")
    actual = digest(partial)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {name}: {actual}")
    partial.replace(target)
    print(json.dumps({"name": name, "status": "downloaded_and_verified", "bytes": size,
                      "sha256": actual}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+", choices=FILES)
    parser.add_argument("--directory", type=Path, default=Path("pretrained/official_tokcom"))
    args = parser.parse_args()
    for selected in args.names:
        fetch(selected, args.directory)
