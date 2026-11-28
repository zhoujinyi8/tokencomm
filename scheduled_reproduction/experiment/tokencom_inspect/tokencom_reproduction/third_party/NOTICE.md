# Vendored code provenance

`maskgit_vqgan.py` is copied without edits from:
https://github.com/xyfJASON/maskgit-pytorch/blob/ca5c7c39cf55d49fb034d7348595c9c3abc2c4ed/models/stage1/maskgit/maskgit_vqgan.py

Its header credits Google LLC, Hugging Face, and ByteDance modifications, under Apache-2.0. The full Apache license and upstream repository MIT license are included here. This is a community PyTorch conversion of the public MaskGIT tokenizer, not code released by either Token Communications paper's authors.

The wrapper `pretrained_vq.py` is independently written, uses the exact upstream configuration, and requires strict state-dict matching. It preserves upstream architecture details rather than silently modifying them.

Weights: https://huggingface.co/fun-research/TiTok/blob/ab646ed225080a3acb7c78440a574d7f67f16fa7/maskgit-vqgan-imagenet-f16-256.bin

SHA256: `424e7f2b11d1d63691ae3fbab74cf6c417ecbe0d8f86187b71f3538c7b56f6d2`

Downloaded using hf-mirror.com because the original host was unreachable locally, then matched against the SHA256 published by the original Hugging Face model page. Model repository license: Apache-2.0.

Public test image: https://github.com/pytorch/hub/blob/master/images/dog.jpg

Git blob ID at download: `12f0e0dd1162b94a5b0919ce8b91821450965985` (661378 bytes). Used only as a natural-image pipeline example, not a benchmark or training dataset. No new rights to the photograph are asserted.
