# Test execution

Install `requirements-ci.txt`, then run `pytest -m "not integration"` for the clean-checkout unit suite used by GitHub Actions. The `integration` marker identifies the exact tests that require local experiment datasets, checkpoints, generated features, or frozen artifact identities. Their assertions are retained.

After restoring those assets and their original implementation identities, run `pytest -m integration` for the artifact suite, or `pytest` for both suites. An asset-dependent test is not considered passed by the unit job. The historical Stage19 implementation hash mismatch observed on a clean checkout must still be resolved or explained when restoring that frozen experiment; expected protocol hashes were not rewritten here.

The ContextMaskGITv4 gradient and state round-trip test uses synthetic inputs and runs in the unit job. Only its checkpoint warm-start test requires the local Stage15 checkpoint.

`scheduled_reproduction/` is a versioned experiment snapshot with separate data, weights, locks and execution instructions. Its historical scripts are excluded from root lint and test discovery. The snapshot verifier checks byte hashes and Python syntax without importing heavy models. Frozen holdout output audits and model evaluation are recorded separately in that folder's evidence and reports.

The separate `Verify reproduction protocol contracts` step imports eight lightweight check modules from that snapshot and calls their `verify()` functions. It checks eight stopping-rule/noise/accounting cases, six ARQ endpoint cases, six matched-neural cases, eight Cheng container cases, six byte-packet ledger cases, nine ImageNet100 manifest/file-identity cases, nine complete-grid rate-selection cases, and eight byte/variable-packet transport cases. These 60 checks supplement the root pytest count. They do not load weights, write experiment receipts, or run a channel/model. Synthetic manifest fixtures are not dataset evidence. Rate-window acceptance is development calibration and cannot establish paper dataset/checkpoint provenance.

The native rANS/CDF checks, actual Cheng coding, fresh-process decoding and gradient/rate pilots require local source builds, checkpoints and image assets; their separately audited receipts do not turn the 11 deselected asset integration tests into passing tests.
