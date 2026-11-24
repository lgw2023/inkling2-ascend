# Historical Ascend V1 Model Runner snapshot

Upstream: https://github.com/vllm-project/vllm-ascend
Commit: f16e5e26801f86b071bca6454c9b9b2b63362e0a
Reviewed: 2026-11-24

This snapshot preserves the historical V1 runner and worker selection logic. It is a source-reading/backport reference, not a complete vLLM Ascend checkout and not a version-qualified runtime pairing with the current vLLM Inkling2 model. The runner source is Apache-2.0 and retains its original copyright/SPDX header. The current upstream main no longer contains this V1 path; adapt it to the selected V1 engine API rather than copying blindly.
