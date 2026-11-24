# Model-specific source snapshots

Reviewed 2026-11-24. Selected source files are copied from the pinned upstream commits in `references/source-lock.json`; original upstream paths are preserved beneath each project directory. These snapshots are for browsing, diffing and migration analysis, not standalone buildable checkouts. Full source trees are available through the corresponding submodules.

## TokenSpeed

MIT, Copyright LightSeek Foundation. Snapshot includes the Inkling runtime, top-k integration, GFX950 monokernel stages (attention, convolution, residual mix, experts, routing, quantization, head and layer orchestration), plan/dimension helpers and correctness tests. GFX950 is AMD hardware, not Ascend. Treat this as algorithm/fusion/layout reference only; port and revalidate all kernels for CANN/Triton Ascend.

## Unsloth

The repository core is Apache-2.0. The copied `unsloth/models/inkling2.py` and `inkling2_flex.py` carry file-level SPDX `AGPL-3.0-only`; the full AGPL text is included at `unsloth/studio/LICENSE.AGPL-3.0`. The loader file and root LICENSE are Apache-2.0. Preserve file-level licenses and do not assume one repository-wide license applies to every file.

## NVIDIA Dynamo

Apache-2.0. Snapshot includes Inkling 2 Mini SGLang deployment recipes for B200/VR200 aggregate and disaggregated deployments, model download/cache manifests and the model recipe page. These are NVIDIA/Kubernetes deployment references, not Ascend-ready manifests.

## Hugging Face blog

The two Inkling articles are linked in `references/articles/README.md`; the full blog text is not copied. The blog repository is large and the article text is not treated as source code to redistribute.
