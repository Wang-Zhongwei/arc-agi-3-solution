# gsq-coder-gguf-runner: research:engine-build

Workflow run `wf_0ae4547f-80a`, phase Research. Serve the ISTA GSQ-RCO Coder GGUF the way its Hugging Face repo documents (llama.cpp or Strata, both shards, 256 experts, mmproj), validate the harness contract locally, build staging/engine/test notebooks and the 2-hour final-run notebook

---

I've finished the build recipe for llama-server b11476 on sm_120a. I tested it as far as I could without x86 hardware or a GPU, using arm64 Ubuntu containers on the laptop. Nothing was pushed, uploaded or committed, and the user's working tree was not touched.

**The catch: Kaggle's current images are Ubuntu 24.04.** As of 2026-10-01, both the default CPU and GPU images have glibc 2.39, gcc 13 and Python 3.13; I read this from the gcr registry configs. Our gsq-stage CPU notebooks ran Python 3.13, so they were on that image.
- Our F001 GPU notebooks instead pin `"docker_image": "gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"`. That image has glibc 2.35, Python 3.12.13 built with GCC 11.4, and g++-11.
- The GPU is an RTX PRO 6000 Blackwell Server Edition on driver 580.159.04, which supports CUDA 13.0 natively.
- A binary built on the default CPU image needs GLIBC_2.38 and GLIBCXX_3.4.32, so it won't load in the pinned GPU image. That's the same problem the official release binaries have.
- So the build notebook pins the same `docker_image`. The script exits with code 3 within seconds if the host's glibc is newer than 2.35.

**The recipe** is `build_llama_server.sh`, run in a CPU notebook with internet and no competition attached:
- **Toolchain:** `pip install --target` of the same CUDA wheels the GPU notebook already mounts from pennyroyal-v253: nvcc/crt/nvvm 13.0.88, cuda-runtime 13.0.96, cccl 13.0.85 and cublas 13.1.1.3. Also cmake 4.4.3 and ninja 1.13.2.
- **Driver stub:** the wheels have no `libcuda.so` to link against. The script takes it from NVIDIA's `cuda-driver-dev-13-0_13.0.96-1_amd64.deb`, checked by sha256.
- **Configure:** the official release flags plus:
  - `-DCMAKE_CUDA_ARCHITECTURES=120a-real -DGGML_CUDA_CCCL_VERSION=v3.4.3`
  - `RUNPATH $ORIGIN`
  - `GGML_NATIVE=OFF` with AVX2, so the build machine's CPU features don't leak into the binary
  - `-Wl,-rpath-link,<wheel lib>`. This is required: without it, `llama-batched-bench` failed to link in my first test.
- **CCCL check:** a probe confirms the fetched CCCL 3.4.3 wins over the wheel's 3.0.1. Without it, the fast top-k path the sparse-attention indexer uses would silently drop out.
- **Package:** the build output plus `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13` and `libgomp.so.1`, behind an entry script `llama-server.sh`. Expect a tarball of about 0.5 GB plus a sha256 manifest.
- **Built-in checks:** glibc ≤ 2.35, `$ORIGIN` on every library, the CUB TopK kernels present, `--version` working with the stub driver, and a CPU smoke test that serves a 1.2 MB GGUF.

**What the tests showed:**
- A fresh build in Ubuntu 22.04 took 371 s with 4 jobs and passed every check.
- The tarball served completions in a clean 22.04 container with no CUDA and no gcc.
- That test caught that `libgomp.so.1` must be shipped, so the script now bundles it.
- The generated notebook cell ran end to end and left only the tarball, manifest and log as output.
- The fallback built on 24.04 also served correctly on 22.04 (details below).

**Expected cost on Kaggle:** about 15–30 minutes on a 4-vCPU CPU notebook. That's an estimate scaled from my local job time, not a measurement. Peak RAM was under 2 GB.

**GPU side:** add the build notebook as a `kernel_sources` input, then paste `gpu_install_llama.py` at the top of the serving cell. It checks the sha256, extracts to `/tmp/lcpp`, runs `--version`, and stops unless `--list-devices` shows `CUDA0`. Then launch `/tmp/lcpp/llama-b11476-cu130-sm120a/llama-server.sh`, which puts the bundled CUDA libraries first on the library path.

**Fallbacks:**
- **If Kaggle won't run the pinned image in a CPU session:** run the same notebook on an RTX Pro 6000 with the competition detached and internet on. Whether Kaggle allows internet there is unverified. It uses about 10 minutes of GPU quota, but you can also check the GPU and rehearse serving there.
- **If only the latest image is available:** `make_build_notebook.py --image latest --bundle-glibc` ships the build machine's glibc and libstdc++. The entry script then starts the server through that bundled loader. This worked on 22.04 in my test, but not with the real 580 driver loaded.

**Strata does need a build:**
- **No Linux binaries:** every release ships Windows zips only.
- **Two heavy CUDA builds:** the engine and a vision encoder, each pinned to llama.cpp commit `3cf03257`, not b11476. Both must use `-DSTRATA_PORTABLE=ON`, because its setup script compiles with `-march=native`.
- **Same toolchain and glibc rules:** CUDA 13.0 nvcc (13.2.0/13.2.1 miscompile sm_120) and the same runtime libraries, plus its Python server.
- **Model packing:** the GGUF has to be converted with `tools/iq_pack.py`, and an MTP layer built from the base Qwen model.
- **8-slot cap:** at most 8 concurrent requests, and our harness runs 10.
- The competitor's Strata dataset is for the unpruned Q2_0, not the Coder. llama.cpp is the simpler path.

**Not verified:**
- Whether Kaggle runs the pinned image in a CPU session.
- The x86_64 compile itself. Rosetta is off in OrbStack, so the test builds were arm64.
- Anything running on an sm_120 GPU.

For the local tests I started OrbStack, which wasn't running before. I've removed the test containers, deleted the test tarballs and stopped it again.

Files are in `/private/tmp/claude-501/-Users-advaith-Documents-studying-stuff-experiments-arc-agi-v4/beaafd04-a725-4273-a762-c3d1b876eb95/scratchpad/gguf-plan/research/`:
- `llamacpp-build-recipe.md`
- `build_llama_server.sh`
- `make_build_notebook.py`
- `push-llama-build/llama-b11476-sm120a-build.ipynb`
- `push-llama-build/kernel-metadata.json`
- `gpu_install_llama.py`
- `docker-out/` (build-arm64-r3.log, build-noble-bundle.log, build-noble-noflag.log, notebook-cell-run.log)
- `kaggle-images_python-config.json`
- `kaggle-gpu-images_python-config.json`
- `strata-src/`
- `strata-ds/`
