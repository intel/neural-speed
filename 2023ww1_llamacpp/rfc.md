Adding Native Support of SYCL for Intel GPUs

Hi the community, following the discussion [#3965](https://github.com/ggerganov/llama.cpp/discussions/3965), we are ready to contribute native SYCL backend to llama.cpp. Here is the official proposal.

## Motivation

Intel Arc series GPU provide accountable VRAM size and bandwidth, which the current OpenCL backend can't fully utilize especially on LLM. We shall expect huge performance improvement with native SYCL backend.

reference:
<br>[SYCL](https://www.khronos.org/sycl/)</br>
[Intel SYCL implementation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html)

## Proposal

### Native Kernels

We will implement the key operators of GGML in SYCL like [Metal](https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.metal) and [Vulkan](https://github.com/ggerganov/llama.cpp/pull/2059), the below is the steps

1. new backend, h2d & d2h
2. OneMKL-dpcpp based FP32 & FP16 GEMM
3. native SYCL kernels for de-quantization GEMM
3. native SYCL kernels for other operators



>note:
<br>Since llama.cpp is evolving quite fast and new feature will usually be updated under CUDA first, we will enable [SYCLomatic ](https://github.com/oneapi-src/SYCLomatic) to migrate CUDA code to SYCL native code smoothly.</br>
The tool ports both CUDA language kernels and library API calls. Typically, 90%-95% of CUDA code automatically migrates to SYCL1. Inline comments will also generated to help you finish writing and tuning your code.
<br>We will contribute verified migrated code to llama.cpp, no more effort from community. Since the tool is still at its early stage, welcome for any issues reporting to make it better.</br>
PS: llama.cpp itself has become the top1 target of SYCLomatic, so we can expect frequent update

Once the community has some plan for template-based library as discussed in [#3965](https://github.com/ggerganov/llama.cpp/discussions/3965), [XeTLA](https://github.com/intel/xetla) will be introduced at the next stage as the alternative BLAS library with de-quantization assembled at micro-kernel level.

## Summary

We will work on native kernels and enable SYCLomatic at the same time, feel free to drop a note.