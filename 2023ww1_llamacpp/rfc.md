<h2> Adding Native Support of SYCL for Intel GPUs </h2>

Authors: [@airMeng](https://github.com/airMeng) [@hshen14](https://github.com/hshen14)

Hi the community, following the discussion [#3965](https://github.com/ggerganov/llama.cpp/discussions/3965), we plan to contribute native SYCL backend to llama.cpp. Here is the proposal:

## Motivation

Intel Arc series GPU provides accountable VRAM size and bandwidth, which the current OpenCL backend can't fully utilize especially on LLM. We shall expect huge performance improvement with native SYCL backend.

reference:
<br>[SYCL](https://www.khronos.org/sycl/)</br>
[Intel SYCL implementation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/data-parallel-c-plus-plus.html)

## Proposal

### Native Kernels

We will implement the key operators of GGML in SYCL like [Metal](https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.metal) and [Vulkan](https://github.com/ggerganov/llama.cpp/pull/2059). The steps are described as below:

1. new backend, h2d & d2h
2. OneMKL-dpcpp based FP32 & FP16 GEMM
3. native SYCL kernels for de-quantization GEMM
3. native SYCL kernels for other operators

>note:
<br>Since llama.cpp is evolving rapidly and new features will probably be supported through CUDA first, we plan to enable [SYCLomatic ](https://github.com/oneapi-src/SYCLomatic) to help migrate the code from CUDA to SYCL.</br>

We plan to introduce the template-based library e.g., [XeTLA](https://github.com/intel/xetla) as mentioned in [#3965](https://github.com/ggerganov/llama.cpp/discussions/3965) as the next stage, while we will be focusing on native SYCL support in this proposal.

## Summary

We will work on native SYCL kernels and enable in llama.cpp for Intel GPUs. Please feel free to drop a note. Thanks.
