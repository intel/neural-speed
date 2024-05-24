# Highly Optimized Low Precision Kernels
Our kernels are based on x64 template library [BesTLA](../../bestla/README.md).
## Support Matrix
Limited by the graph framework, we only add kernels which accept float tensor as input and output tensor.

input dtype | output dtype | compute type | compute ISA
--- |---|---|---
float32 | float32 | float32 | AVX2
float32 | float32 | float32 | AVX512F
float32<sup>1</sup> | float32<sup>2</sup> | int8 | AVX512_VNNI
float32<sup>1</sup> | float32<sup>2</sup> | int8 | AVX512BW
float32<sup>1</sup> | float32<sup>2</sup> | int8 | AVX_VNNI
float32<sup>1</sup> | float32<sup>2</sup> | int8 | AMX_INT8
float32<sup>1</sup> | float32<sup>2</sup> | int8 | AVX2
float32/bf16 | float32/bf16 | bf16 | AMX_BF16
float32/fp16 | float32/fp16 | fp16 | AVX512_FP16

<sup>1</sup>: per-batch and per-K group-wise dynamic quantization for input tensor, where per-K group-wise also applies to weight quantization
group size of weight tensor; support both symmetric and asymmetric quantization.
<sup>2</sup>: per-batch dynamic dequantization for output tensor.

### Weight-only Quantization Support
dtype | algo | group size
--- | --- | ---
int4 | symmetric or asymmetric | multiplier of 8, -1<sup>1</sup>
int3 | symmetric or asymmetric | multiplier of 8, -1<sup>1</sup>
int2 | symmetric or asymmetric | multiplier of 8, -1<sup>1</sup>
int5 | symmetric or asymmetric | multiplier of 8, -1<sup>1</sup>
int6 | symmetric or asymmetric | multiplier of 8, -1<sup>1</sup>
int7 | symmetric or asymmetric<sup>2</sup> | multiplier of 8, -1<sup>1</sup>
int1 | symmetric or asymmetric | multiplier of 8, -1<sup>1</sup>
int8<sup>3</sup> | symmetric | multiplier of 8, -1<sup>1</sup>
fp4 | | multiplier of 8
nf4 | | multiplier of 8

<sup>1</sup>: group size=-1 means per channel quantization on output channel (or group size equals to input channel size).  
<sup>2</sup>: int7 + asymmetric may cause numeric overflow if the device only has AVX2 without AVX_VNNI or uses AVX512BW to compute.
<sup>3</sup>: It may cause numeric overflow if the device only has AVX2 without AVX_VNNI or uses AVX512BW to compute.

NOTE:
1. AMX_INT8 requires group size is aligend to 128 (best hardware efficiency)
2. int1, int2 and int3 have accuracy loss using RTN quantization.

### Hybrid quantization support
We can support the hybrid quantization combination. E.g. int4 x int2 mixed quantization.   
Each model can have a unique quantization configuration.  This configuration can tell the engine what quantization parameter will be applied to each weight. This allows layers can have different quantization
bits, algorithms and group sizes. Referring [llama int2&int4 mixed L252](../models/llama/llama_utils.cpp)

## Fusion Support
We support three kinds of kernel fusion for transformer models: QKV, MHA (multi-head attention), and FFN (feed-forward network) fusion.

<table>
    <thead>
        <tr>
            <th>fusion type</th>
            <th>models</th>
            <th>runtime ISA</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>QKV</td>
            <td >GPT-J<br>LLaMA</td>
            <td>AMX_INT8, AVX512_VNNI, AVX512BW, AVX512F, AMX_BF16, AVX_VNNI, AVX2</td>
        </tr>
        <tr>
            <td>FFN</td>
            <td>GPT-J<br>LLaMA<br>BLOOM<br>ChatGLM<br>Falcon<br>MPT</td>
            <td>AMX_INT8, AVX512_VNNI, AVX512BW, AVX512F, AMX_BF16, AVX_VNNI, AVX2</td>
        </tr>
        <tr>
            <td>MHA</td>
            <td colspan=2>

Referring [the fused-attention doc for details](../docs/fused_attention.md#supported-models)
</td>
        </tr>
    </tbody>
</table>

## Recommended Configuration for CPUs
codename | weight config | runtime ISA
---|---|---
Sapphire Rapids<br>Emerald Rapids | symmetric<br>int4<br>group size=128<br>compute type=int8 | AMX_INT8
Ice Lake<br>Cascade Lake<br>Cooper Lake<br>Tiger Lake<br>Rocket Lake | symmetric<br>int4<br>group size=128<br>compute type=int8 | AVX512_VNNI
Skylake<br>Cannon Lake |  symmetric<br>int4<br>group size=128<br>compute type=int8 | AVX512BW
Alder Lake (12th Gen)<br>Raptor Lake (13th and 14th Gen)| symmetric<br>int4<br>group size=128<br>compute type=int8 | AVX_VNNI
Older architecture (before 12th Gen)|  symmetric<br>int4<br>group size=128<br>compute type=int8 | AVX2

`sym int4 group=128 comp_dtype=int8` has almost the same accuracy as `group=32`, but is much faster (validated with LLaMa2-7B).  
`sym int5 group=-1 comp_dtype=int8` is the fastest configuration for the first-token with good accuracy (validated with LLaMa2-7B).  
`sym int3 group=128 comp_dtype=int8` is the fastest configuration for the next-token with good accuracy (validated with LLaMa2-7B).

NOTE:  
1. group_size=-1 has the smallest model size, and the best performance. But it requires the INC's finetuned model, or it may have lower accuracy than small group sizes.
2. group_size=128 is a balance of accuracy and speed if you want RTN quantization only.
3. group_size=32, scale_dtype=bf16, compute_dtype=int8, alg=sym equals llama.cpp's Q4_0.
