# Proposal of block-aware sub-byte dtype introduction to PyTorch

**Authors:**
* @xinhe3, hengyume


## **Summary**
The community are working on Deep Learning acceleration with sub-byte support. Considering alignment, elements are organized as blocks, and each block share a scale (and maybe a zero point). Some great examples are like 

* [llama.cpp](https://github.com/ggerganov/llama.cpp) supports 2-6bits.
* [MX format](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai) proposed by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm, consists of 4 block-aware datatypes: MXFP8, MXFP6, MXFP4, and MXINT8

This RFC proposes adding sub-byte data types variants to PyTorch.


## **Motivation**
TODO: lower precision, always organized as blocks


## **Storage for lower precision**

### Option 1: 4 bits based on 8 bits storage
If we only considerd int4/fp4/nf4, we can follow pytorch int4 implementation already and no newer storage needed. Actually Pytorch refer to uint8 to store 1 pair of int4 data.

```cpp
struct bit4x2 {
  int8_t x : 4;
  int8_t y : 4;
  bit4x2(int8_t v) : x(v), y(v) {}
  bit4x2() : x(0), y(0) {}
};

struct int4x2 : bit4x2 {
    int4x2(int8_t v) : bit4x2(v) {}
    int4x2() : bit4x2() {}
    static int8_t convert(int8_t src) {
        int32_t dst = src;
        dst = dst >= 0 ? dst + 8 : dst - 8;
        dst = dst / 16;
        dst = dst > 7 ? 7 : dst;
        dst = dst < -8 ? -8 : dst;
        return static_cast<int8_t>(dst);
    }
}
```

```python
x_uint8 = torch.tensor([1, 9]).to(torch.uint8)
x_uint4 = x_uint8.to(torch.uint4x2)
# x_uint4: 00010001/b

scales = torch.rand(1)
zero_points = torch.rand(1)

woqlinear = nn.WOQLinear(group_size=...)
output = woqlinear(x_uint4, scales, zero_points)
```

### Option 1.1: 8 bits storage with unified tensor interface

Odd bits need to be packed into block for higher efficiency load/store. For example, MX format is one of the most promising data format, which also needs block-wise parameters, like block size, scales, zero_points. 
```python
class Bits4Tensor(torch.tensor):
    def __init__(self, block, data, scale, zero_points = torch.empty()):
        assert(block in BLOCK_SIZE_SUPPORTED)
        self.data = data
        self.scale = scale
        self.zero_points = zero_points

x = Bits4Tensor(...)
print(x.get_scales())
#print(x.get_zero_points())
output = WOQLinear(X)
```

### Option 2: block-aware storage for the future

Further we propose block aware storage which is a comprehensive data format, organized in block not elements.

* the basic part is a block, including raw data, block size, scales and zero_points(optional)
* there will be a default padding, so there will be an optional "real size"
* other parts will follow MX paper

```cpp
template<typename SRC_T_, typename S_T_, typename DST_T_ = float>
class StorageWeight: {
  OptionalBuffer<S_T_> scales;
  OptionalBuffer<SRC_T_> zero_points;
  Buffer<SRC_T_> raw_data;

  DST_T_ dequantize(){...}
  StorageWeight(DST_T_ inputs){
    ... = quantize(inputs);
  }
}
```

```python
input_fp32 = torch.rand((128, 16))
input_bf16 = torch.rand((128, 16), dtype=torch.bf16)
input_e4b32 = torch.rand((128, 16)).to(torch.mx.e4_b32)  // quantized to int4, block size = 32
input_e2m1b32 = torch.rand((128, 16)).to(torch.mx.e2m1_b32) // quantized to e2m1, block size = 32

res_f32 = torch.add(input_f32, input_e4b32)   # dequantize to f32 and add
res_bf16 = torch.add(input_bf16, input_e2m1b32)   # dequantize to bf16 and add
```

### **Scales**

For all options, scales and zero points should be the same size as ```upper_bound(size/block_size)```


### **Padding**

mostly the blocks are distributed alone K dimensions. when K is not devisible by block size, there will be padings, which is almost the same with other padding.

```python
x = torch.rand(65)
x_e4 = x.to(torch.mx.e4_b32)
assert(x_e4.size == 65) // real size
# or assert(x_e4.size == 96), showing the padded size?

... cpp side
static_assert(sizeof(x_e4_buf) == 96)
```

### **Auto Quantization and Dequantization**

for option 1, there will be no auto quantization and dequantization and scales (+ zero points) is necessary

```python
x, scales = torch.quantize(torch.randn(3,3))
y = torch.dequantize(x, scales)
```

for option 1.1 and 2, there will be auto quantization and dequantization in the explicit data type conversion and scales (+ zero points) is not needed.

```python
x = torch.randn(3,3).to(torch.uint4x2)
y = x.to(torch.float)
```

While option 2 is built-in functions and can be fully optimized.

### **Computation support**

sub-byte support are mostly for WOQ LLM acceleration. All math operators are done with default float numbers of the device: fp16 for GPU, bf16 for Xeon.

It??s enough to have basic math operations implemented via casting to float:

```cpp
inline C10_HOST_DEVICE float* operator+(MX* a, float* b) {
  return dequantize<float*>(a->data, a->scale) + b;
}

inline C10_HOST_DEVICE float* operator*(MX* a, float* b) {
  return dequantize<float*>(a->data, a->scale) * b;
}
```


## Quantization Workflows

quantization flow in stock pytorch
```python
dtype = torch.quint4x2
from torch.ao.quantization import default_float_qparams_observer_4bit
uint4_obs = default_float_qparams_observer_4bit()
uint4_obs(weights)
qparams = uint4_obs.calculate_qparams()
uint4_weight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=dtype)
```

quantization flow in INC and other ecosystem tools
```python
conf = PostTrainingQuantConfig(
    approach="weight_only",
    op_type_dict={
        ".*": {  # re.match
            "weight": {
                "bits": 4,  # 1-8 bit
                "group_size": -1,  # -1 (per-channel)
                "scheme": "sym",
                "algorithm": "RTN",
            },
        },
    },
)
q_model = quantization.fit(model, conf, eval_func=eval_func)
# INT4 weight is compressed into torch.int32 tensor
compressed_model = q_model.export_compressed_model()
```


## Serialization
```python
torch.save(uint4_weight, 'tmp.pt')
```
or
```python
torch.save(compressed_model.state_dict(), 'tmp.pt')
```


## Conslusion

### Overview of Cons. and Pros.
|| Option 1| Option 1.1 | Option 2|
| --- | --- | --- | --- |
|archtecture change| zero| few | new base storage introduced|
|user friendly| poor, need tools' help | good| good|
| performance | no change | no change | good |
| development effort | low | medium, most tool side | high |

### Evolution

Option 1.1 is a slight improvements to Option 1, in which all API can be forward compatible.

Option 1.1 and Option 2 might play different role in different stages. After being widely verified, some data types can be re-implemented as Option 2 for further performance optimization.



## **Open points**
### **How should data types be exposed?**

For 4 bits, there in fact are only 2 kinds of data format: int4 and e2m1, we can easily expose to end users as uint4.
For other sub-byte data types including 2/3/5/6/7 bits, there will be a lot of combination.
Option 1: no alignment for other sub-bytes
Option 2 & 3: be able to cover.

### **Block size**

Although it seems that block size could be any for option 1, block size will impact the backend kernel generation and finally the performance.
For GPU with AOT support, various block size will make the binary redundant, we will only support limited block size which means block size is not configurable.


There are few questions regarding details of this solution in the context of being an alternative for true dtype.

* There are no difference between activation tensors and weight tensors, but lower precision only applies to weight?
* What are the limitations comparing to native built-in type?
* Does it have properties of floating-point format like infs/nans, underflow numbers, rounding modes?
* Is it configurable in terms of size of exponent/mantissa, bias, special values encoding?
* Can it be included in type promotion matrix?
