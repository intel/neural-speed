# Proposal of sub-byte low precision inference for PyTorch/IPEX

**Authors:**
* @xinhe3, hengyume

## **Summary**
In this proposal, we mainly discuss the storage data type, compute data type, quantization, and serialization, to facilitate the low-bit (INT4/FP4) LLM inference for PyTorch/IPEX.

## **Motivation**
The community is active to support sub-byte low precision inference for LLMs, where the sub-byte may include INT4, FP4, NF4, INT2/3/5/6/7, MX formats, etc. The ecosystem is bringing up new idea of group-wise sub-byte support from the recent work like [llama.cpp](https://github.com/ggerganov/llama.cpp) and [OCP MX format](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai).

## **Storage data type**

### Option 1: reuse torch.uint8 
We follow the existing storage data type torch.uint8 to represent INT4/FP4 and there is no new storage required. In particular, uint8 is interchangeable with a pair of uint4 in PyTorch.

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
x_uint8 = torch.tensor([1, 7]).to(torch.uint8)
x_uint4 = x_uint8.to(torch.uint4x2)
# x_uint4: 00011111/b

scales = torch.rand(1)
zero_points = torch.rand(1)

woqlinear = nn.WOQLinear(group_size=2) # new op required
output = woqlinear(x_uint4, scales, zero_points) 
```

### Option 1.1: support group-wise 4-bit tensor

In addition to Option 1, we plan to introduce a native group-wise 4-bit tensor representation, which consists of data, scales, and zero points per group.

```python
class Bits4Tensor(torch.tensor): # Scale to NBitsTensor
    def __init__(self, shape:List[], group_size:int, data, scales, zero_points = torch.empty()):
        super().__init__(shape)
        assert(is_group_size_valid(shape, group_size)) # More checkers
        self.data = data
        self.scales = scales
        self.zero_points = zero_points

x = Bits4Tensor(...)
print(x.get_scales())
#print(x.get_zero_points())

woqlinear = nn.WOQLinear() # new op required
output = woqlinear(x)
```

Note that we may just keep Option 1, if low precision op in PyTorch is trending to have group_size, scales, zero points etc.

### Option 2: introduce new group-wise storage data type

We propose native group-wise storage data type. The minimal unit is a group tensor with raw data, group size, scales, zero_points (optional), is_padded (optional), padded_size (optional).

```cpp
template<typename SRC_T_, typename S_T_>
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
input_int4_g32 = torch.rand((128, 16)).to(torch.int4_g32)  # group size = 32
input_fp4_g32 = torch.rand((128, 16)).to(torch.fp4_g32) /# group size = 32

res_f32 = torch.add(input_f32, input_int4_g32)   # dequantize to f32 and add
res_bf16 = torch.add(input_bf16, input_fp4_g32)   # dequantize to bf16 and add
```

### **Scales & Zero-points**

Scales and zero points (optional) should be the same size as ```round_up(tensor_size/group_size)```.


### **Padding**

If tensor_size is not dividable by group_size, padding is required.

```python
x = torch.rand(65)
x_int4 = x.to(torch.int4_g32)
assert(x_int4.size == 65)
assert(x_int4.padded_size == 96)

... cpp side
static_assert(sizeof(x_int4_size) == 96))
```

## **Compute data type**

Sub-byte is mainly used by low precision inference for LLMs. The recommended practice is to dequantize the weight from sub-byte to floating point, so the compute data type will be BF16/FP16 on Xeon/GPU and FP16/BF16/FP8 on Gaudi/FS1.

Below is the sample code to implement the dequantization to floating points

```cpp

inline C10_HOST_DEVICE float* operator+(_SRC_T* a, _F_T* b) {
  return dequantize<_F_T*>(a->data, a->scale) + b;
}

inline C10_HOST_DEVICE float* operator*(_SRC_T* a, _F_T* b) {
  return dequantize<_F_T*>(a->data, a->scale) * b;
}
```

## **Tensor quantization/dequantization**

Sample code for Option 1:

```python
x, scales = torch.quantize(torch.randn(3,3))
y = torch.dequantize(x, scales)
```

Sample code for Option 1.1 and 2 (with better user experience):

```python
x = torch.randn(3,3).to(torch.uint4x2)
y = x.to(torch.float)
```

## **Quantization workflow/tools**

The quantization flow in PyTorch:

```python
dtype = torch.quint4x2
from torch.ao.quantization import default_float_qparams_observer_4bit
uint4_obs = default_float_qparams_observer_4bit()
uint4_obs(weights)
qparams = uint4_obs.calculate_qparams()
uint4_weight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=dtype)
```

The quantization flow in INC:
```python
conf = PostTrainingQuantConfig(
    approach="weight_only",
    op_type_dict={
        ".*": {
            "weight": {
                "bits": 4,  # 1-8 bit
                "group_size": -1,  # -1 (per-channel)
                "scheme": "sym",
                "algorithm": "RTN",
            },
        },
    },
)
int4_model = quantizer.fit(model, conf)
```

## **Serialization**

TBD (Hengyu & Xin)

## **Conclusion**

### Overview of Cons. and Pros.
|| Option 1| Option 1.1 | Option 2|
| --- | --- | --- | --- |
| Architecture| No | Newly-introduced tensor data type | Newly-introduced storage data type|
| Native UX| Medium | High | High |
| Performance Benefits | - | - | Potential |
| Engineering/Upstreaming Efforts | Low | Medium | High |

Based on the above table, Option 1 (and 1.1) is recommended in this proposal.

### **Evolution**

Option 1.1 is an improved version of Option 1, in which all APIs can be backward compatible. We are positioning Option 1.1 as the intermediate version for Option 2, while we'll try submitting RFC for Option 2 to PyTorch though we may face the potential upstreaming challenges and uncertainties of fast-growing sub-byte low precision inference.

## **Opens**

### **Shall we expose the storage data type for all sub-bytes?**

No, only limited sub-bytes e.g., INT4, FP4.

### **Shall we expose the group size in the storage data type?**

No need for Option 1. We are seeing the benefits for Option 1.1 and 2 in terms of performance and binary size (AOT build).

### **Other questions**
* Weight-only INT4 vs. model INT4?
* Standard data type representation (e.g., exp/mantissa) for sub-bytes (e.g., FP4)? Follow MX format?

## **Reference**
* [llama.cpp](https://github.com/ggerganov/llama.cpp): 2-6bit inference.
* [OCP MX format](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai) proposed by AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm, defines 4 group-wise data types: MXFP8, MXFP6, MXFP4, and MXINT8.
