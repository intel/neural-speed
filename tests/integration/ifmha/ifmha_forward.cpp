#define USE_XETLA_XE_HPC 1
#if USE_XETLA_XE_HPC
#include "ifmha_forward.h"
// #include "../../mha.h"

using cgf_t = std::function<void(sycl::handler&)>;
using cgfs_t = std::vector<cgf_t>;

#ifdef _WIN32
#define XETLA_KERNEL_EXPORT __declspec(dllexport)
#define XETLA_KERNEL_IMPORT __declspec(dllimport)
#else // _WIN32
#if defined(__GNUC__)
#define XETLA_KERNEL_EXPORT __attribute__((__visibility__("default")))
#else // defined(__GNUC__)
#define XETLA_KERNEL_EXPORT
#endif // defined(__GNUC__)
#define XETLA_KERNEL_IMPORT XETLA_KERNEL_EXPORT
#endif // _WIN32

#ifdef BUILD_XETLA_KERNEL_LIB
#define XETLA_KERNEL_API XETLA_KERNEL_EXPORT
#else
#define XETLA_KERNEL_API XETLA_KERNEL_IMPORT
#endif

namespace gpu::xetla {
namespace fmha {

template <
    typename ifmha_policy,
    typename T,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsTraining,
    bool kIsBiasBroadcast>
class IfmhaForwardKernel;

template <typename ifmha_forward_op_t, typename T>
struct IfmhaForwardImplKernelFunctor {
  KERNEL_MAIN void operator()(sycl::nd_item<2> item) const {
    // init ifmha forward op and arguments
    ifmha_forward_op_t ifmha_fwd_op;
    typename ifmha_forward_op_t::arguments_t args(
        query,
        key0,
        key1,
        value0,
        value1,
        index,
        alibi,
        bias,
        dropout,
        dropout_prob,
        sm_scale,
        out,
        num_batches,
        beam,
        num_heads,
        head_size,
        kv_len0,
        kv_len1,
        alibi_padding,
        attn_mask_padding);

    // call the functor
    ifmha_fwd_op(item, args);
  }
  IfmhaForwardImplKernelFunctor(
      T* query_,
      T* key0_,
      T* key1_,
      T* value0_,
      T* value1_,
      int32_t* index_,
      T* alibi_,
      T* bias_,
      uint8_t* dropout_,
      float dropout_prob_,
      float sm_scale_,
      T* out_,
      uint32_t num_batches_,
      uint32_t beam_,
      uint32_t num_heads_,
      uint32_t head_size_,
      uint32_t kv_len0_,
      uint32_t kv_len1_,
      uint32_t alibi_padding_,
      uint32_t attn_mask_padding_)
      : query(query_),
        key0(key0_),
        key1(key1_),
        value0(value0_),
        value1(value1_),
        index(index_),
        alibi(alibi_),
        bias(bias_),
        dropout(dropout_),
        dropout_prob(dropout_prob_),
        sm_scale(sm_scale_),
        out(out_),
        num_batches(num_batches_),
        beam(beam_),
        num_heads(num_heads_),
        head_size(head_size_),
        kv_len0(kv_len0_),
        kv_len1(kv_len1_),
        alibi_padding(alibi_padding_),
        attn_mask_padding(attn_mask_padding_) {}

 private:
  T* query;
  T* key0;
  T* key1;
  T* value0;
  T* value1;
  int32_t* index;
  T* alibi;
  T* bias;
  uint8_t* dropout;
  float dropout_prob;
  float sm_scale;
  T* out;
  uint32_t num_batches;
  uint32_t beam;
  uint32_t num_heads;
  uint32_t head_size;
  uint32_t kv_len0;
  uint32_t kv_len1;
  uint32_t alibi_padding;
  uint32_t attn_mask_padding;
};

// The launcher of indexed flash mha forward kernel
template <
    typename ifmha_policy,
    typename T,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsTraining,
    bool kIsBiasBroadcast>
cgfs_t ifmha_forward_impl(
    T* query,
    T* key0,
    T* key1,
    T* value0,
    T* value1,
    int32_t* index,
    T* alibi,
    T* bias,
    uint8_t* dropout,
    float dropout_prob,
    float sm_scale,
    T* out,
    uint32_t num_batches,
    uint32_t beam,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t kv_len0,
    uint32_t kv_len1,
    uint32_t alibi_padding,
    uint32_t attn_mask_padding) {
#ifdef SDP_DBG
  printf(
      "B, Bm, N, F, T0, T1, H: %d, %d, %d, %d, %d, %d, %d, UseAlibi: %d, UseBias: %d, IsTraining: %d, IsBiasBc: %d, uAT %d, uPT %d, scale %f alibi @ 0x%llx\n",
      num_batches,
      beam,
      num_heads,
      1,
      kv_len0,
      kv_len1,
      head_size,
      kUseAlibi,
      kUseBias,
      kIsTraining,
      kIsBiasBroadcast,
      alibi_padding,
      attn_mask_padding,
      sm_scale,
      (unsigned long long)alibi);
#endif

  static constexpr gpu_arch arch_tag =
      gpu_arch::XeHpc; // Later to be modified for multi-arch
  // ifmha forward kernel
  using ifmha_forward_op_t = ifmha_forward_t<
      ifmha_policy,
      T,
      kUseAlibi,
      kUseBias,
      kIsTraining,
      kIsBiasBroadcast,
      arch_tag>;

  sycl::nd_range<2> NdRange =
      ifmha_forward_op_t::get_nd_range(num_batches, beam, num_heads);

  IfmhaForwardImplKernelFunctor<ifmha_forward_op_t, T> kfn(
      query,
      key0,
      key1,
      value0,
      value1,
      index,
      alibi,
      bias,
      dropout,
      dropout_prob,
      sm_scale,
      out,
      num_batches,
      beam,
      num_heads,
      head_size,
      kv_len0,
      kv_len1,
      alibi_padding,
      attn_mask_padding);

  return {[=](sycl::handler& cgh) {
    cgh.parallel_for<class IfmhaForwardKernel<
        ifmha_policy,
        T,
        kUseAlibi,
        kUseBias,
        kIsTraining,
        kIsBiasBroadcast>>(NdRange, kfn);
  }};
}

} // namespace fmha

#define CALL_IMPL_FUNC(P)   \
  fmha::ifmha_forward_impl< \
      P,                    \
      T,                    \
      kUseAlibi,            \
      kUseBias,             \
      kIsTraining,          \
      kIsBiasBroadcast>(    \
      query,                \
      key0,                 \
      key1,                 \
      value0,               \
      value1,               \
      index,                \
      alibi,                \
      bias,                 \
      dropout,              \
      dropout_prob,         \
      sm_scale,             \
      out,                  \
      num_batches,          \
      beam,                 \
      num_heads,            \
      head_size,            \
      kv_len0,              \
      kv_len1,              \
      alibi_padding,        \
      attn_mask_padding)

/// @brief Main execution function for indexed flash mha forward.
template <
    typename T,
    bool kUseAlibi,
    bool kUseBias,
    bool kIsTraining,
    bool kIsBiasBroadcast>
cgfs_t ifmha_forward(
    T* query,
    T* key0,
    T* key1,
    T* value0,
    T* value1,
    int32_t* index,
    T* alibi,
    T* bias,
    uint8_t* dropout,
    float dropout_prob,
    float sm_scale,
    T* out,
    uint32_t num_batches,
    uint32_t beam,
    uint32_t num_heads,
    uint32_t head_size,
    uint32_t kv_len0,
    uint32_t kv_len1,
    uint32_t alibi_padding,
    uint32_t attn_mask_padding) {
  // occupancy first
  constexpr int hardware_concurrent_wg = 64;
  if (head_size <= 64) {
    return CALL_IMPL_FUNC(ifmha_policy_64x64);
  } else if (head_size <= 128) {
    return CALL_IMPL_FUNC(ifmha_policy_128x64);
  } else if (head_size <= 256) {
    return CALL_IMPL_FUNC(ifmha_policy_256x64);
  } else {
    assert(("SDP Index fusion kernel requires head_dim <= 256 ...", 0));
    return {};
  }
}

#undef CALL_IMPL_FUNC

XETLA_KERNEL_API cgfs_t fmha_forward_index_kernel(
    void* query,
    void* key,
    void* value,
    void* key_cache,
    void* value_cache,
    int32_t* index,
    void* alibi,
    void* attn_mask,
    uint8_t* dropout,
    void* out,
    uint32_t timestep,
    float alpha,
    float beta,
    float dropout_p,
    uint32_t num_batches,
    uint32_t beam_width,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t num_queries,
    uint32_t num_keys_in,
    uint32_t num_keys_out,
    uint32_t alibi_padding,
    uint32_t attn_mask_padding,
    bool is_causal,
    bool is_bias_broadcast) {
  using T = sycl::half;
  assert(
      ("SDP Index fusion kernel requires num_queries == 1 so far ...",
       num_queries == 1));
  assert(
      ("SDP Index fusion kernel doesn't support causal so far ...",
       is_causal == false));

#define DISPATCH_TEMPLATE(T, USE_ALIBI, USE_BIAS, IS_TRAINING, IS_BROADCAST) \
  return ifmha_forward<T, USE_ALIBI, USE_BIAS, IS_TRAINING, IS_BROADCAST>(   \
      (T*)query,                                                             \
      (T*)key,                                                               \
      (T*)key_cache,                                                         \
      (T*)value,                                                             \
      (T*)value_cache,                                                       \
      index,                                                                 \
      (T*)alibi,                                                             \
      (T*)attn_mask,                                                         \
      dropout,                                                               \
      dropout_p,                                                             \
      alpha,                                                                 \
      (T*)out,                                                               \
      num_batches,                                                           \
      beam_width,                                                            \
      num_heads,                                                             \
      head_dim,                                                              \
      num_keys_in,                                                           \
      num_keys_out,                                                          \
      alibi_padding,                                                         \
      attn_mask_padding);

  if (alibi) {
    if (attn_mask) {
      if (is_bias_broadcast) {
        DISPATCH_TEMPLATE(T, true, true, false, true)
      } else {
        DISPATCH_TEMPLATE(T, true, true, false, false)
      }
    } else {
      if (is_bias_broadcast) {
        DISPATCH_TEMPLATE(T, true, false, false, true)
      } else {
        DISPATCH_TEMPLATE(T, true, false, false, false)
      }
    }
  } else {
    if (attn_mask) {
      if (is_bias_broadcast) {
        DISPATCH_TEMPLATE(T, false, true, false, true)
      } else {
        DISPATCH_TEMPLATE(T, false, true, false, false)
      }
    } else {
      if (is_bias_broadcast) {
        DISPATCH_TEMPLATE(T, false, false, false, true)
      } else {
        DISPATCH_TEMPLATE(T, false, false, false, false)
      }
    }
  }
}
} // namespace gpu::xetla
#endif
