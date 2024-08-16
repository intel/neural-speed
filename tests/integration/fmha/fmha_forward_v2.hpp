
#pragma once
/*
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/
#include <sys/types.h>
#include <limits>

namespace gpu::xetla {
namespace fmha {
template <
    typename scaler_t,
    gpu_arch arch_tag,
    int sg_num,
    int head_dim,
    bool kUseBias>
class fmha_forward_v2_t {
 public:
  using accum_t = float;

  struct arguments_t {
    const scaler_t* query;
    const scaler_t* key;
    const scaler_t* value;
    const scaler_t* mask;
    scaler_t* output;

    const size_t num_batch;
    const size_t num_heads;
    const size_t num_kv_heads;
    const size_t ctx_len;

    const size_t q_batch_step;
    const size_t q_head_step;
    const size_t k_batch_step;
    const size_t k_head_step;
    const size_t k_seq_step;
    const size_t v_batch_step;
    const size_t v_head_step;
    const size_t v_seq_step;
    const size_t mask_batch_step;
    const size_t mask_head_step;
    const size_t out_batch_step;
    const size_t out_head_step;
  };
  static constexpr size_t LOAD_BYTES_LEN = 128;
  static constexpr size_t O_offset = 0;
  static constexpr size_t O_size = sg_num * head_dim * sizeof(accum_t);
  static constexpr size_t L_offset = O_size;
  static constexpr size_t L_size = sg_num * sizeof(accum_t);
  static constexpr size_t M_offset = L_offset + L_size;
  static constexpr size_t M_size = sg_num * sizeof(accum_t);

  /// @brief Gets local memory size consumption.
  /// Users query and get a local memory consumption size in compile time.
  /// @return The size of local memory required.
  inline static constexpr uint32_t get_slm_size() {
    const auto size = O_size + L_size + M_size;
    static_assert(
        size <= (arch_attr_t<arch_tag>::local_mem_size),
        "The local memory size should be less than arch total local memory size");
    return size;
  };

  inline static void check_slm_size(const sycl::device& d) {
    constexpr auto slm_size = get_slm_size();
    if (slm_size > d.get_info<sycl::info::device::local_mem_size>())
      throw std::runtime_error(
          "Head SLM size too large for the current device!");
  }

  /// @brief Helper function to get the nd_range under the Fmha policy.
  /// @return Expected nd_range.
  static sycl::nd_range<3> get_nd_range(
      uint32_t num_batch,
      uint32_t num_heads) {
    sycl::range<3> local_range = sycl::range<3>{1, 1, sg_num};
    sycl::range<3> group_range{num_batch, num_heads, 1};
    return sycl::nd_range<3>{group_range * local_range, local_range};
  };

  template <typename T>
  static inline void dump_mat_reg(T mat, size_t tile_x, size_t tile_y) {
#pragma unroll
    for (size_t row = 0; row < tile_y; row++) {
#pragma unroll
      for (size_t col = 0; col < tile_x; col++) {
        sycl::ext::oneapi::experimental::printf(
            "%f ", (float)(typename T::element_type)mat[row * tile_x + col]);
      }
      sycl::ext::oneapi::experimental::printf("\n");
    }
    sycl::ext::oneapi::experimental::printf("\n");
  }

  inline KERNEL_FUNC void operator()(nd_item<3>& item, arguments_t& args) {
    const size_t kv_group_num = args.num_heads / args.num_kv_heads;
    const size_t sg_ctx_len = args.ctx_len / sg_num;
    const size_t rem_ctx_len = args.ctx_len % sg_num;
    const size_t sg_head_dim = head_dim / sg_num;
    static_assert(head_dim % sg_num == 0);

    __ESIMD_NS::slm_init<get_slm_size()>();

    const size_t batch_idx = item.get_group(0);
    const size_t head_idx = item.get_group(1);
    const size_t kv_head_idx = head_idx / kv_group_num;
    const size_t sg_id = item.get_local_id(2);
    const scaler_t attn_scale = 1.f / ::std::sqrt(float(head_dim));

    const auto query_head = args.query + batch_idx * args.q_batch_step +
        head_idx * args.q_head_step;
    const auto key_head = args.key + batch_idx * args.k_batch_step +
        kv_head_idx * args.k_head_step;
    const auto value_head = args.value + batch_idx * args.v_batch_step +
        kv_head_idx * args.v_head_step;
    const auto mask_head = args.mask + batch_idx * args.mask_batch_step +
        head_idx * args.mask_head_step;
    const auto output_head = args.output + batch_idx * args.out_batch_step +
        head_idx * args.out_head_step;
    // if (item.get_global_linear_id() == 0) {
    //   sycl::ext::oneapi::experimental::printf("\n\n!!!!!!!\n");
    // }

    __ESIMD_NS::simd<fp16, head_dim> query_row;
    constexpr int BLK = LOAD_BYTES_LEN / sizeof(scaler_t);
    static_assert(head_dim % BLK == 0);
#pragma unroll
    for (int i = 0; i < head_dim; i += BLK) {
      query_row.template select<BLK, 1>(i) =
          __ESIMD_NS::block_load<scaler_t, BLK>(query_head + i);
    }

    size_t start_ctx_id = sg_ctx_len * sg_id + std::min(sg_id, rem_ctx_len);
    size_t end_ctx_id =
        start_ctx_id + sg_ctx_len + (sg_id < rem_ctx_len ? 1 : 0);

    // M_i = rowmax(S_i)
    accum_t M = std::numeric_limits<accum_t>::lowest();
    // L_i = rowsum(exp(S_i) - M_i)
    accum_t L = std::numeric_limits<accum_t>::min();
    __ESIMD_NS::simd<accum_t, head_dim> O = 0;
    __ESIMD_NS::simd<scaler_t, head_dim> kv_row;
    for (size_t i = start_ctx_id; i < end_ctx_id; ++i) {
      kv_row = __ESIMD_NS::block_load<scaler_t, head_dim>(
          key_head + i * args.k_seq_step);
      accum_t attn =
          __ESIMD_NS::detail::sum<accum_t, fp16, head_dim>(query_row * kv_row);
      accum_t S;
      if constexpr (kUseBias) {
        S = attn * attn_scale + mask_head[i];
      } else {
        S = attn * attn_scale;
      }
      accum_t M_old = M;
      M = std::max(M, S);
      accum_t attn_exp = __ESIMD_NS::exp(S - M);
      accum_t L_old = L * __ESIMD_NS::exp(M_old - M);
      L = L_old + attn_exp;
      kv_row = __ESIMD_NS::block_load<scaler_t, head_dim>(
          value_head + i * args.v_seq_step);
      O = (O * L_old + kv_row * attn_exp) / L;
      // if (item.get_global_linear_id() == 1) {
      //   sycl::ext::oneapi::experimental::printf(
      //       "\n\n%d %d S:%f M:%f L:%f attn_exp:%f v:%f", (int)start_ctx_id,
      //       (int)end_ctx_id, S, M, L, attn_exp,
      //       (float)(scaler_t)(kv_row[0]));
      //   sycl::ext::oneapi::experimental::printf(
      //       " %f\n", (float)(scaler_t)(O[0]));
      // }
    }

    __ESIMD_NS::simd<uint32_t, head_dim> offset(
        O_offset + sg_id * sizeof(accum_t), sg_num * sizeof(accum_t));
#pragma unroll
    for (int i = 0; i < head_dim; i += BLK) {
      __ESIMD_NS::simd<uint32_t, BLK> par_offset =
          offset.template select<BLK, 1>(i);
      __ESIMD_NS::simd<accum_t, BLK> par_result = O.template select<BLK, 1>(i);
      __ESIMD_NS::slm_scatter(par_offset, par_result);
    }

    __ESIMD_NS::slm_scalar_store<accum_t>(
        L_offset + sg_id * sizeof(accum_t), L);
    __ESIMD_NS::slm_scalar_store<accum_t>(
        M_offset + sg_id * sizeof(accum_t), M);
    // if (item.get_global_linear_id() == 0) {
    //   sycl::ext::oneapi::experimental::printf("\n\n%f\n", (float)M);
    // }

    __ESIMD_NS::barrier();

    __ESIMD_NS::simd<accum_t, sg_num> M_sg =
        __ESIMD_NS::slm_block_load<accum_t, sg_num>(M_offset);
    accum_t M_total = __ESIMD_NS::detail::reduce<
        accum_t,
        accum_t,
        sg_num,
        __ESIMD_NS::detail::esimd_apply_reduced_max>(M_sg);
    // if (item.get_global_linear_id() == 0) {
    //   sycl::ext::oneapi::experimental::printf("\n\nM_sg\n");
    //   dump_mat_reg(M_sg, sg_num, 1);
    // }

    __ESIMD_NS::simd<accum_t, sg_num> L_sg =
        __ESIMD_NS::slm_block_load<accum_t, sg_num>(L_offset);
    L_sg *= __ESIMD_NS::exp(M_sg - M_total);
    // if (item.get_global_linear_id() == 0) {
    //   sycl::ext::oneapi::experimental::printf("\n\nL_sg\n");
    //   dump_mat_reg(L_sg, sg_num, 1);
    // }
    accum_t L_total = __ESIMD_NS::detail::sum<accum_t, accum_t, sg_num>(L_sg);
    __ESIMD_NS::simd<accum_t, sg_num> L_ratio = L_sg / L_total;
    // if (item.get_global_linear_id() == 0) {
    //   sycl::ext::oneapi::experimental::printf("\n\nL_ratio\n");
    //   dump_mat_reg(L_ratio, sg_num, 1);
    // }

    const size_t start_idx = sg_head_dim * sg_id;
#pragma unroll
    for (size_t i = start_idx; i < start_idx + sg_head_dim; ++i) {
      accum_t O_total = __ESIMD_NS::detail::sum<accum_t, accum_t, sg_num>(
          __ESIMD_NS::slm_block_load<accum_t, sg_num>(
              O_offset + i * sg_num * sizeof(accum_t)) *
          L_ratio

      );
      output_head[i] = O_total;
    }
  }
}; // fmha_forward_t
} // namespace fmha
} // namespace gpu::xetla
