
#pragma once
/*
Fused Multi-Head Attention Forward

This is an implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf)
*/
#include "xetla.hpp"

namespace gpu::xetla {
namespace fmha {
template <
    typename scaler_t,
    gpu_arch arch_tag,
    int sg_num,
    int head_dim,
    int kv_group_size,
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

    accum_t sm_scale;

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
  static constexpr size_t O_size =
      kv_group_size * sg_num * head_dim * sizeof(accum_t);
  static constexpr size_t L_offset = O_size;
  static constexpr size_t L_size = kv_group_size * sg_num * sizeof(accum_t);
  static constexpr size_t M_offset = L_offset + L_size;
  static constexpr size_t M_size = kv_group_size * sg_num * sizeof(accum_t);
  xetla_nbarrier_t<sg_num, sg_num, arch_tag> nbarrier;

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
    sycl::range<3> group_range{num_batch, num_heads / kv_group_size, 1};
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
    const size_t sg_ctx_len = args.ctx_len / sg_num;
    const size_t rem_ctx_len = args.ctx_len % sg_num;
    const size_t sg_head_dim = head_dim / sg_num;
    static_assert(head_dim % sg_num == 0);

    const size_t batch_idx = item.get_group(0);
    const size_t kv_head_idx = item.get_group(1);
    const size_t head_idx = kv_head_idx * kv_group_size;
    const size_t sg_id = item.get_local_id(2);
    xetla_local_init<get_slm_size()>();
    nbarrier.init_nbarrier(sg_id, nbarrier_role::producer_consumer);

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

    xetla_vector<fp16, kv_group_size * head_dim> q;
    constexpr int BLK = LOAD_BYTES_LEN / sizeof(scaler_t);
    static_assert(head_dim % BLK == 0);
#pragma unroll
    for (int i = 0; i < kv_group_size; ++i)
#pragma unroll
      for (int j = 0; j < head_dim; j += BLK) {
        q.xetla_select<BLK, 1>(j + i * head_dim) =
            xetla_load_global<scaler_t, BLK>(
                query_head, sizeof(scaler_t) * (i * args.q_head_step + j));
      }

    // if (batch_idx == 0 && head_idx == 0 & sg_id == 0) {
    //   dump_mat_reg(q, head_dim * kv_group_size, 1);
    // }
    size_t start_ctx_id = sg_ctx_len * sg_id + std::min(sg_id, rem_ctx_len);
    size_t end_ctx_id =
        start_ctx_id + sg_ctx_len + (sg_id < rem_ctx_len ? 1 : 0);

    // M_i = max(S_i)
    xetla_vector<accum_t, kv_group_size> M =
        std::numeric_limits<accum_t>::lowest();
    // L_i = sum(exp(S_i) - M_i)
    xetla_vector<accum_t, kv_group_size> L =
        std::numeric_limits<accum_t>::min();
    xetla_vector<accum_t, kv_group_size* head_dim> O = 0;
    for (size_t i = start_ctx_id; i < end_ctx_id; ++i) {
      const auto k_i = xetla_load_global<scaler_t, head_dim>(
          key_head, sizeof(scaler_t) * i * args.k_seq_step);
      xetla_vector<accum_t, kv_group_size> attn;
#pragma unroll
      for (int j = 0; j < kv_group_size; ++j)
        attn[j] = xetla_reduce<accum_t, accum_t, head_dim, reduce_op::sum>(
            q.xetla_select<head_dim, 1>(j * head_dim) * k_i);
      xetla_vector<accum_t, kv_group_size> S;
      if constexpr (kUseBias) {
        S = attn * args.sm_scale + mask_head[i];
      } else {
        S = attn * args.sm_scale;
      }
      xetla_vector<accum_t, kv_group_size> M_old = M;
      M = xetla_max<accum_t>(M, S);
      xetla_vector<accum_t, kv_group_size> attn_exp =
          __ESIMD_NS::exp<accum_t>(S - M);
      xetla_vector<accum_t, kv_group_size> L_old =
          L * __ESIMD_NS::exp<accum_t>(M_old - M);
      L = L_old + attn_exp;
      const auto v_i = xetla_load_global<scaler_t, head_dim>(
          value_head, sizeof(scaler_t) * i * args.v_seq_step);

#pragma unroll
      for (int j = 0; j < kv_group_size; ++j)
        O.xetla_select<head_dim, 1>(j * head_dim) =
            (O.xetla_select<head_dim, 1>(j * head_dim) * L_old[j] +
             v_i * accum_t(attn_exp[j])) /
            accum_t(L[j]);

      // if (batch_idx == 0 && head_idx == 0 & sg_id == 0) {
      //   dump_mat_reg(O, head_dim * kv_group_size, 1);
      // }
    }

#pragma unroll
    for (int j = 0; j < kv_group_size; ++j)
#pragma unroll
      for (int i = 0; i < head_dim; i += BLK) {
        xetla_vector<uint32_t, BLK> offset_i(
            O_offset +
                (j * sg_num * head_dim + sg_id + i * sg_num) * sizeof(accum_t),
            sg_num * sizeof(accum_t));
        xetla_vector<accum_t, BLK> O_i =
            O.xetla_select<BLK, 1>(j * head_dim + i);
        xetla_store_local<accum_t, 1, data_size::default_size, BLK>(
            offset_i, O_i);
      }

    xetla_store_local<accum_t, kv_group_size>(
        L_offset + sg_id * kv_group_size * sizeof(accum_t), L);
    xetla_store_local<accum_t, kv_group_size>(
        M_offset + sg_id * kv_group_size * sizeof(accum_t), M);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    xetla_vector<accum_t, kv_group_size* sg_num> M_sg =
        xetla_load_local<accum_t, kv_group_size * sg_num>(M_offset);
    xetla_vector<accum_t, kv_group_size> M_total;
#pragma unroll
    for (int i = 0; i < kv_group_size; ++i)
      M_total[i] = xetla_reduce<accum_t, accum_t, sg_num, reduce_op::max>(
          M_sg.xetla_select<sg_num, 1>(i * sg_num));

    xetla_vector<accum_t, kv_group_size* sg_num> L_sg =
        xetla_load_local<accum_t, kv_group_size * sg_num>(L_offset);
#pragma unroll
    for (int i = 0; i < kv_group_size; ++i)
      L_sg.xetla_select<sg_num, 1>(i * sg_num) *= xetla_exp<accum_t, sg_num>(
          M_sg.xetla_select<sg_num, 1>(i * sg_num) - M_total[i]);
    xetla_vector<accum_t, kv_group_size> L_total;
#pragma unroll
    for (int i = 0; i < kv_group_size; ++i)
      L_total[i] = xetla_reduce<accum_t, accum_t, sg_num, reduce_op::sum>(
          L_sg.xetla_select<sg_num, 1>(i * sg_num));
    xetla_vector<accum_t, kv_group_size * sg_num> L_ratio;
#pragma unroll
    for (int i = 0; i < kv_group_size; ++i)
      L_ratio.xetla_select<sg_num, 1>(i * sg_num) =
          L_sg.xetla_select<sg_num, 1>(i * sg_num) / L_total[i];

    // if (batch_idx == 0 && head_idx == 0 & sg_id == 0) {
    //   dump_mat_reg(L_ratio, kv_group_size * sg_num, 1);
    // }
    const size_t start_idx = sg_head_dim * sg_id;
#pragma unroll
    for (int j = 0; j < kv_group_size; ++j) {
#pragma unroll
      for (size_t i = start_idx; i < start_idx + sg_head_dim; ++i) {
        auto tmp = xetla_load_local<accum_t, sg_num>(
            O_offset + (i * sg_num + j * head_dim * sg_num) * sizeof(accum_t));
        accum_t O_total =
            xetla_reduce<accum_t, accum_t, sg_num, reduce_op::sum>(
                tmp * L_ratio.xetla_select<sg_num, 1>(j * sg_num)

            );
        output_head[i + j * args.out_head_step] = O_total;
      }
    }
  }
}; // fmha_forward_t
} // namespace fmha
} // namespace gpu::xetla
