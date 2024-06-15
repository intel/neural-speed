//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include "bestla_common.hpp"
#include "bestla_gemm.h"
using namespace bestla;     // NOLINT
using namespace ne_bestla;  // NOLINT

#ifdef NS_SYCL
#include "bestla/sycl/sycl_device.h"
#include "bestla/sycl/sycl_storage.h"
#include "bestla/sycl/sycl_gemm.h"
#include "bestla/sycl/sycl_prologue_b.h"
#include "bestla/sycl/sycl_wrapper.h"

void* bestla_create_device(bool profile) {
  auto ptr = new sycl_device::SyclDevice(profile);
  ptr->print();
  return ptr;
}

void* bestla_get_device_queue(void* device) {
  if (device) {
    auto ptr = (sycl_device::SyclDevice*)device;
    auto q = ptr->getQueue();
    return q;
  }
  return NULL;
}

void bestla_release_device(void* device) {
  if (device) {
    auto ptr = (sycl_device::SyclDevice*)device;
    delete ptr;
  }
}

size_t bestla_device_gmem_size(void* device) {
  if (device) {
    auto ptr = (sycl_device::SyclDevice*)device;
    return ptr->getGlobalMemSize();
  }
}

void* bestla_device_malloc(size_t size, void* queue) {
  if (queue) {
    auto ptr = (sycl::queue*)queue;
    auto tmp = sycl::malloc_device<char>(size, *ptr);
    return tmp;
  }
}

void bestla_device_free(void* obj, void* queue) {
  if (queue && obj) {
    auto ptr = (sycl::queue*)queue;
    sycl::free(obj, *ptr);
  }
}

void bestla_device_memcpy_sync(void* dstptr, const void* srcptr, size_t size, void* queue) {
  if (queue && srcptr && dstptr) {
    auto ptr = (sycl::queue*)queue;
    ptr->memcpy(dstptr, srcptr, size);
    ptr->wait();
  }
}

void bestla_device_memcpy(void* dstptr, const void* srcptr, size_t size, void* queue) {
  if (queue && srcptr && dstptr) {
    auto ptr = (sycl::queue*)queue;
    ptr->memcpy(dstptr, srcptr, size);
  }
}

void bestla_device_sync(void* queue) {
  if (queue) {
    auto ptr = (sycl::queue*)queue;
    ptr->wait();
  }
}

size_t bestla_device_storage_size() { return sizeof(sycl_storage::StorageWeightKBlockNInteger); }

void bestla_device_load_storage(void* hoststor, void* devstor, void* deviceptr, void* device_queue) {
  auto ptr = storage::gemm::PackedWeightParser::deserialBuffer(const_cast<void*>(hoststor));
  GetCPUDevice();
  if (ptr && devstor && deviceptr) {
    auto dstor = (sycl_storage::StorageWeightKBlockNInteger*)devstor;
    if (ptr->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
      auto sptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(ptr);
      auto transtor = sptr->toTrans();
      utils::avector<int8_t> buffer1(transtor.mSize);
      transtor.assign(buffer1.data());
      auto coretype = sptr->mCoreId;
      auto NTile = gemm::CoreAttr::get_mask_val(sptr->mCoreId, gemm::CoreAttr::NTILE_MASK, gemm::CoreAttr::NTILE_SHIFT);
      auto PackRow = gemm::CoreAttr::get_packrow(sptr->mCoreId);
      auto CType = gemm::CoreAttr::get_comp(sptr->mCoreId);
      auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(CType));
      if (btype == gemm::CompType::tFP32 && PackRow == 1) {
        if (NTile == tAVX512F::NTILE) {
          prologue_b::gemm::WeightKBlockNInteger<tAVX512F>::convertTransStorage(*sptr, transtor,
                                                                                ne_bestla::ne_threading::get());
        } else if (NTile == tAVX2::NTILE) {
          prologue_b::gemm::WeightKBlockNInteger<tAVX2>::convertTransStorage(*sptr, transtor,
                                                                             ne_bestla::ne_threading::get());
        }
      }
      if (btype == gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE) {
          prologue_b::gemm::WeightKBlockNInteger<tAMX_INT8_SS_KBlock>::convertTransStorage(
              *sptr, transtor, ne_bestla::ne_threading::get());
        } else if (NTile == tAVX512_VNNI_KBlock::NTILE) {
          prologue_b::gemm::WeightKBlockNInteger<tAVX512_VNNI_KBlock>::convertTransStorage(
              *sptr, transtor, ne_bestla::ne_threading::get());
        } else if (NTile == tAVX_VNNI_KBlock::NTILE) {
          prologue_b::gemm::WeightKBlockNInteger<tAVX_VNNI_KBlock>::convertTransStorage(*sptr, transtor,
                                                                                        ne_bestla::ne_threading::get());
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE) {
          prologue_b::gemm::WeightKBlockNInteger<tAMX_BF16>::convertTransStorage(*sptr, transtor,
                                                                                 ne_bestla::ne_threading::get());
        }
      }
      *dstor = sycl_storage::StorageWeightKBlockNInteger(transtor);
      dstor->assign((int8_t*)deviceptr);
      dstor->fromHost(transtor, (sycl::queue*)device_queue);
    }
  }
}

template <class GCT>
using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
template <class GCT>
using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
template <class GCT>
using EpiT = sycl_epilogue::OutputBase<GCT, float>;
void bestla_device_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda,
                                  int ldo, void* workspace, void* queue) {
  using GemmCore = sycl_gemm::xve::DefaultSGemmCore;
  auto dstor = (sycl_storage::StorageWeightKBlockNInteger*)weiptr;
  auto q = (sycl::queue*)queue;
  if (_m == 1) {
    using ProB = ProBTransT<GemmCore>;
    ProB::gemv(activation, {(uint8_t*)dstor->mQBuf, (float*)dstor->mSBuf, dstor->mCStep}, output, _n, _k,
               dstor->mBlockSize, q);
  } else {
    using KernelTLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBTransT, EpiT, GemmCore>;
    utils::GemmProblem gp(1, _m, _n, _k);
    KernelTLauncher::compute(
        q, _m, _n, _k, dstor->mBlockSize,
        {{activation, lda}, {(uint8_t*)dstor->mQBuf, (float*)dstor->mSBuf, dstor->mCStep}, {output, ldo}});
  }
  if (sycl_device::SyclDevice::is_cpu(q)) {
    q->wait();
  }
}

void bestla_device_mul_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                           const struct ne_tensor* src1, struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  auto q = (sycl::queue*)params->dev_queue;

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = ne11 == 1 ? 0 : src1->nb[1];
  const size_t nb12 = ne12 == 1 ? 0 : src1->nb[2];
  const size_t nb13 = ne13 == 1 ? 0 : src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];
  sycl::range<1> num_items{ne00 * ne01 * ne02 * ne03};
  auto src0ptr = (float*)src0->data;
  auto src1ptr = (float*)src1->data;
  auto dstptr = (float*)dst->data;
  auto ev = q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(num_items, [=](auto it) {
      int i = it;
      int i00 = i % ne00;
      i /= ne00;
      int i01 = i % ne01;
      i /= ne01;
      int i02 = i % ne02;
      i /= ne02;
      int i03 = i % ne03;

      int i13 = i03 % ne13;
      int i12 = i02 % ne12;
      int i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dstptr + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0ptr + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float* src1_ptr = (float*)((char*)src1ptr + i13 * nb13 + i12 * nb12 + i11 * nb11);
      dst_ptr[i00] = src0_ptr[i00] * src1_ptr[i00];
    });
  });
  if (sycl_device::SyclDevice::is_cpu(q)) {
    q->wait();
  }
}

void bestla_device_add_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                           const struct ne_tensor* src1, struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  auto q = (sycl::queue*)params->dev_queue;

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb10 = src1->nb[0];
  const size_t nb11 = ne11 == 1 ? 0 : src1->nb[1];
  const size_t nb12 = ne12 == 1 ? 0 : src1->nb[2];
  const size_t nb13 = ne13 == 1 ? 0 : src1->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];
  sycl::range<1> num_items{ne00 * ne01 * ne02 * ne03};
  auto src0ptr = (float*)src0->data;
  auto src1ptr = (float*)src1->data;
  auto dstptr = (float*)dst->data;
  auto ev = q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(num_items, [=](auto it) {
      int i = it;
      int i00 = i % ne00;
      i /= ne00;
      int i01 = i % ne01;
      i /= ne01;
      int i02 = i % ne02;
      i /= ne02;
      int i03 = i % ne03;

      int i13 = i03 % ne13;
      int i12 = i02 % ne12;
      int i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dstptr + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0ptr + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float* src1_ptr = (float*)((char*)src1ptr + i13 * nb13 + i12 * nb12 + i11 * nb11);
      dst_ptr[i00] = src0_ptr[i00] + src1_ptr[i00];
    });
  });
  if (sycl_device::SyclDevice::is_cpu(q)) {
    q->wait();
  }
}

void bestla_device_elewise_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                               struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }

  auto q = (sycl::queue*)params->dev_queue;
  auto op = dst->op;
  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  auto srcptr = (float*)src0->data;
  auto dstptr = (float*)dst->data;
  sycl::range<1> num_items{ne00 * ne01 * ne02 * ne03};
  auto ev = q->submit([&](sycl::handler& cgh) {
    cgh.parallel_for(num_items, [=](auto it) {
      int i = it;
      float srcval = srcptr[i];
      if (op == NE_OP_SILU) {
        srcval = ne_silu_f32(srcval);
      }
      dstptr[i] = srcval;
    });
  });
  if (sycl_device::SyclDevice::is_cpu(q)) {
    q->wait();
  }
}

void bestla_device_rms_norm_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }
  auto q = (sycl::queue*)params->dev_queue;
  float eps;
  memcpy(&eps, dst->op_params, sizeof(float));
  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];
  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];
  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];
  int64_t constexpr WgSize = 1024;
  int constexpr SgSize = 16;
  int64_t ne00_ = bestla::utils::padto_le(ne00, WgSize);
  auto src0ptr = (float*)src0->data;
  auto dstptr = (float*)dst->data;
  auto ev = q->submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> slm(sycl::range(WgSize), cgh);
    cgh.parallel_for(sycl::nd_range<1>(ne01 * ne02 * ne03 * WgSize, WgSize),
                     [=](auto it) [[intel::reqd_sub_group_size(SgSize)]] {
                       auto sg = it.get_sub_group();
                       auto sg_idx = sg.get_group_id()[0];
                       auto wg_idx = it.get_group(0);
                       auto wg_loc_id = it.get_local_id();
                       auto lane_id = sg.get_local_id()[0];
                       int i = wg_idx;
                       int i01 = i % ne01;
                       i /= ne01;
                       int i02 = i % ne02;
                       i /= ne02;
                       int i03 = i % ne03;

                       float* dst_ptr = (float*)((char*)dstptr + i03 * nb3 + i02 * nb2 + i01 * nb1);
                       float* src0_ptr = (float*)((char*)src0ptr + i03 * nb03 + i02 * nb02 + i01 * nb01);
                       float sum = 0.0;
                       int64_t i00 = wg_loc_id;
                       for (; i00 < ne00_; i00 += WgSize) {
                         sum += (src0_ptr[i00] * src0_ptr[i00]);
                       }
                       if (i00 < ne00) {
                         sum += (src0_ptr[i00] * src0_ptr[i00]);
                       }
                       slm[wg_loc_id] = sum;
                       it.barrier(sycl::access::fence_space::local_space);
                       if (sg_idx == 0) {
                         for (size_t i = wg_loc_id; i < WgSize - SgSize; i += SgSize) {
                           sum += slm[i + SgSize];
                         }
                         float gsum = 0.f;
                         for (int i = 0; i < SgSize; i += 1) {
                           gsum += sg.shuffle(sum, i);
                         }
                         float mean = gsum / ne00;
                         const float scale = 1.0f / sqrtf(mean + eps);
                         slm[0] = scale;
                       }
                       it.barrier(sycl::access::fence_space::local_space);

                       float scale = slm[0];
                       i00 = wg_loc_id;
                       for (; i00 < ne00_; i00 += WgSize) {
                         dst_ptr[i00] = src0_ptr[i00] * scale;
                       }
                       if (i00 < ne00) {
                         dst_ptr[i00] = src0_ptr[i00] * scale;
                       }
                     });
  });
  if (sycl_device::SyclDevice::is_cpu(q)) {
    q->wait();
  }
}

extern void ggml_rope_yarn_corr_dims(int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow,
                                     float dims[2]);

static float rope_yarn_ramp(const float low, const float high, const int i0) {
  const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
  return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(float theta_extrap, float freq_scale, float corr_dims0, float corr_dims1, int64_t i0,
                      float ext_factor, float mscale, float* cos_theta, float* sin_theta) {
  // Get n-d rotational scaling corrected for extrapolation
  float theta_interp = freq_scale * theta_extrap;
  float theta = theta_interp;
  if (ext_factor != 0.0f) {
    float ramp_mix = rope_yarn_ramp(corr_dims0, corr_dims1, i0) * ext_factor;
    theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

    // Get n-d magnitude scaling corrected for interpolation
    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
  }
  *cos_theta = cosf(theta) * mscale;
  *sin_theta = sinf(theta) * mscale;
}

void bestla_device_rope_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                            const struct ne_tensor* src1, struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }
  auto q = (sycl::queue*)params->dev_queue;
  const int bs = src0->ne[3];
  NE_ASSERT(src1->type == NE_TYPE_I32);

  const float freq_base = ((float*)(dst->op_params))[0];
  const float freq_scale = 1 / ((float*)(dst->op_params))[1];
  const int n_orig_ctx = (int)((float*)(dst->op_params))[2];
  const float ext_factor = ((float*)(dst->op_params))[3];
  const float attn_factor = ((float*)(dst->op_params))[4];
  const float beta_fast = ((float*)(dst->op_params))[5];
  const float beta_slow = ((float*)(dst->op_params))[6];
  const float scale_factor = ((float*)(dst->op_params))[7];
#define ROPE_PARAMS_NUM 5
#define ROPE_NPAST_IDX 0
#define ROPE_NDIMS_IDX 1
#define ROPE_MODE_IDX 2
#define ROPE_PROMPTSIZE_IDX 3
#define ROPE_NKEEP_IDX 4
#define ROPE_PADDING_IDX 5
  const int64_t n_past = ((int32_t*)src1->data)[ROPE_NPAST_IDX];
  const int64_t n_dims = ((int32_t*)src1->data)[ROPE_NDIMS_IDX];
  const int64_t mode = ((int32_t*)src1->data)[ROPE_MODE_IDX];
  const int64_t prompt_size = ((int32_t*)src1->data)[ROPE_PROMPTSIZE_IDX];
  const int64_t n_keep = ((int32_t*)src1->data)[ROPE_NKEEP_IDX];
  assert(n_past >= 0);

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];
  const int64_t ne02 = src0->ne[2];
  const int64_t ne03 = src0->ne[3];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];
  const int64_t ne2 = dst->ne[2];
  const int64_t ne3 = dst->ne[3];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];
  const size_t nb02 = src0->nb[2];
  const size_t nb03 = src0->nb[3];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];
  const size_t nb2 = dst->nb[2];
  const size_t nb3 = dst->nb[3];

  const int nr = ne1 * ne2 * ne3;

  const float theta_scale = powf(freq_base, -2.0f / n_dims);
  const float inv_ndims = -1.f / n_dims;
  float corr_dims[2];
  ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims);
  float corr_dims0 = corr_dims[0];
  float corr_dims1 = corr_dims[1];
  int constexpr SgSize = 16;
  auto src0ptr = (float*)src0->data;
  auto dstptr = (float*)dst->data;
  auto ev = q->submit([&](sycl::handler& cgh) {
    // sycl::local_accessor<float, 1> slm(sycl::range(WgSize), cgh);
    cgh.parallel_for(sycl::nd_range<1>(nr * SgSize, SgSize), [=](auto it) [[intel::reqd_sub_group_size(SgSize)]] {
      auto sg = it.get_sub_group();
      auto sg_idx = sg.get_group_id()[0];
      auto wg_idx = it.get_group(0);
      auto wg_loc_id = it.get_local_id();
      auto lane_id = sg.get_local_id()[0];
      int i = wg_idx;
      int i1 = i % ne1;
      i /= ne1;
      int i2 = i % ne2;
      i /= ne2;
      int i3 = i % ne3;

      const int64_t p = n_past + i2;
      float theta_base = (float)p;
      for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
        float cos_theta, sin_theta;
        rope_yarn(theta_base, freq_scale, corr_dims0, corr_dims1, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

        theta_base *= theta_scale;

        const float* const src = (float*)((char*)src0ptr + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
        float* dst_data = (float*)((char*)dstptr + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

        const float x0 = src[0];
        const float x1 = src[1];

        dst_data[0] = x0 * cos_theta - x1 * sin_theta;
        dst_data[1] = x0 * sin_theta + x1 * cos_theta;
      }
    });
  });
  if (sycl_device::SyclDevice::is_cpu(q)) {
    q->wait();
  }
}
#endif
