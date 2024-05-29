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

void bestla_init() {
  GetCPUDevice();
  if (_cd->AMX_BF16() || _cd->AMX_INT8()) {
    utils::request_perm_xtile_data();
  }
  _cd->print();
}

void bestla_timer(bool _init) {
  static utils::timer<utils::microseconds> tr;
  if (_init)
    tr.start();
  else
    printf("time :%f us\n", tr.stop());
}

int bestla_set_threads(int _nth) {
  ne_bestla::ne_threading::get()->set_threads(_nth);
  return ne_bestla::ne_threading::get()->num_threads();
}

void* bestla_get_thread_handle() { return ne_bestla::ne_threading::get(); }

void bestla_parallel_for(forward_compute_fptr fcomp, ne_compute_params* mainparams, ne_tensor* node) {
  auto threading = ne_bestla::ne_threading::get();
  if (mainparams->nth == 1) {
    struct ne_compute_params params = *mainparams;
    params.type = NE_TASK_INIT;
    fcomp(&params, node);
    params.type = NE_TASK_COMPUTE;
    fcomp(&params, node);
    params.type = NE_TASK_FINALIZE;
    fcomp(&params, node);
  } else {
    threading->parallel_for([&](int tidx) {
      struct ne_compute_params params = *mainparams;
      params.ith = tidx;
      params.type = NE_TASK_INIT;
      if (tidx == 0) {
        fcomp(&params, node);
      }
      threading->sync(tidx, 0);
      params.type = NE_TASK_COMPUTE;
      if (params.ith < params.nth) {
        fcomp(&params, node);
      }
      threading->sync(tidx, 1);
      params.type = NE_TASK_FINALIZE;
      if (params.ith < params.nth) {
        fcomp(&params, node);
      }
    });
  }
}

void bestla_unpackweight_fp32(void* wptr, int n, int k, float* fp32data, int ld) {
  BTLAGemmUnPackB(fp32data, wptr, static_cast<size_t>(n), static_cast<size_t>(k), static_cast<size_t>(ld),
                  ne_bestla::ne_threading::get());
}

void bestla_packweight_copyattr(const float* f32ptr, void* dstptr, int n, int k, int ld, void* srcptr) {
  auto wtmp = storage::gemm::PackedWeightParser::deserialBuffer(srcptr);
  if (wtmp != nullptr) {
    auto proID = wtmp->mPrologueID;
    if (wtmp->mPrologueID != BTLA_PROLOGUEB_IDS::WeightPack) {
      auto kwtmp = reinterpret_cast<storage::gemm::IWeightKBlockBase*>(wtmp);
      auto coreID = wtmp->mCoreId;
      auto comptype = gemm::CoreAttr::get_comp(coreID);
      auto btype = static_cast<gemm::CompType>(gemm::CompTypeHelper::get_B(comptype));
      ne_comp_type ne_comptype{ne_comp_type::NE_COMP_UNDEF};
      if (btype == gemm::CompType::tBF16) {
        ne_comptype = ne_comp_type::NE_COMP_BF16;
      }
      if (btype == gemm::CompType::tS8) {
        ne_comptype = ne_comp_type::NE_COMP_INT8;
      }
      if (btype == gemm::CompType::tFP32) {
        ne_comptype = ne_comp_type::NE_COMP_F32;
      }
      if (kwtmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNInteger) {
        auto niptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNInteger*>(kwtmp);

        BTLAGemmQuantPackB(dstptr, f32ptr, n, k, ld, niptr->mBlockSize, niptr->mDType, niptr->SDtype(), niptr->IsAsym(),
                           ne_comptype, false, ne_bestla::ne_threading::get());
      } else if (kwtmp->mPrologueID == BTLA_PROLOGUEB_IDS::WeightKBlockNFloat) {
        auto f4ptr = reinterpret_cast<storage::gemm::StorageWeightKBlockNFloat*>(kwtmp);
        BTLAGemmQuantPackB(dstptr, f32ptr, n, k, ld, f4ptr->mBlockSize, f4ptr->mDType, f4ptr->SDtype(), false,
                           ne_comptype, false, ne_bestla::ne_threading::get());
      }
    }
  }
  safe_delete(wtmp);
}

void bestla_layernormalization(int norm_count, int norm_size, bool isrms, float epsilon, const float* FpIn,
                               float* FpOut) {
  BTLALayerNorm(norm_count, norm_size, isrms, epsilon, FpIn, FpOut, ne_threading::get());
}

void bestla_mul(int batch, int vsize, const float* tensor, const float* vector, int vstep, float* out) {
  auto pth = ne_bestla::ne_threading::get();
  int threads = batch <= 4 ? 1 : pth->num_threads();
  parallel::Scheduler2D sch({ threads, batch, vsize, 1, 16 });
  auto threadfunc = [&](int tidx) {
    parallel::ThreadProblem2D tp{ tidx };
    sch.getIndex(tp);
    if (tp.valid) {
      for (size_t i = 0; i < tp.size[0]; i++) {
        auto tptr = tensor + (tp.loc[0] + i) * vsize + tp.loc[1];
        auto vptr = vector + (tp.loc[0] + i) * vstep + tp.loc[1];
        auto dstptr = out + (tp.loc[0] + i) * vsize + tp.loc[1];
        auto ret = kernel::wrapper::Mul<float>::forward_auto(tptr, vptr, dstptr, tp.size[1]);
      }
    }
    };
  if (threads == 1) {
    parallel::SingleThread st;
    st.parallel_for(threadfunc);
  }
  else {
    pth->parallel_for(threadfunc);
  }
}

void bestla_add(int batch, int vsize, const float* tensor, const float* vector, int vstep, float* out) {
  auto pth = ne_bestla::ne_threading::get();
  int threads = batch <= 4 ? 1 : pth->num_threads();
  parallel::Scheduler2D sch({ threads, batch, vsize, 1, 16 });
  auto threadfunc = [&](int tidx) {
    parallel::ThreadProblem2D tp{ tidx };
    sch.getIndex(tp);
    if (tp.valid) {
      for (size_t i = 0; i < tp.size[0]; i++) {
        auto tptr = tensor + (tp.loc[0] + i) * vsize + tp.loc[1];
        auto vptr = vector + (tp.loc[0] + i) * vstep + tp.loc[1];
        auto dstptr = out + (tp.loc[0] + i) * vsize + tp.loc[1];
        auto ret = kernel::wrapper::Add<float>::forward_auto(tptr, vptr, dstptr, tp.size[1]);
      }
    }
    };
  if (threads == 1) {
    parallel::SingleThread st;
    st.parallel_for(threadfunc);
  }
  else {
    pth->parallel_for(threadfunc);
  }
}

#ifdef NS_SYCL
#include "bestla/sycl/sycl_device.h"
#include "bestla/sycl/sycl_storage.h"
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
    ptr->wait();
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
        if (NTile == tAVX512F::NTILE && _cd->AVX512F()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX512F, tAVX512F::ISA> proB;
          proB.convertTransStorage(*sptr, transtor, ne_bestla::ne_threading::get());
        } else if (NTile == tAVX2::NTILE && _cd->AVX2()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX2, tAVX2::ISA> proB;
          proB.convertTransStorage(*sptr, transtor, ne_bestla::ne_threading::get());
        }
      }
      if (btype == gemm::CompType::tS8 && PackRow == 4) {
        if (NTile == tAMX_INT8_SS_KBlock::NTILE && _cd->AMX_INT8()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAMX_INT8_SS_KBlock, tAMX_INT8_SS_KBlock::ISA> proB;
          proB.convertTransStorage(*sptr, transtor, ne_bestla::ne_threading::get());
        } else if (NTile == tAVX512_VNNI_KBlock::NTILE && _cd->AVX512_VNNI()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX512_VNNI_KBlock, tAVX512_VNNI_KBlock::ISA> proB;
          proB.convertTransStorage(*sptr, transtor, ne_bestla::ne_threading::get());
        } else if (NTile == tAVX_VNNI_KBlock::NTILE && _cd->AVX_VNNI()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAVX_VNNI_KBlock, tAVX_VNNI_KBlock::ISA> proB;
          proB.convertTransStorage(*sptr, transtor, ne_bestla::ne_threading::get());
        }
      }
      if (btype == gemm::CompType::tBF16 && PackRow == 2) {
        if (NTile == tAMX_BF16::NTILE && _cd->AMX_BF16()) {
          static prologue_b::gemm::WeightKBlockNInteger<tAMX_BF16, tAMX_BF16::ISA> proB;
          proB.convertTransStorage(*sptr, transtor, ne_bestla::ne_threading::get());
        }
      }
      *dstor = sycl_storage::StorageWeightKBlockNInteger(transtor);
      dstor->assign((int8_t*)deviceptr);
      dstor->fromHost(transtor, (sycl::queue*)device_queue);
    }
  }
}

#include "bestla/sycl/sycl_gemm.h"
#include "bestla/sycl/sycl_prologue_b.h"
#include "bestla/sycl/sycl_wrapper.h"
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
  if (_m == 1) {
    using ProB = ProBTransT<GemmCore>;
    auto e_esimd = ProB::gemv(activation, {(uint8_t*)dstor->mQBuf, (float*)dstor->mSBuf, dstor->mCStep}, output, _n, _k,
                              dstor->mBlockSize, (sycl::queue*)queue);
  } else {
    using KernelTLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBTransT, EpiT, GemmCore>;
    utils::GemmProblem gp(1, _m, _n, _k);
    auto e_esimd = KernelTLauncher::compute(
        (sycl::queue*)queue, _m, _n, _k, dstor->mBlockSize,
        {{activation, lda}, {(uint8_t*)dstor->mQBuf, (float*)dstor->mSBuf, dstor->mCStep}, {output, ldo}});
  }
}
#endif
