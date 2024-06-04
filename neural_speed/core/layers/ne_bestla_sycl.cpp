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

void bestla_device_mul_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
  const struct ne_tensor* src1, struct ne_tensor* dst)
{

}
#endif
