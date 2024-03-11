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
