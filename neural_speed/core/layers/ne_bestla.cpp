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
  parallel::Scheduler2D sch({threads, batch, vsize, 1, 16});
  auto threadfunc = [&](int tidx) {
    parallel::ThreadProblem2D tp{tidx};
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
  } else {
    pth->parallel_for(threadfunc);
  }
}

void bestla_add(int batch, int vsize, const float* tensor, const float* vector, int vstep, float* out) {
  auto pth = ne_bestla::ne_threading::get();
  int threads = batch <= 4 ? 1 : pth->num_threads();
  parallel::Scheduler2D sch({threads, batch, vsize, 1, 16});
  auto threadfunc = [&](int tidx) {
    parallel::ThreadProblem2D tp{tidx};
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
  } else {
    pth->parallel_for(threadfunc);
  }
}

static inline bool ne_is_contiguous(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");
  return tensor->nb[0] <= tensor->nb[1] && tensor->nb[1] <= tensor->nb[2] && tensor->nb[2] <= tensor->nb[3];
}

static inline int ne_nrows(const struct ne_tensor* tensor) {
  static_assert(NE_MAX_DIMS == 4, "NE_MAX_DIMS is not 4 - update this function");
  return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

ne_backend bestla_backend_support(struct ne_tensor* src0, struct ne_tensor* src1, enum ne_op op) {
  ne_backend bk = NE_BACKEND_CPU;
#ifdef NS_SYCL
  bool src_on_device = src0->backend == NE_BACKEND_SYCL;
  if (src1) {
    src_on_device |= src1->backend == NE_BACKEND_SYCL;
  }
  switch (op) {
    case NE_OP_MUL_MAT: {
      struct ne_tensor* wei = src0;
      if (src0->type == NE_TYPE_BTLA) {
        bk = src_on_device ? NE_BACKEND_SYCL : NE_BACKEND_CPU;
      }
    } break;
    case NE_OP_RMS_NORM:
    case NE_OP_SILU:
    case NE_OP_ADD:
    case NE_OP_MUL: {
      if (src0->type == NE_TYPE_F32) {
        bk = src_on_device ? NE_BACKEND_SYCL : NE_BACKEND_CPU;
      }
    } break;
    default:
      break;
  }
#endif
  return bk;
}

bool bestla_support(struct ne_tensor* node, int n_threads, size_t* workspace, size_t* dev_workspace) {
  size_t ws_h = 0;
  size_t ws_d = 0;
  bool support = false;
  if (node->backend == NE_BACKEND_SYCL) {
    support = true;
  }
  switch (node->op) {
    case NE_OP_MUL_MAT_ID:
    case NE_OP_MUL_MAT_BIAS:
    case NE_OP_MUL_MAT: {
      struct ne_tensor* wei = node->src0;
      if (node->op == NE_OP_MUL_MAT_ID) {
        wei = node->opt[0];
      }
      if (node->src0->type == NE_TYPE_BTLA) {
        if (node->src0->backend == NE_BACKEND_CPU) {
          ws_h = bestla_f32f32_get_workspace_size(node->src1->ne[1], wei->ne[1], node->src1->ne[0], wei->data);
        }
        support = true;
      }
    } break;
    case NE_OP_ROPE:
      if (node->type == NE_TYPE_BTLA) support = true;
      break;
    case NE_OP_MUL:
    case NE_OP_ADD: {
      if (ne_is_contiguous(node->src1) && ne_is_contiguous(node->src0) &&
          (ne_nrows(node->src1) == 1 || ne_nrows(node->src1) == ne_nrows(node->src0)) &&
          node->src0->ne[0] == node->src1->ne[0] && node->nb[0] == sizeof(float)) {
        support = true;
      }
    } break;
    case NE_OP_MUL_FFN_SILU:
    case NE_OP_MUL_FFN_GELU:
    case NE_OP_MUL_FFN_GELU_MUL:
    case NE_OP_MUL_FFN_ADD_GELU: {
      if (node->src0->backend == NE_BACKEND_CPU) {
        ws_h = bestla_fusion_FFN_f32f32_get_workspace_size(node->src0->ne[1], node->src0->ne[0], node->src1->ne[1],
                                                           node->opt[0]->ne[1], node->src1->data, node->opt[0]->data);
        support = true;
      }
    } break;
    case NE_OP_MUL_ID_FFN_GELU:
    case NE_OP_MUL_ID_FFN_SILU: {
      if (node->src0->backend == NE_BACKEND_CPU) {
        ws_h = bestla_fusion_FFN_f32f32_get_workspace_size(node->src0->ne[1], node->src0->ne[0], node->opt[0]->ne[1],
                                                           node->opt[9]->ne[1], node->opt[0]->data, node->opt[9]->data);
        support = true;
      }
    } break;
    case NE_OP_MUL_QKV: {
      ws_h = bestla_fusion_QKV_f32f32_get_workspace_size(node->src0->ne[1], node->src1->ne[1], node->src1->ne[0],
                                                         node->src1->data);
      support = true;
    } break;
    case NE_OP_NORM:
    case NE_OP_RMS_NORM: {
      if (ne_is_contiguous(node->src0)) {
        support = true;
      }
    } break;
    default:
      break;
  }
  if (support) {
    node->n_tasks = 1;
  }
  *workspace = ws_h;
  *dev_workspace = ws_d;
  return support;
}
