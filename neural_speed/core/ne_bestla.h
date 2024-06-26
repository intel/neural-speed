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
#ifndef NE_CORE_GRAPH_INNER_PRODUCT_H
#define NE_CORE_GRAPH_INNER_PRODUCT_H
#include "ne.h"
#ifdef __cplusplus
extern "C" {
#endif

void bestla_timer(bool _init);

int bestla_set_threads(int _nth);

void* bestla_get_thread_handle();

typedef void (*forward_compute_fptr)(struct ne_compute_params*, struct ne_tensor*);

void bestla_parallel_for(forward_compute_fptr, struct ne_compute_params*, struct ne_tensor*);

void bestla_init();

unsigned long long bestla_f32f32_get_workspace_size(int _m, int _n, int _k, void* wptr);

void bestla_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda, int ldo,
                           void* workspace);

bool bestla_fusion_add_f32f32_support(void* weiptr, int _m, int _n, int _k);
void bestla_fusion_add_f32f32_forward(float* activation, void* weiptr, float* bias, float* output, int _m, int _n,
                                      int _k, int lda, int ldo, bool boardcast_bias, void* workspace);

unsigned long long bestla_fusion_QKV_f32f32_get_workspace_size(int _m, int _n, int _k, void* w1ptr);

bool bestla_fusion_QKV_f32f32_support(void* wqptr, void* wkptr, void* wvptr, int _m, int _n, int _k);

void bestla_fusion_QKV_f32f32_forward(float* activation, void* wqptr, void* wkptr, void* wvptr, float* output, int _m,
                                      int _n, int _k, int lda, int ldo, void* workspace);

unsigned long long bestla_fusion_FFN_f32f32_get_workspace_size(int seq, int fin, int fmid, int fout, void* w1ptr,
                                                               void* w2ptr);

bool bestla_fusion_FFN_Gelu_Mul_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid,
                                               int fout);
void bestla_fusion_FFN_Gelu_Mul_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                               float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                               void* workspace);

bool bestla_fusion_FFN_SiLu_f32f32_support(void* w1ptr, void* w2ptr, void* w3ptr, int seq, int fin, int fmid, int fout);
void bestla_fusion_FFN_SiLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, void* w3ptr, float* tmp1,
                                           float* tmp2, float* output, int seq, int fin, int fmid, int fout,
                                           void* workspace);

bool bestla_fusion_FFN_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout);
void bestla_fusion_FFN_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* tmp1, float* output,
                                           int seq, int fin, int fmid, int fout, void* workspace);

bool bestla_fusion_FFN_Add_GeLu_f32f32_support(void* w1ptr, void* w2ptr, int seq, int fin, int fmid, int fout);
void bestla_fusion_FFN_Add_GeLu_f32f32_forward(float* activation, void* w1ptr, void* w2ptr, float* b1ptr, float* b2ptr,
                                               float* tmp1, float* output, int seq, int fin, int fmid, int fout,
                                               bool boardcast_bias, void* workspace);

void bestla_unpackweight_fp32(void* wptr, int n, int k, float* fp32data, int ld);
// packweight to dstptr, copy weight attributes from srcptr
void bestla_packweight_copyattr(const float* f32ptr, void* dstpr, int n, int k, int ld, void* srcptr);

void bestla_layernormalization(int norm_count, int norm_size, bool isrms, float epsilon, const float* FpIn,
                               float* FpOut);

void bestla_mul(int batch, int vsize, const float* tensor, const float* vector, int vstep, float* out);
void bestla_add(int batch, int vsize, const float* tensor, const float* vector, int vstep, float* out);

enum ne_backend bestla_backend_support(struct ne_tensor* src0, struct ne_tensor* src1, enum ne_op op);
bool bestla_support(struct ne_tensor* node, int n_threads, size_t* workspace, size_t* dev_workspace);

#ifdef NS_SYCL
void* bestla_create_device(bool profile);
void* bestla_get_device_queue(void* device);
void bestla_release_device(void* device);
size_t bestla_device_gmem_size(void* device);
void* bestla_device_malloc(size_t size, void* queue);
void bestla_device_free(void* ptr, void* queue);
void bestla_device_memcpy(void* dstptr, const void* srcptr, size_t size, void* queue);
void bestla_device_memcpy_sync(void* dstptr, const void* srcptr, size_t size, void* queue);
void bestla_device_sync(void* queue);
size_t bestla_device_storage_size();
void bestla_device_load_storage(void* hoststor, void* devstor, void* deviceptr, void* queue);
void bestla_device_f32f32_forward(float* activation, void* weiptr, float* output, int _m, int _n, int _k, int lda,
                                  int ldo, void* workspace, void* queue);
void bestla_device_mul_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                           const struct ne_tensor* src1, struct ne_tensor* dst);
void bestla_device_add_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                           const struct ne_tensor* src1, struct ne_tensor* dst);
void bestla_device_elewise_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                               struct ne_tensor* dst);
void bestla_device_rms_norm_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                struct ne_tensor* dst);
void bestla_device_rope_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                            const struct ne_tensor* src1, struct ne_tensor* dst);
void bestla_device_dup_f32(const struct ne_compute_params* params, const struct ne_tensor* src0, struct ne_tensor* dst);
void bestla_device_mha_f32(const struct ne_compute_params* params, const struct ne_tensor* q, const struct ne_tensor* k,
                           const struct ne_tensor* v, struct ne_tensor* dst);
#endif
#ifdef __cplusplus
}
#endif
#endif  // NE_CORE_GRAPH_INNER_PRODUCT_H
