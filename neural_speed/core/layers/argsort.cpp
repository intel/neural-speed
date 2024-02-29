//  Copyright (c) 2024 Intel Corporation
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

#include "argsort.h"
#include <algorithm>
#include <cstdio>

static void ne_compute_forward_argsort_f32(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                           struct ne_tensor* dst) {
  if (params->type == NE_TASK_INIT || params->type == NE_TASK_FINALIZE) {
    return;
  }
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
  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t nr = src0->ne[1] * src0->ne[2] * src0->ne[3];

  for (int64_t i = ith; i < nr; i += nth) {
    int32_t* dst_data = (int32_t*)((char*)dst->data + i * nb1);
    const float* src_data = (float*)((char*)src0->data + i * nb01);

    for (int64_t j = 0; j < ne0; j++) {
      dst_data[j] = j;
    }
    std::sort(dst_data, dst_data + ne0, [src_data](int pos1, int pos2) { return (src_data[pos1] > src_data[pos2]); });
  }
}
void ne_compute_forward_argsort(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                struct ne_tensor* dst) {
  switch (src0->type) {
    case NE_TYPE_F32: {
      ne_compute_forward_argsort_f32(params, src0, dst);
    } break;
    default: {
      NE_ASSERT(false);
    } break;
  }
}
