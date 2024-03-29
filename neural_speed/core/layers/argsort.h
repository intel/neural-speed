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

#pragma once
#include "core/ne.h"
#include "core/data_types.h"

#ifdef __cplusplus
extern "C" {
#endif

void ne_compute_forward_argsort(const struct ne_compute_params* params, const struct ne_tensor* src0,
                                struct ne_tensor* dst);

#ifdef __cplusplus
}
#endif
