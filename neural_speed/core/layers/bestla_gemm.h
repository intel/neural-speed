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

/*++
Module Name:

    bestla_gemm.h

Abstract:

    C APIs of BesTLA GEMMs.
--*/

#pragma once

#include "data_types.h"
#include "bestla/bestla.h"

struct BTLA_GEMM_DATA_PACKED_PARAMS {
  const float* A = nullptr; /**< address of A (float32 matrix)*/
  const void* B = nullptr;  /**< address of B (packed nbits blob)*/
  float* C = nullptr;       /**< address of result matrix */
  int lda = 0;              /**< leading dimension of A */
  int ldc = 0;              /**< leading dimension of C*/
};

size_t BTLAGemmPackBSize(size_t N, size_t K, size_t BlkSize, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym,
                         ne_comp_type CompType, int* shuffle_indice);

bool BTLAGemmQuantPackB(void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb, size_t BlkSize,
                        BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym, ne_comp_type CompType, bool isTrans,
                        void* ThreadPool);

// QData:  K*N quantized int8 weight
// Scales: K/BlkSize * N scales
// Zp:     K/BlkSize * N zero points
bool BTLAGemmPackB(void* PackedBuf, const int8_t* QData, const float* Scales, const int8_t* Zp, size_t N, size_t K,
                   size_t ldb, size_t BlkSize, BTLA_DTYPE QuantType, BTLA_DTYPE ScaleDtype, bool isAsym,
                   ne_comp_type CompType, int* shuffle_indice, void* ThreadPool);

bool BTLAGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* ThreadPool);

bool BTLAGemmBatchDriver(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                         const BTLA_GEMM_DATA_PACKED_PARAMS* DataParams, int8_t* WorkSpace, void* ThreadPool);

bool BTLALayerNorm(size_t norm_count, size_t norm_size, bool isrms, float epsilon, const float* FpIn, float* FpOut,
                   void* ThreadPool);
