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

#pragma once
#include "bestla/bestla_prologue_b.h"
#include "bestla/bestla_wrapper.h"

namespace bestla {
template <class GemmCore_T, template <class, BTLA_ISA> class Wei_T>
using tLauncher_Fp_F32F32 =
    wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockBaseF32, Wei_T,
                                  epilogue::gemm::CompFp32BlockEpilogue, epilogue::gemm::AccumulatorWriteBackFp32>;

template <class GemmCore_T, template <class, BTLA_ISA> class Wei_T>
using tLauncher_Int8_F32F32 =
    wrapper::gemm::LauncherIntKBlock<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ShuffleActivationKBlockQuantizeF32,
                                     Wei_T, epilogue::gemm::AccumulatorWriteBackFp32>;

using tAVX2 = gemm::SCoreRowNAvx2<24, 4>;
using tAVX_VNNI = gemm::ICoreRowNAvxvnni<24, 4>;
using tAVX512F = gemm::SCoreRowNAvx512f<48, 8>;
using tAVX512_VNNI = gemm::ICoreRowNAvx512vnni<48, 8>;
using tAMX_BF16 = gemm::HCoreRowNAmxbf16<48, 16>;
using tAVX512_BF16 = gemm::HCoreRowNAvx512bf16<48, 8>;
using tAVX512_FP16 = gemm::HCoreRowNAvx512fp16<96, 8>;
using tAMX_INT8_US = gemm::ICoreRowNAmxint8<48, 16>;
using tAMX_INT8_SS = gemm::ICoreRowNAmxint8SS<48, 16>;

using tAVX_VNNI_KBlock = gemm::ICoreRowNAvxvnniKBlock<24, 2>;
using tAVX512_VNNI_KBlock = gemm::ICoreRowNAvx512vnniKBlock<48, 4>;
using tAMX_INT8_US_KBlock = gemm::ICoreRowNAmxint8KBlock<48, 16>;
using tAMX_INT8_SS_KBlock = gemm::ICoreRowNAmxint8SSKBlock<48, 16>;

template <class GC_T, BTLA_ISA ISA_T>
using tWeiNInt = prologue_b::gemm::WeightKBlockNInteger<GC_T, ISA_T>;
template <class GC_T, BTLA_ISA ISA_T>
using tWeiNFloat = prologue_b::gemm::WeightKBlockNFloat<GC_T, ISA_T>;

template <class GC_T, BTLA_ISA ISA_T>
using tActKBaseF32 = prologue_a::gemm::ShuffleActivationKBlockBaseF32<GC_T, ISA_T>;

constexpr uint64_t Fp32Cores[] = {tAVX2::ID, tAVX512F::ID};
constexpr uint64_t Bf16Cores[] = {tAMX_BF16::ID};
constexpr uint64_t Fp16Cores[] = {tAVX512_FP16::ID};
constexpr uint64_t Int8Cores[] = {tAVX_VNNI::ID, tAVX512F::ID, tAVX512_VNNI::ID, tAMX_INT8_US::ID, tAMX_INT8_SS::ID};
constexpr uint64_t FloatCores[] = {tAVX2::ID, tAVX512F::ID, tAMX_BF16::ID, tAVX512_FP16::ID};
constexpr uint64_t AllKBlockCores[] = {tAVX2::ID,
                                       tAVX512F::ID,
                                       tAMX_BF16::ID,
                                       tAVX512_FP16::ID,
                                       tAVX_VNNI_KBlock::ID,
                                       tAVX512_VNNI_KBlock::ID,
                                       tAMX_INT8_US_KBlock::ID,
                                       tAMX_INT8_SS_KBlock::ID};

}  // namespace bestla
