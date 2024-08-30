/*******************************************************************************
 * Copyright (c) 2022-2023 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <gtest/gtest.h>
#include <utils/utils.hpp>
#include "common.hpp"
#include "kernel_func.hpp"

std::string esimd_compile_string =
    " -vc-codegen -doubleGRF "
    " -vc-disable-indvars-opt "
    " -Xfinalizer ' -printregusage -enableBCR -DPASTokenReduction ' ";

template <typename T>
class unaligned_gemm_test : public ::testing::Test {};
TYPED_TEST_SUITE_P(unaligned_gemm_test);
TYPED_TEST_P(unaligned_gemm_test, esimd) {
  gemm_exec<
      TypeParam,
      result_validate<TypeParam>,
      unaligned_gemm_func<TypeParam>>(esimd_compile_string);
}
REGISTER_TYPED_TEST_SUITE_P(unaligned_gemm_test, esimd);
using tests = ::testing::Types<
    Test0x,
    Test1x,
    Test2x,
    Test3x,
    Test4x,
    Test5x,
    Test6x,
    Test7x,
    Test8x,
    Test9x,
    Test10x,
    Test11x,
    Test12x,
    Test13x,
    Test14x,
    Test15x,
    Test16x,
    Test17x,
    Test18x,
    Test19f,
    Test19x,
    Test20x,
    Test20f,
    Test21x,
    Test21f,
    Test22x,
    Test22f,
    Test23x,
    Test23f>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    unaligned_gemm_test_suite,
    unaligned_gemm_test,
    tests);
