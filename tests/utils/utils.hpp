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

#pragma once

#include "buff_compare.hpp"
#include "common.hpp"
#include "execution.hpp"
#include "gemm_gen.hpp"
#include "profiling.hpp"

#if defined(TEST_GPU_ARCH_XE_LPG)
inline constexpr gpu_arch TEST_GPU_ARCH = gpu_arch::XeLpg;
#elif defined(TEST_GPU_ARCH_XE_HPG)
inline constexpr gpu_arch TEST_GPU_ARCH = gpu_arch::XeHpg;
#elif defined(TEST_GPU_ARCH_XE_HPC)
inline constexpr gpu_arch TEST_GPU_ARCH = gpu_arch::XeHpc;
#else
static_assert(false, "TEST_GPU_ARCH not defined");
#endif
