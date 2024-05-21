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
#include "sycl_utils.h"
#include "bestla/bestla_storage.h"

namespace bestla {
namespace sycl_storage {
class StorageWeightKBlockNInteger {
 public:
  BTLA_PROLOGUEB_IDS mPrologueID = BTLA_PROLOGUEB_IDS::Undef;
  uint64_t mCoreId = 0;
  BTLA_DTYPE mDType = BTLA_DTYPE::F32;
  BTLA_DTYPE mScaT = BTLA_DTYPE::F32, mZpT = BTLA_DTYPE::F32, mRedT = BTLA_DTYPE::F32;
  int mNPad = 0, mKPad = 0;
  int mN = 0, mK = 0;
  int mBlockSize = 1;
  int mDqBlockSize = 0;
  size_t mCSize = 0;
  int mCStep = 0;
  sycl_utils::sycl_vector<int8_t> mQBuf;
  sycl_utils::sycl_vector<int8_t> mScaleBuf;
  sycl_utils::sycl_vector<int8_t> mZpBuf, mRedBuf;
  sycl_utils::sycl_vector<int8_t> mDQCorrectionBuf;
  sycl_utils::sycl_vector<int8_t> mShuffleIndices;

  StorageWeightKBlockNInteger(bestla::storage::gemm::StorageWeightKBlockNInteger& _hoststor, sycl::queue* queue) {
    mPrologueID = BTLA_PROLOGUEB_IDS::WeightKBlockNInteger;
    mCoreId = 0;
    mDType = _hoststor.mDType;
    mScaT = _hoststor.SDtype();
    mZpT = _hoststor.ZDtype();
    mRedT = _hoststor.RDtype();
    mNPad = _hoststor.mNPad;
    mKPad = _hoststor.mKPad;
    mN = _hoststor.mN;
    mK = _hoststor.mK;
    mBlockSize = _hoststor.mBlockSize;
    mDqBlockSize = _hoststor.mDqBlockSize;
    mCSize = _hoststor.CSize();
    mCStep = _hoststor.CStep();
    if (_hoststor.template WPtr<void>()) {
      mQBuf.resize(_hoststor.template WSize<int8_t>(), queue);
      queue->memcpy(mQBuf.data(), _hoststor.template WPtr<void>(), mQBuf.size()).wait();
    }
    size_t csize = _hoststor.CSize();
    if (_hoststor.template SPtr<void>()) {
      mScaleBuf.resize(csize * _hoststor.mCorrection.mScaEleSize, queue);
      queue->memcpy(mScaleBuf.data(), _hoststor.template SPtr<void>(), mScaleBuf.size()).wait();
    }
    if (_hoststor.template ZPtr<void>()) {
      mZpBuf.resize(csize * _hoststor.mCorrection.mZpEleSize, queue);
      queue->memcpy(mZpBuf.data(), _hoststor.template ZPtr<void>(), mZpBuf.size()).wait();
    }
    if (_hoststor.template RPtr<void>()) {
      mRedBuf.resize(csize * _hoststor.mCorrection.mRedEleSize, queue);
      queue->memcpy(mRedBuf.data(), _hoststor.template RPtr<void>(), mRedBuf.size()).wait();
    }
    // TODO DQ,shuffle support
  }
  void toHost(bestla::storage::gemm::StorageWeightKBlockNInteger& _hoststor, sycl::queue* queue) {
    if (mQBuf.data()) {
      queue->memcpy(_hoststor.template WPtr<void>(), mQBuf.data(), mQBuf.size()).wait();
    }
    if (mScaleBuf.data()) {
      queue->memcpy(_hoststor.template SPtr<void>(), mScaleBuf.data(), mScaleBuf.size()).wait();
    }
    if (mZpBuf.data()) {
      queue->memcpy(_hoststor.template ZPtr<void>(), mZpBuf.data(), mZpBuf.size()).wait();
    }
    if (mRedBuf.data()) {
      queue->memcpy(_hoststor.template RPtr<void>(), mRedBuf.data(), mRedBuf.size()).wait();
    }
  }
};
}  // namespace sycl_storage
}  // namespace bestla
