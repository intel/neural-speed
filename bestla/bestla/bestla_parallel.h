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
#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#if BTLA_OPENMP
#include <omp.h>
#endif
#include "bestla_utils.h"
#include "bestla_device.h"

namespace bestla {
namespace parallel {

using thread_func = std::function<void(int tid)>;

class IThreading {
 public:
  explicit IThreading(bool supportPE) : isSupportPE(supportPE) {}

  // equal to "for(int i=begin1;i<end1;i+=step1)"
  void parallel_for_collapse(const int& begin1, const int& end1, const int& step1,
                             const std::function<void(int)>& func) {
    assert(end1 >= begin1 && step1 > 0);
    parallel_for([&](int tidx) {
      const int length = (end1 - begin1 + step1 - 1) / step1;
      const int block_size = length / mThreadNum;
      const int block_remain = length % mThreadNum;
      const int offset = std::min(tidx, block_remain) + tidx * block_size;
      for (int i = begin1 + offset * step1; i < begin1 + (offset + block_size + (tidx < block_remain)) * step1;
           i += step1)
        func(i);
    });
  }

  // equal to "for(int i=begin1;i<end1;i+=step1)for(int j=begin2;i<end2;i+=step2)"
  void parallel_for_collapse(const int& begin1, const int& end1, const int& step1, const int& begin2, const int& end2,
                             const int& step2, const std::function<void(int, int)>& func) {
    assert(end1 >= begin1 && step1 > 0 && end2 >= begin2 && step2 > 0);
    parallel_for([&](int tidx) {
      const int length1 = (end1 - begin1 + step1 - 1) / step1;
      const int length2 = (end2 - begin2 + step2 - 1) / step2;
      const int block_size = (length1 * length2) / mThreadNum;
      const int block_remain = (length1 * length2) % mThreadNum;
      const int offset = std::min(tidx, block_remain) + tidx * block_size;
      for (int i = offset; i < offset + block_size + (tidx < block_remain); ++i)
        func(begin1 + (i / length2) * step1, begin2 + (i % length2) * step2);
    });
  }
  virtual void parallel_for(const thread_func& func) = 0;
  virtual inline void sync(int tidx, int idx = 0) = 0;
  virtual int num_threads() const { return mThreadNum; };
  virtual int is_support_PE() const { return isSupportPE; };
  virtual void set_threads(int nthreads) = 0;
  virtual std::pair<float, float> get_PEtime() const { return {0.0f, 0.0f}; };

 protected:
  int mThreadNum = 0;
  const bool isSupportPE = false;
};

#if BTLA_OPENMP
class OMPThreading : public IThreading {
 public:
  explicit OMPThreading() : IThreading(false) {}

  void parallel_for(const thread_func& func) override {
    if (mThreadNum > 1) {
#pragma omp parallel
      {
        int tidx = omp_get_thread_num();
        func(tidx);
        (void)(tidx);
        (void)(0);
      }
    } else {
      func(0);
      void(0);
    }
  }
  virtual void set_threads(int nthreads) override {
    mThreadNum = nthreads;
    omp_set_num_threads(nthreads);
  }
  virtual inline void sync(int tidx, int idx = 0) override {
    (void)(tidx);
    (void)(idx);
#pragma omp barrier
    (void)(0);  // make msvc happy with c++20
  }
};
#endif

class StdThreading : public IThreading {
 public:
  using Timer_T = utils::timer<utils::microseconds>;
  explicit StdThreading() : IThreading(true) {
    cr = nullptr;
    memset(func_, 0, sizeof(func_));
    memset(flag, 0, sizeof(flag));
    stop = true;
    time_per_p = -1.f;
    time_per_e = -1.f;
  }

  void parallel_for(const thread_func& func) override {
    time_per_p = 0;
    time_per_e = 0;
    Timer_T tm;
    if (mThreadNum > 1) {
      running.store(mThreadNum - 1);
      for (int i = 0; i < 10; i++) flag[i].store(mThreadNum);
      if (cr->mHybrid) {
        int64_t time_p = 0, time_e = 0;

        for (size_t i = 0; i < mThreadNum - 1; i++) func_[i] = &func;
        thread_time[0] = 0;
        tm.start();
        func(0);
        thread_time[0] += int(tm.stop());
        while (true) {
          if (running.load() == 0)
            break;
          else
            _mm_pause();
        }
        for (int i = 0; i < mThreadNum; i++)
          if (i >= cr->P_core_num && i < cr->P_core_num + cr->E_core_num)
            time_e += thread_time[i];
          else
            time_p += thread_time[i];
        time_per_p = (time_p) / (1.0f * (mThreadNum - cr->E_core_num));
        time_per_e = (time_e) / (1.0f * cr->E_core_num);
        // printf("%d %d %f %f\n", time_p, time_e, time_per_p, time_per_e);
      } else {
        for (size_t i = 0; i < mThreadNum - 1; i++) {
          func_[i] = &func;
        }
        func(0);
        while (true) {
          if (running.load() == 0)
            break;
          else
            _mm_pause();
        }
      }
    } else {
      func(0);
    }
  }

  void set_threads(int nthreads) override {
    if (nthreads != mThreadNum) {
      stop_threads();
      mThreadNum = nthreads;
      cr = &device::CpuRuntime::getInstance(nthreads);
      create_threads();
    }
  }

  inline void sync(int tidx, int idx = 0) override {
    if (mThreadNum > 1) {
      flag[idx]--;
      if (cr->mHybrid) {
        Timer_T tm;
        tm.start();
        while (true) {
          if (flag[idx].load() == 0)
            break;
          else
            _mm_pause();
        }
        thread_time[tidx] -= int(tm.stop());
      } else {
        while (true) {
          if (flag[idx].load() == 0)
            break;
          else
            _mm_pause();
        }
      }
    }
  }

  std::pair<float, float> get_PEtime() const override { return {time_per_p, time_per_e}; };

  ~StdThreading() { stop_threads(); }

 private:
  void stop_threads() {
    stop = 1;
    for (int i = 0; i < mThreadNum - 1; i++) thdset[i].join();
    thdset.clear();
  }
  void create_threads() {
    thdset.resize(mThreadNum - 1);
    stop = 0;
    GetCPUDevice();
    std::vector<int> core_order;
    if (_cd->isHybrid()) {
      core_order.resize(_cd->getThreads());
      memcpy(reinterpret_cast<void*>(core_order.data()), reinterpret_cast<void*>(_cd->getPCores()),
             _cd->getPcoreNum() * sizeof(int));
      memcpy(reinterpret_cast<void*>(core_order.data() + _cd->getPcoreNum()), reinterpret_cast<void*>(_cd->getECores()),
             _cd->getEcoreNum() * sizeof(int));
      memcpy(reinterpret_cast<void*>(core_order.data() + _cd->getPcoreNum() + _cd->getEcoreNum()),
             reinterpret_cast<void*>(_cd->getSMTCores()), _cd->getSMTcoreNum() * sizeof(int));
    } else {
      core_order.resize(_cd->getCores() * 2);  // *2 for SMT
      if (_cd->isClient()) {
        for (int i = 0; i < _cd->getCores(); i++) {
          core_order[i] = 2 * i;
          core_order[i + _cd->getCores()] = 2 * i + 1;
        }
      } else {
        for (int i = 0; i < _cd->getCores() * 2; i++) core_order[i] = i;
      }
    }
    _cd->core_bond(core_order[0]);
    if (cr->mHybrid) {
      thread_time.resize(mThreadNum);
      for (size_t i = 0; i < mThreadNum - 1; i++) {
        thdset[i] = std::thread(
            [&](int tidx, int core_id) {
              _cd->core_bond(core_id);
              Timer_T tm;
              while (true) {
                if (stop) break;
                if (func_[tidx] != nullptr) {
                  thread_time[tidx + 1] = 0;
                  tm.start();
                  (*func_[tidx])(tidx + 1);
                  func_[tidx] = nullptr;
                  thread_time[tidx + 1] += int(tm.stop());
                  running.fetch_sub(1);
                } else {
                  _mm_pause();
                }
              }
            },
            int(i), core_order[i + 1]);
      }
    } else
      for (size_t i = 0; i < mThreadNum - 1; i++) {
        thdset[i] = std::thread(
            [&](int tidx, int core_id) {
              _cd->core_bond(core_id);
              while (true) {
                if (stop) break;
                if (func_[tidx] != nullptr) {
                  (*func_[tidx])(tidx + 1);
                  func_[tidx] = nullptr;
                  running--;
                } else {
                  _mm_pause();
                }
              }
            },
            int(i), core_order[i + 1]);
      }
  }
  device::CpuRuntime* cr;
  std::atomic_int64_t running;
  std::atomic_int64_t flag[10];
  const thread_func* func_[100];
  bool stop;
  std::vector<std::thread> thdset;
  std::vector<int64_t> thread_time;
  float time_per_p, time_per_e;
};

class SingleThread : public IThreading {
 public:
  SingleThread() : IThreading(false) { mThreadNum = 1; }

  void set_threads(int nthreads) override {
    assert(0);
    (void)(nthreads);
  }

  inline void parallel_for(const thread_func& func) override { func(0); }

  inline void sync(int tidx, int idx = 0) override {
    (void)(tidx);
    (void)(idx);
  }
};

struct Config2D {
  int threads;
  int size[2];
  int step[2];
  int offset[2];
};
struct ThreadProblem2D {
  int tid;
  int tidx[2];
  int loc[2];
  int size[2];
  bool valid;
  void print() {
    printf("Thread %d indice:(%d,%d)\n", tid, tidx[0], tidx[1]);
    printf("Thread location:(%d,%d)\n", loc[0], loc[1]);
    printf("Thread problem size:(%d,%d)\n", size[0], size[1]);
  }
};
class Scheduler2D {
 public:
  Scheduler2D() = default;
  Scheduler2D(const Config2D& config) { update(config); }
  using ThreadProblem = ThreadProblem2D;

  virtual void getIndex(ThreadProblem& problem) const {
    if (problem.tid >= mThdValid) {
      problem.size[0] = 0;
      problem.size[1] = 0;
      problem.valid = false;
      return;
    }
    auto& tid = problem.tid;
    problem.tidx[1] = tid % mThdPerRow;
    problem.tidx[0] = tid / mThdPerRow;
    problem.loc[0] = problem.tidx[0] * mThdSize[0];
    problem.loc[1] = problem.tidx[1] * mThdSize[1];
    problem.size[0] = utils::remainsize(problem.loc[0], mSize[0], mThdSize[0]);
    problem.size[1] = utils::remainsize(problem.loc[1], mSize[1], mThdSize[1]);
    problem.loc[0] += moffset[0];
    problem.loc[1] += moffset[1];
    problem.valid = true;
  }

  virtual void update(const Config2D& config) {
    mThdCount = config.threads;
    for (size_t i = 0; i < 2; i++) {
      mSize[i] = config.size[i];
      mStep[i] = config.step[i];
      moffset[i] = config.offset[i];
    }
    schedule();
  }

  constexpr static BTLA_ISA gemm_ISA() { return BTLA_ISA::NoSIMD; }

  void print() {
    printf("Thread Block:(%d,%d)\n", mThdSize[0], mThdSize[1]);
    printf("Thread in use:%d of %d, Nx%d\n", mThdValid, mThdCount, mThdPerRow);
  }

  constexpr int* thread_size() { return mThdSize; }

 protected:
  void set(const int* thdsize, const int* size, const int* step) {
    for (size_t i = 0; i < 2; i++) {
      mThdSize[i] = thdsize[i];
      mSize[i] = size[i];
      mStep[i] = step[i];
    }
  }
  void schedule() {
    int rownum = utils::updiv(mSize[0], mStep[0]);
    int colnum = utils::updiv(mSize[1], mStep[1]);
    float ratio = colnum * rownum / static_cast<float>(mThdCount);
    if (ratio <= 1) {
      mThdSize[0] = mStep[0];
      mThdSize[1] = mStep[1];
      mThdPerRow = colnum;
      calc_valid_threads();
      return;
    }
    float colratio = ratio > colnum ? colnum : ceil(ratio);
    mThdSize[1] = static_cast<int>(colratio * mStep[1]);
    mThdPerRow = static_cast<int>(ceil(static_cast<float>(colnum) / colratio));
    mThdSize[0] = static_cast<int>(ceil(rownum / (static_cast<float>(mThdCount) / mThdPerRow)) * mStep[0]);
    calc_valid_threads();
  }
  void calc_valid_threads() {
    mThdValid = mThdPerRow * static_cast<int>(std::ceil(static_cast<float>(mSize[0]) / mThdSize[0]));
  }

  int mThdPerRow = 0;
  int mThdValid = 0;
  int mThdCount = 0;
  int moffset[2] = {0, 0};

 private:
  int mThdSize[2] = {0, 0};
  int mSize[2] = {0, 0};
  int mStep[2] = {0, 0};
};

namespace gemm {

struct Config {
  const int threads;
  const utils::GemmProblem problem;
  const int offset[2];
  const size_t l2cache = 1024ULL * 1024;
  const size_t l1cache = 32ULL * 1024;
};

struct ThreadProblemBase : ThreadProblem2D {
  int block[3];
  size_t stacksize;
  size_t tmpcachesize;
};

template <class _GemmCore_T>
class SchedulerBase : public Scheduler2D {
 public:
  using ThreadProblem = ThreadProblemBase;
  SchedulerBase() = default;
  SchedulerBase(const Config& config) { update(config); }
  virtual void getIndex(ThreadProblem& problem) {
    problem.tmpcachesize = mL2Size - mL2Use;
    problem.stacksize = mL2Size;
    problem.block[0] = mBlock[0];
    problem.block[1] = mBlock[1];
    problem.block[2] = mBlock[2];
    Scheduler2D::getIndex(problem);
  }

  virtual void update(const Config& config) {
    for (size_t i = 0; i < 3; i++) {
      mSize[i] = config.problem.dims[i + 1];  // skip 0th batch
      mSizePadded[i] = utils::padto(mSize[i], mStep[i]);
    }
    mThdCount = config.threads;
    mL2Size = config.l2cache;
    mL1Size = config.l1cache;
    Scheduler2D::moffset[0] = config.offset[0];
    Scheduler2D::moffset[1] = config.offset[1];
    if (mSize[0] <= 0 || mSize[1] <= 0 || mSize[2] <= 0) {
      return;
    }
    schedule();
    assert(this->mL2Use <= this->mL2Size - ReservedSize);
    assert(this->mBlock[0] > 0);
    assert(this->mBlock[1] > 0);
    assert(this->mBlock[2] > 0);
  }

  constexpr static BTLA_ISA gemm_ISA() { return _GemmCore_T::ISA; }

  constexpr int valid_theads() { return mThdValid; }

  virtual void print() {
    printf("Thread Block:(%d,%d)\n", mThdSize[0], mThdSize[1]);
    printf("Thread in use:%d of %d, Nx%d\n", mThdValid, mThdCount, mThdPerRow);
    printf("GEMM MStep:%d NStep:%d KStep:%d\n", mBlock[0], mBlock[1], mBlock[2]);
    printf("Cache Size:%zu used:%zu\n", mL2Size, mL2Use);
  }

  template <class T>
  friend class SchedulerDispatcher;

 protected:
  virtual void schedule() {
    int rownum = utils::updiv(mSize[0], mStep[0]);
    int colnum = utils::updiv(mSize[1], mStep[1]);
    mDensity = static_cast<float>(mSize[0]) * mSize[1] / (mSize[0] + mSize[1]);
    int maxN = 0;
    float maxScore = std::numeric_limits<float>::min();
    int core_enum = static_cast<int>(std::sqrt(mThdCount));
    for (int i = 1; i <= core_enum; i += 1) {
      generate_by_cores(i, mThdCount / i, rownum, colnum);
      auto thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = i;
      }
      generate_by_cores(mThdCount / i, i, rownum, colnum);
      thdscore = calculate_score();
      if (maxScore < thdscore) {
        maxScore = thdscore;
        maxN = mThdCount / i;
      }
    }
    generate_by_cores(maxN, mThdCount / maxN, rownum, colnum);
    update_cache_blocking();
    Scheduler2D::set(mThdSize, mSize, mStep);
    mL2Use = static_cast<size_t>(mBlock[0]) * mBlock[1] * mEleSize[2];
    mL2Use += static_cast<size_t>(mBlock[1]) * mBlock[2] * mEleSize[1] * 2;
    mL2Use += static_cast<size_t>(mStep[0]) * mBlock[2] * mEleSize[0] * 2;
  }
  static float constexpr DensityThres = 16;
  static size_t constexpr ReservedSize = 32ULL * 1024ULL;

  virtual float calculate_score() {
    int tmpnstep = mThdSize[1] < _GemmCore_T::PREFERRED_N ? mThdSize[1] : _GemmCore_T::PREFERRED_N;
    float threadratio = static_cast<float>(mThdValid) / mThdCount;
    float density = static_cast<float>(tmpnstep) * mThdSize[0] / (tmpnstep + mThdSize[0]);
    if (mDensity < DensityThres) {
      return threadratio;
    }
    return (threadratio * 1.f + density * 0.0016f);
  }

  virtual void generate_by_cores(int ny, int nx, int rownum, int colnum) {
    mThdSize[0] = utils::updiv(rownum, ny) * mStep[0];
    mThdSize[1] = utils::updiv(colnum, nx) * mStep[1];
    mThdPerRow = utils::updiv(mSize[1], mThdSize[1]);
    mThdValid = utils::updiv(mSize[0], mThdSize[0]) * mThdPerRow;
  }

  // cache = mMStep * mNStep * CSize + mNStep * mKStep * BSize
  //       = mNStep * (mMStep*CSize + mKStep*BSize)
  // C Access = K/mKStep
  // B Access = M/mMStep
  // A Access = N/mNStep
  virtual void update_cache_blocking() {
    if (mDensity <= DensityThres) {
      return cache_blocking_memory();
    } else {
      return cache_blocking_compute();
    }
  }

  virtual void cache_blocking_compute() {
    size_t constexpr KRef = 256;
    size_t valid_total = mL2Size - ReservedSize;
    size_t asize = KRef * mStep[0] * mEleSize[0] * 2;
    size_t bsize = _GemmCore_T::PREFERRED_N * KRef * mEleSize[1] * 2;
    int maxM = static_cast<int>((valid_total - bsize - asize) / (_GemmCore_T::PREFERRED_N * mEleSize[2]));
    maxM = utils::downdiv(maxM, mStep[0]);
    int nthdm = mThdSize[0] / mStep[0];
    if (maxM < nthdm) {
      int niter = utils::updiv(nthdm, maxM);
      mBlock[0] = utils::updiv(nthdm, niter) * mStep[0];
    } else {
      mBlock[0] = mThdSize[0];
    }
    int maxN = static_cast<int>((valid_total - asize) / (mBlock[0] * mEleSize[2] + KRef * mEleSize[1] * 2));
    maxN = utils::downdiv(maxN, mStep[1]);
    int nthdn = mThdSize[1] / mStep[1];
    if (maxN < nthdn) {
      int niter = utils::updiv(nthdn, maxN);
      mBlock[1] = utils::updiv(nthdn, niter) * mStep[1];
    } else {
      mBlock[1] = mThdSize[1];
    }
    bsize = KRef * mBlock[1] * mEleSize[1] * 2;
    size_t csize = static_cast<size_t>(mBlock[0]) * mBlock[1] * mEleSize[2];
    auto rawk = static_cast<int>((valid_total - csize) / (mStep[0] * mEleSize[0] + mBlock[1] * mEleSize[1]) / 2);
    rawk = std::min(rawk, mSizePadded[2]);
    mBlock[2] = utils::padto_le(rawk, mStep[2]);
  }

  virtual void cache_blocking_memory() {
    mBlock[0] = mThdSize[0];
    mBlock[1] = mStep[1];
    size_t reservsize = static_cast<size_t>(mBlock[0]) * mBlock[1] * mEleSize[2];
    size_t maxK = (mL1Size - reservsize) / (mBlock[1] * mEleSize[1] + mBlock[0] * mEleSize[0]);
    size_t Bsize = maxK * mBlock[1] * mEleSize[1];
    size_t Bsize_1K = utils::padto_le(Bsize, 1024);
    mBlock[2] = static_cast<int>(Bsize_1K / mEleSize[1] / mBlock[1]);
    mBlock[2] = utils::padto_le(mBlock[2], mStep[2]);
  }

  size_t mL2Size = 0, mL1Size = 0, mL2Use = 0;
  float mDensity = 0.f;
  int mSize[3] = {0, 0, 0};
  int mThdSize[3] = {0, 0, 0};
  static constexpr int mStep[3] = {_GemmCore_T::MTILE, _GemmCore_T::NTILE, _GemmCore_T::KTILE};
  static constexpr int mEleSize[3] = {sizeof(typename _GemmCore_T::AType), sizeof(typename _GemmCore_T::BType),
                                      sizeof(typename _GemmCore_T::CType)};
  int mSizePadded[3] = {0, 0, 0};
  int mBlock[3] = {0, 0, 0};
};

template <class _GemmCore_T>
class SchedulerKBlockS : public SchedulerBase<_GemmCore_T> {
  // Block[2]: block size of K must be multiplier of mKBlock
  //           or factor of mKBlock
 public:
  using ThreadProblem = ThreadProblemBase;
  using BaseScheduler = SchedulerBase<_GemmCore_T>;
  SchedulerKBlockS() = default;
  SchedulerKBlockS(const Config& config) { update(config); }

  void update(const Config& config) {
    mKBlock = config.problem.dims[4];
    BaseScheduler::update(config);
    auto blks = utils::updiv(this->mBlock[2], mKBlock);
    this->mL2Use += static_cast<size_t>(blks) * (this->mBlock[1] + this->mStep[0]) *
                    (sizeof(float) + sizeof(int8_t) + sizeof(float));  // scale+zp+reduce
    assert(this->mL2Use <= this->mL2Size - ReservedSize);
    assert(this->mBlock[0] >= 0);
    assert(this->mBlock[1] >= 0);
    assert(this->mBlock[2] >= 0);
    assert(this->mBlock[2] % _GemmCore_T::KTILE == 0);
  }

  constexpr static BTLA_ISA gemm_ISA() { return _GemmCore_T::ISA; }

  template <class T>
  friend class SchedulerDispatcher;

 protected:
  static float constexpr DensityThres = 16;
  static size_t constexpr ReservedSize = 32ULL * 1024ULL;

  void cache_blocking_compute() override {
    size_t constexpr KRef = 256;
    size_t constexpr CorSize = sizeof(float) + sizeof(int8_t) + sizeof(float);
    size_t valid_total = this->mL2Size - ReservedSize;
    auto blks = utils::updiv(KRef, this->mKBlock);
    auto asize = KRef * this->mStep[0] * this->mEleSize[0] + blks * this->mStep[0] * CorSize;
    asize *= 2;
    auto bsize = _GemmCore_T::PREFERRED_N * KRef * this->mEleSize[1] + _GemmCore_T::PREFERRED_N * blks * CorSize;
    asize *= 2;
    size_t csize_total = valid_total - asize - bsize;
    int maxM = static_cast<int>(csize_total / _GemmCore_T::PREFERRED_N / this->mEleSize[2]);
    maxM = utils::downdiv(maxM, this->mStep[0]);
    int nthdm = this->mThdSize[0] / this->mStep[0];
    if (maxM < nthdm) {
      int niter = utils::updiv(nthdm, maxM);
      this->mBlock[0] = utils::updiv(nthdm, niter) * this->mStep[0];
    } else {
      this->mBlock[0] = this->mThdSize[0];
    }
    int maxN = static_cast<int>((valid_total - asize) / (this->mBlock[0] * this->mEleSize[2] +
                                                         (KRef * this->mEleSize[1] + blks * CorSize) * 2));
    maxN = utils::downdiv(maxN, this->mStep[1]);
    int nthdn = this->mThdSize[1] / this->mStep[1];
    if (maxN < nthdn) {
      int niter = utils::updiv(nthdn, maxN);
      this->mBlock[1] = utils::updiv(nthdn, niter) * this->mStep[1];
    } else {
      this->mBlock[1] = this->mThdSize[1];
    }
    size_t csize = static_cast<size_t>(this->mBlock[0]) * this->mBlock[1] * this->mEleSize[2];
    auto rawk = static_cast<int>((valid_total - csize) / 2 /
                                 (this->mStep[0] * this->mEleSize[0] +
                                  float(CorSize * (this->mStep[0] + this->mBlock[1])) / this->mKBlock +
                                  this->mBlock[1] * this->mEleSize[1]));
    if (rawk < this->mKBlock) {
      rawk = static_cast<int>((valid_total - csize - 1 * CorSize * (this->mStep[0] + this->mBlock[1])) / 2 /
                              (this->mStep[0] * this->mEleSize[0] + this->mBlock[1] * this->mEleSize[1]));
    }
    rawk = std::min(rawk, this->mSizePadded[2]);
    this->mBlock[2] = utils::padto_le(rawk, this->mStep[2]);
    if (this->mBlock[2] > this->mKBlock) {
      this->mBlock[2] = utils::padto_le(this->mBlock[2], this->mKBlock);
    }
  }

  void cache_blocking_memory() override {
    this->mBlock[0] = _GemmCore_T::MTILE;
    size_t startK = std::max(16, _GemmCore_T::KTILE);
    auto getMaxN = [&](size_t refk) {
      size_t sizeA = refk * this->mEleSize[0] * this->mBlock[0];
      auto blks = utils::updiv(refk, mKBlock);
      sizeA += blks * this->mBlock[0] * (sizeof(float) + sizeof(uint8_t));
      size_t maxN = (this->mL1Size - sizeA) / (this->mBlock[0] * this->mEleSize[2] + refk * this->mEleSize[1]);
      return maxN;
    };
    auto getMaxK = [&](size_t refN) {
      size_t sizeC = refN * this->mEleSize[2] * this->mBlock[0];
      size_t maxK = (this->mL1Size - sizeC) / (this->mBlock[0] * this->mEleSize[0] + refN * this->mEleSize[1]);
      return maxK;
    };
    if (mKBlock <= 32) {
      this->mBlock[2] = mKBlock;
      auto maxN = getMaxN(startK);
      this->mBlock[1] = static_cast<int>(maxN);
      this->mBlock[1] = std::min(this->mBlock[1], this->mThdSize[1]);
      this->mBlock[1] = utils::padto_le(this->mBlock[1], this->mStep[1]);
      return;
    }
    auto maxN = getMaxN(startK);
    if (maxN <= this->mThdSize[1]) {
      this->mBlock[1] = static_cast<int>(maxN);
      this->mBlock[1] = utils::padto_le(this->mBlock[1], this->mStep[1]);
      this->mBlock[2] = static_cast<int>(startK);
    } else {
      this->mBlock[1] = this->mThdSize[1];
      this->mBlock[2] = static_cast<int>(getMaxK(this->mBlock[1]));
      this->mBlock[2] = utils::padto_le(this->mBlock[2], this->mStep[2]);
      this->mBlock[2] = std::min(mKBlock, this->mBlock[2]);
    }
  }

  int mKBlock{0};
};

template <class Scheduler>
class SchedulerDispatcher {
 public:
  using ThreadProblem = ThreadProblemBase;
  SchedulerDispatcher() = default;
  ~SchedulerDispatcher() {
    std::pair<float, float> PEtime = th_->get_PEtime();
    if (needDispatch && int(PEtime.first) > 0 && int(PEtime.second) > 0)
      cr->adjustPE(Scheduler::gemm_ISA(), PEtime.second / PEtime.first);
  }
  SchedulerDispatcher(const IThreading* th, const utils::GemmProblem& problem) {
    th_ = th;
    cr = &device::CpuRuntime::getInstance(th->num_threads());
    needDispatch = cr->mHybrid && th->is_support_PE();
    if (!needDispatch) {
      Scheduler_P = std::move(Scheduler({th->num_threads(), problem, {0, 0}, cr->mL2Cache, cr->mL1Cache}));
    } else {
      Pcore_num = cr->P_core_num;
      Ecore_num = cr->E_core_num;
      utils::GemmProblem problem_P = problem, problem_E = problem;
      const int N = problem.dims[2];
      auto PE_Ratio = cr->getPE(Scheduler::gemm_ISA());
      int N_offset = utils::padto(N - int(N / (1 + PE_Ratio)), Scheduler::mStep[1]);
      N_offset = N_offset <= N ? N_offset : N;
      problem_P.dims[2] = N_offset;
      Scheduler_P =
          std::move(Scheduler({th->num_threads() - cr->E_core_num, problem_P, {0, 0}, cr->mL2Cache_P, cr->mL1Cache_P}));
      problem_E.dims[2] = N - N_offset;
      Scheduler_E = std::move(Scheduler({cr->E_core_num, problem_E, {0, N_offset}, cr->mL2Cache_E, cr->mL1Cache_E}));
    }
  }

  void getIndex(ThreadProblem& problem) {
    if (!needDispatch) {
      Scheduler_P.getIndex(problem);
    } else {
      if (problem.tid >= Pcore_num + Ecore_num) {
        problem.tid -= Ecore_num;
        Scheduler_P.getIndex(problem);
      } else if (problem.tid >= Pcore_num) {
        problem.tid -= Pcore_num;
        Scheduler_E.getIndex(problem);
      } else {
        Scheduler_P.getIndex(problem);
      }
    }
  }

  void print() {
    printf("dispatch to hybrid:%d\n", needDispatch);
    Scheduler_P.print();
    if (needDispatch) Scheduler_E.print();
  }

 private:
  Scheduler Scheduler_P, Scheduler_E;
  const IThreading* th_;
  device::CpuRuntime* cr;
  bool needDispatch = false;
  int Pcore_num = 0, Ecore_num = 0;
};

template <>
class SchedulerDispatcher<Scheduler2D> {
 public:
  using ThreadProblem = ThreadProblem2D;
  SchedulerDispatcher() = default;
  ~SchedulerDispatcher() {}
  SchedulerDispatcher(const IThreading* th, const Config2D& config) {
    device::CpuRuntime& cr = device::CpuRuntime::getInstance(config.threads);
    needDispatch = cr.mHybrid && th->is_support_PE();
    if (!needDispatch) {
      Scheduler_P = std::move(Scheduler2D(config));
    } else {
      Pcore_num = cr.P_core_num;
      Ecore_num = cr.E_core_num;
      Config2D config_P = config, config_E = config;
      const int N = config.size[1];
      const auto pe = cr.getPE(BTLA_ISA::NoSIMD);
      const int N_offset = utils::padto(N - int(N / (1 + pe)), config.step[1]);
      config_P.threads = config.threads - cr.E_core_num;
      config_P.size[1] = N_offset;
      Scheduler_P = std::move(Scheduler2D(config_P));
      config_E.threads = cr.E_core_num;
      config_E.size[1] = N - N_offset;
      config_E.offset[1] += N_offset;
      Scheduler_E = std::move(Scheduler2D(config_E));
    }
  }

  void getIndex(ThreadProblem& problem) {
    if (!needDispatch) {
      Scheduler_P.getIndex(problem);
    } else {
      if (problem.tid >= Pcore_num + Ecore_num) {
        problem.tid -= Ecore_num;
        Scheduler_P.getIndex(problem);
      } else if (problem.tid >= Pcore_num) {
        problem.tid -= Pcore_num;
        Scheduler_E.getIndex(problem);
      } else {
        Scheduler_P.getIndex(problem);
      }
    }
  }

  void print() {
    printf("dispatch to hybrid:%d\n", needDispatch);
    Scheduler_P.print();
    if (needDispatch) Scheduler_E.print();
  }

 private:
  Scheduler2D Scheduler_P, Scheduler_E;
  bool needDispatch = false;
  int Pcore_num = 0, Ecore_num = 0;
};

}  // namespace gemm

template <class Parallel_T, class Launch_T>
void GemmRun(Launch_T& launcher, const typename Launch_T::Param& args, parallel::IThreading* th) {
  gemm::SchedulerDispatcher<Parallel_T> para(th, args.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename Parallel_T::ThreadProblem thdp{tidx};
    para.getIndex(thdp);
    if (thdp.valid) {
      launcher.run(args, thdp);
    }
  });
}

template <class Parallel_T, class Launch_T>
void GemmRunWithA(Launch_T& launcher, const typename Launch_T::Param& args, parallel::IThreading* th) {
  gemm::SchedulerDispatcher<Parallel_T> para(th, args.problem);
  using AParall = typename Launch_T::PrologueA::Parallel;
  AParall apara = Launch_T::PrologueA::createParallel(th->num_threads(), args.problem);
  static bool flag = false;
  if (flag) {
    printf("%s\n", __FUNCTION__);
    para.print();
    flag = false;
  }
  th->parallel_for([&](int tidx) {
    typename AParall::ThreadProblem thdpA{tidx};
    apara.getIndex(thdpA);
    if (thdpA.valid) {
      Launch_T::PrologueA::run(args.paramA, thdpA);
    }
    th->sync(tidx);
    typename Parallel_T::ThreadProblem thdp{tidx};
    para.getIndex(thdp);
    if (thdp.valid) {
      launcher.run(args, thdp);
    }
  });
}

}  // namespace parallel
}  // namespace bestla
