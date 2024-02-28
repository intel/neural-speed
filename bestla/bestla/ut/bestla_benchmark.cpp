#include <stdio.h>
#include "bestla_wrapper.h"
#include "bestla_ut.h"

namespace bestla {
using namespace utils;
namespace ut {
class Benchmark_Fp32Fp32 {
 public:
  Benchmark_Fp32Fp32() {
    UT_START();
    benchmark_all(1, 4096, 4096, 32);
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
  }

  using AType = float;
  using BType = float;
  using CType = float;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n ", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch, 0), RefC(m * n * batch, 0);
    fill_buffer_randn(A.data(), m * k, -0.5f, 0.5f);
    fill_buffer_randn(B.data(), n * k, -0.5f, 0.5f);
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;

    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AVX512F()) {
      benchmark<sAVX512F, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getThreads());
      benchmark<sAVX512F, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getCores());
    }
    if (_cd->AVX2()) {
      if (_cd->isHybrid()) {
        benchmark<sAVX2, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getThreads());
        benchmark<sAVX2, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getCores());
        benchmark<sAVX2, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getPcoreNum());
      } else {
        benchmark<sAVX2, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getThreads());
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Fp32Fp32 sBenchmark_Fp32Fp32;
#endif

class Benchmark_U8S8S32 {
 public:
  Benchmark_U8S8S32() {
    UT_START();
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
  }

  using AType = uint8_t;
  using BType = int8_t;
  using CType = int;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackInt32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      benchmark<gemm::ICoreRowNAmxint8<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                     _cd->getThreads());
      benchmark<gemm::ICoreRowNAmxint8<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                     _cd->getThreads());
      benchmark<gemm::ICoreRowNAmxint8<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                     _cd->getThreads());
    }
    if (_cd->AVX512_VNNI()) {
      benchmark<gemm::ICoreRowNAvx512vnni<48, 8>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                       _cd->getThreads());
    }
    if (_cd->AVX_VNNI()) {
      if (_cd->isHybrid()) {
        benchmark<gemm::ICoreRowNAvxvnni<48, 2>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getThreads());
        benchmark<gemm::ICoreRowNAvxvnni<48, 2>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getCores());
        benchmark<gemm::ICoreRowNAvxvnni<48, 2>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getPcoreNum());
        benchmark<gemm::ICoreRowNAvxvnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getThreads());
        benchmark<gemm::ICoreRowNAvxvnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getCores());
        benchmark<gemm::ICoreRowNAvxvnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getPcoreNum());
      } else {
        benchmark<gemm::ICoreRowNAvxvnni<48, 2>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getThreads());
        benchmark<gemm::ICoreRowNAvxvnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                      _cd->getThreads());
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_U8S8S32 sBenchmark_U8S8S32;
#endif

class Benchmark_S8S8S32 {
 public:
  Benchmark_S8S8S32() {
    UT_START();
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
  }

  using AType = int8_t;
  using BType = int8_t;
  using CType = int;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackInt32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(AType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_INT8()) {
      benchmark<gemm::ICoreRowNAmxint8SS<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                       _cd->getThreads());
      benchmark<gemm::ICoreRowNAmxint8SS<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                       _cd->getThreads());
      benchmark<gemm::ICoreRowNAmxint8SS<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                       _cd->getThreads());
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_S8S8S32 sBenchmark_S8S8S32;
#endif

class Benchmark_Bf16Bf16Fp32 {
 public:
  Benchmark_Bf16Bf16Fp32() {
    UT_START();
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
  }

  using AType = utils::bf16;
  using BType = utils::bf16;
  using CType = float;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        utils::GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, BType(-0.5f), BType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_BF16()) {
      request_perm_xtile_data();
      benchmark<gemm::HCoreRowNAmxbf16<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                     _cd->getThreads());
      benchmark<gemm::HCoreRowNAmxbf16<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                     _cd->getThreads());
      benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                     _cd->getThreads());
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Bf16Bf16Fp32 sBenchmark_Bf16Bf16Fp32;
#endif

class Benchmark_Fp16Fp16Fp16 {
 public:
  Benchmark_Fp16Fp16Fp16() {
    UT_START();
    benchmark_all(1024, 4096, 4096, 32);
    benchmark_all(2048, 4096, 4096, 32);
  }

  using AType = utils::fp16;
  using BType = utils::fp16;
  using CType = utils::fp16;
  template <typename Core_T, typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems, int threads) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, prologue_b::gemm::WeightPack,
                                    epilogue::gemm::AccumulatorWriteBackFp16>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, {B + i * n * k, n, &packBs[i]}, UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {0, 0, &packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
                 flops / threads);
        }
      }
    }
  }

  void benchmark_all(size_t m, size_t n, size_t k, size_t batch) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<AType> A(m * k * batch);
    avector<BType> B(k * n * batch);
    avector<CType> C(m * n * batch), RefC(m * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, AType(-0.5f), AType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AVX512_FP16()) {
      benchmark<sAVX512_FP16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getThreads());
      benchmark<gemm::HCoreRowNAvx512fp16<64, 12>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                        _cd->getThreads());
      benchmark<sAVX512_FP16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, _cd->getThreads());
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Fp16Fp16Fp16 sBenchmark_Fp16Fp16Fp16;
#endif

class UTBenchmark_CompFp32 {
 public:
  UTBenchmark_CompFp32() {
    UT_START();
    CheckISA(AVX512F);
    ut_s4();
    /*   ut_s8();
       ut_f4();*/
  }

  void ut_s4() {
    // benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(32, 4096, 4096, 128, BTLA_DTYPE::S3_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(32, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, 128, BTLA_DTYPE::S3_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2048, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(4096, 4096, 11008, 128, BTLA_DTYPE::S4_CLIP);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    //  benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32,
    //  BTLA_DTYPE::S4_FULLRANGE);
  }

  // void ut_s8() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  // }

  // void ut_f4() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  // }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int blocksize, int batch, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                 prologue_b::gemm::WeightKBlockNInteger<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);

    } else if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                        prologue_b::gemm::WeightKBlockNFloat<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    kernel.mProB.packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          double band = double(memsize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, corestr,
                 log.get_log_str(), flops, flops / threads, band);
        }
      }
    }
  }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_mem(int m, int n, int k, int blocksize, int batch, float* A, float* B, float* C, float timems,
                     int threads, BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerKBlock<Core_T>;
    // using Launcher =
    //     wrapper::gemm::LauncherKBlock<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, Wei,
    //                                   epilogue::gemm::CompFp32BlockEpilogue,
    //                                   epilogue::gemm::AccumulatorWriteBackFp32>;
    using Launcher =
        wrapper::gemm::LauncherIntKBlock<Core_T::ISA, Core_T, prologue_a::gemm::ActivationF32KBlockQuantize,
                                         prologue_b::gemm::WeightKBlockNInteger,
                                         epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                 prologue_b::gemm::WeightKBlockNInteger<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);

    } else if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                        prologue_b::gemm::WeightKBlockNFloat<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    auto quanA = kernel.mProA.createStorage(m, k, blocksize, false);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    kernel.mProB.packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      log.start();
      for (size_t i = 0; i < batch; i++) {
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{gp,
                                      {A + i * m * k, k, &quanA},
                                      {&packBs[i]},
                                      // {packBs[i].template SPtr<int8_t>(), packBs[i].SDtype(), packBs[i].CStep()},
                                      {C + i * m * n, n}};
        // parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        parallel::GemmRunWithA<Parallel>(kernel, args, UT_Threading::get());
      }
      if (log.stop()) {
        double t = log.avg_val / batch;
        double flops = double(psize) / t / 1e6;
        double band = double(memsize) / t / 1e6;
        printf("Threads %d %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, corestr, flops,
               flops / threads, band);
      }
    }
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(size_t m, size_t n, size_t k, size_t batch, BTLA_DTYPE qtype) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<float> A(m * k * batch);
    avector<float> B(k * n);
    avector<float> C(m * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (size_t i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AVX512F()) {
      // int blocksize = 32;
      // benchmark<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
      //                                                             C.data(), testtime, 48, qtype);
      // benchmark_mem<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
      //                                                                 C.data(), testtime, 48, qtype);
      int blocksize = 128;
      // benchmark<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
      //                                                             C.data(), testtime, 48, qtype);
      // benchmark_mem<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
      benchmark_mem<gemm::ICoreRowNAmxint8KBlock<48, 16>, LOG, Wei, Scale_T>(
          m, n, k, blocksize, batch, A.data(), B.data(), C.data(), testtime, _cd->getThreads(), qtype);
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UTBenchmark_CompFp32 sUTBenchmark_CompFp32;
#endif

class UTBenchmark_CompBf16 {
 public:
  UTBenchmark_CompBf16() {
    UT_START();
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    ut_s4();
    /*   ut_s8();
       ut_f4();*/
  }

  void ut_s4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(2048, 4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(4096, 4096, 11008, 128, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 32, BTLA_DTYPE::S4_FULLRANGE);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, 128, BTLA_DTYPE::S4_FULLRANGE);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, float>(2, 4096, 4096, -1, BTLA_DTYPE::S4_FULLRANGE);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
    // benchmark_all<prologue_b::gemm::WeightKBlockS4, utils::bf16>(2, 4096, 4096, 32,
    // BTLA_DTYPE::S4_FULLRANGE);
  }

  // void ut_s8() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, 128, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, float>(2, 4096, 4096, -1, BTLA_DTYPE::S8);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockS8, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::S8);
  // }

  // void ut_f4() {
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, float>(2, 4096, 4096, -1, BTLA_DTYPE::F4_NF4);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_BNB);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_E2M1);
  //   ut<sAMX_BF16, prologue_b::gemm::WeightKBlockF4, utils::bf16>(2, 4096, 4096, 32, BTLA_DTYPE::F4_NF4);
  // }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int blocksize, int batch, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerBase<Core_T>;
    using Launcher = wrapper::gemm::LauncherBase<Core_T::ISA, Core_T, prologue_a::gemm::ActivationConverterFp32, Wei,
                                                 epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB(0);
    if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                 prologue_b::gemm::WeightKBlockNInteger<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);
    } else if constexpr (std::is_same_v<Wei<Core_T, Core_T::ISA>,
                                        prologue_b::gemm::WeightKBlockNFloat<Core_T, Core_T::ISA>>) {
      tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>);
    }
    std::vector<WType> packBs(batch, 0);
    std::vector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
      kernel.mProB.packWeight(n, k, B + i * n * k, n, &packBs[i], UT_Threading::get());
    }
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        if (log.stop()) {
          double flops = double(psize) / log.avg_val / 1e6;
          double band = double(memsize) / log.avg_val / 1e6;
          printf("Threads %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, corestr,
                 log.get_log_str(), flops, flops / threads, band);
        }
      }
    }
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(size_t m, size_t n, size_t k, size_t batch, BTLA_DTYPE qtype) {
    printf("%s %d %d %d %d\n", __FUNCTION__, int(m), int(n), int(k), int(batch));
    avector<float> A(m * k * batch);
    avector<float> B(k * n * batch);
    avector<float> C(m * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<100>;
    float testtime = 500.f;
    GetCPUDevice();
    if (_cd->AMX_BF16()) {
      int blocksize = 32;
      benchmark<gemm::HCoreRowNAmxbf16<32, 32>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                   C.data(), testtime, _cd->getThreads(), qtype);
      benchmark<gemm::HCoreRowNAmxbf16<48, 16>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                   C.data(), testtime, _cd->getThreads(), qtype);
      benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG, Wei, Scale_T>(m, n, k, blocksize, batch, A.data(), B.data(),
                                                                   C.data(), testtime, _cd->getThreads(), qtype);
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UTBenchmark_CompBf16 sUTBenchmark_CompBf16;
#endif

}  // namespace ut
}  // namespace bestla
int main() {
  printf("BesTLA Benchmark done\n");
  return 0;
}
