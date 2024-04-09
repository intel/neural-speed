#include <stdio.h>
#include "bestla_wrapper.h"
#include "bestla_ut.h"
#undef BTLA_UT_WRAPPER
#undef BTLA_UT_PROLOGUE_B
namespace bestla {
using namespace utils;
namespace ut {
int constexpr TestMs = 500;
class Benchmark_Fp32Fp32 {
 public:
  Benchmark_Fp32Fp32() {
    UT_START();
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
    benchmark_all(2048, 4096, 4096);
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
    avector<int8_t> bufB(tmpB.mSize * batch);
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
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch, 0);
    fill_buffer_randn(A.data(), m * k, -0.5f, 0.5f);
    fill_buffer_randn(B.data(), n * k, -0.5f, 0.5f);
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AVX512F()) {
        benchmark<sAVX512F, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
      if (_cd->AVX2()) {
        benchmark<sAVX2, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
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
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
    benchmark_all(2048, 4096, 4096);
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
    static Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    auto tmpB = kernel.mProB.createStorage(n, k);
    std::vector<storage::gemm::StoragePackedWeight> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
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
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::U8, BTLA_DTYPE::S8, BTLA_DTYPE::S32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::U8), bestla_dtype_str(BTLA_DTYPE::S8),
           bestla_dtype_str(BTLA_DTYPE::S32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AMX_INT8()) {
        benchmark<gemm::ICoreRowNAmxint8<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::ICoreRowNAmxint8<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::ICoreRowNAmxint8<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
      if (_cd->AVX512_VNNI()) {
        benchmark<gemm::ICoreRowNAvx512vnni<64, 6>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
        benchmark<gemm::ICoreRowNAvx512vnni<48, 8>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
      }
      if (_cd->AVX_VNNI()) {
        benchmark<gemm::ICoreRowNAvxvnni<48, 2>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::ICoreRowNAvxvnni<24, 4>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
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
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
    benchmark_all(2048, 4096, 4096);
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
    avector<int8_t> bufB(tmpB.mSize * batch);
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
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::S8, BTLA_DTYPE::S8, BTLA_DTYPE::S32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::S8), bestla_dtype_str(BTLA_DTYPE::S8),
           bestla_dtype_str(BTLA_DTYPE::S32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), m * k, AType(0), AType(255));
    fill_buffer_randn(B.data(), k * n, BType(-127), BType(127));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(AType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AMX_INT8()) {
        benchmark<gemm::ICoreRowNAmxint8SS<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
        benchmark<gemm::ICoreRowNAmxint8SS<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
        benchmark<gemm::ICoreRowNAmxint8SS<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                         threads);
      }
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
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
    benchmark_all(2048, 4096, 4096);
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
    avector<int8_t> bufB(tmpB.mSize * batch);
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
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::BF16, BTLA_DTYPE::BF16, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::BF16),
           bestla_dtype_str(BTLA_DTYPE::BF16), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, BType(-0.5f), BType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AMX_BF16()) {
        benchmark<gemm::HCoreRowNAmxbf16<32, 32>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::HCoreRowNAmxbf16<48, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
      }
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
    benchmark_all(1, 4096, 4096);
    benchmark_all(1024, 4096, 4096);
    benchmark_all(2048, 4096, 4096);
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
    avector<int8_t> bufB(tmpB.mSize * batch);
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
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf("Threads %d %s %s Flops:%.3f PerCoreFlops:%.3f\n", threads, corestr, log.get_log_str(), flops,
           flops / threads);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F16, BTLA_DTYPE::F16, BTLA_DTYPE::F16);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F16),
           bestla_dtype_str(BTLA_DTYPE::F16), bestla_dtype_str(BTLA_DTYPE::F16));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, AType(-0.5f), AType(0.5f));
    fill_buffer_randn(B.data(), k * n, AType(-0.5f), AType(0.5f));
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      if (_cd->AVX512_FP16()) {
        benchmark<sAVX512_FP16, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime, threads);
        benchmark<gemm::HCoreRowNAvx512fp16<64, 12>, LOG>(m, n, k, batch, A.data(), B.data(), C.data(), testtime,
                                                          threads);
      }
    }
  }
};
#ifdef BTLA_UT_WRAPPER
static Benchmark_Fp16Fp16Fp16 sBenchmark_Fp16Fp16Fp16;
#endif

class UTWOQ_CompFp32 {
 public:
  UTWOQ_CompFp32() {
    UT_START();
    ut_s4();
    ut_s8();
    ut_f4();
  }

  void ut_s4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s8() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S8);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S8);
  }

  void ut_f4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::F4_BNB);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::F4_BNB);
  }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems, int threads,
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
    avector<int8_t> bufB(tmpB.mSize * batch);
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
      for (int i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / threads, band);
  }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_mem(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems,
                     int threads, BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerKBlock<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherKBlock<Core_T::ISA, Core_T, prologue_a::gemm::ActivationBase, Wei,
                                      epilogue::gemm::CompFp32BlockEpilogue, epilogue::gemm::AccumulatorWriteBackFp32>;
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
    auto memsize = (size_t)tmpB.mSize + (m * k + m * n) * sizeof(float);
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
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
    tm.start();
    while (tm.stop() < timems) {
      log.start();
      for (size_t i = 0; i < batch; i++) {
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{gp,
                                      {A + i * m * k, k},
                                      {&packBs[i]},
                                      {packBs[i].template SPtr<int8_t>(), packBs[i].SDtype(), packBs[i].CStep()},
                                      {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
      }
      log.stop();
    }
    log.record();
    double t = log.min_val / batch;
    double flops = double(psize) / t / 1e6;
    double band = double(memsize) / t / 1e6;
    printf("Threads %d Block %d %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, flops, flops / threads, band);
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(int m, int n, int k, BTLA_DTYPE qtype) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, qtype, BTLA_DTYPE::F32);
    int batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(qtype),
           bestla_dtype_str(BTLA_DTYPE::F32));
    avector<float> A(size_t(m) * k * batch);
    avector<float> B(size_t(k) * n);
    avector<float> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (int i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      for (auto blocksize : {32, 128}) {
        if (_cd->AVX512F()) {
          if (m > 4) {
            benchmark<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                        C.data(), testtime, threads, qtype);
          } else {
            benchmark_mem<gemm::SCoreRowNAvx512f<48, 8>, LOG, Wei, Scale_T>(
                m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
            benchmark_mem<gemm::SCoreRowNAvx512f<96, 4>, LOG, Wei, Scale_T>(
                m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
          }
        }
        if (_cd->AVX2()) {
          if (m > 4) {
            benchmark<gemm::SCoreRowNAvx2<24, 4>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                     C.data(), testtime, threads, qtype);
          } else {
            benchmark_mem<gemm::SCoreRowNAvx2<24, 4>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                         C.data(), testtime, threads, qtype);
            benchmark_mem<gemm::SCoreRowNAvx2<48, 2>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                         C.data(), testtime, threads, qtype);
          }
        }
      }
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UTWOQ_CompFp32 sUTWOQ_CompFp32;
#endif

class UTWOQ_CompBf16 {
 public:
  UTWOQ_CompBf16() {
    UT_START();
    ut_s4();
    ut_s8();
    ut_f4();
  }

  void ut_s4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s8() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S8);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S8);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::S8);
  }

  void ut_f4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::F4_BNB);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::F4_BNB);
    benchmark_all<prologue_b::gemm::WeightKBlockNFloat, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::F4_BNB);
  }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems, int threads,
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
    avector<int8_t> bufB(tmpB.mSize * batch);
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
      for (int i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k);
        typename Launcher::Param args{gp, {A + i * m * k, k}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / threads, band);
  }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_mem(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems,
                     int threads, BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerKBlock<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherKBlock<Core_T::ISA, Core_T, prologue_a::gemm::ActivationConverterFp32, Wei,
                                      epilogue::gemm::CompFp32BlockEpilogue, epilogue::gemm::AccumulatorWriteBackFp32>;
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
    auto memsize = (size_t)tmpB.mSize + (m * k + m * n) * sizeof(float);
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
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
    tm.start();
    while (tm.stop() < timems) {
      log.start();
      for (size_t i = 0; i < batch; i++) {
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{gp,
                                      {A + i * m * k, k},
                                      {&packBs[i]},
                                      {packBs[i].template SPtr<int8_t>(), packBs[i].SDtype(), packBs[i].CStep()},
                                      {C + i * m * n, n}};
        parallel::GemmRun<Parallel>(kernel, args, UT_Threading::get());
      }
      log.stop();
    }
    log.record();
    double t = log.min_val / batch;
    double flops = double(psize) / t / 1e6;
    double band = double(memsize) / t / 1e6;
    printf("Threads %d %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, corestr, flops,
           flops / threads, band);
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(int m, int n, int k, BTLA_DTYPE qtype) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, qtype, BTLA_DTYPE::F32);
    int batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(qtype),
           bestla_dtype_str(BTLA_DTYPE::F32));
    avector<float> A(size_t(m) * k * batch);
    avector<float> B(size_t(k) * n);
    avector<float> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (int i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      for (auto blocksize : {32, 128}) {
        if (_cd->AMX_BF16()) {
          if (m > 4) {
            benchmark<gemm::HCoreRowNAmxbf16<64, 16>, LOG, Wei, Scale_T>(m, n, k, batch, blocksize, A.data(), B.data(),
                                                                         C.data(), testtime, threads, qtype);
          } else {
            benchmark_mem<gemm::HCoreRowNAmxbf16<64, 16>, LOG, Wei, Scale_T>(
                m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
            benchmark_mem<gemm::HCoreRowNAvx512bf16<96, 4>, LOG, Wei, Scale_T>(
                m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
          }
        }
      }
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
static UTWOQ_CompBf16 sUTWOQ_CompBf16;
#endif

class UTWOQ_CompInt8 {
 public:
  UTWOQ_CompInt8() {
    UT_START();
    ut_s4();
    // ut_s8();
  }

  void ut_s4() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S4_CLIP);
    /*benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S4_CLIP);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::S4_CLIP);*/
  }

  void ut_s8() {
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1, 4096, 4096, BTLA_DTYPE::S8);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(1024, 4096, 4096, BTLA_DTYPE::S8);
    benchmark_all<prologue_b::gemm::WeightKBlockNInteger, utils::bf16>(2048, 4096, 4096, BTLA_DTYPE::S8);
  }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerKBlockS<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherIntKBlock<Core_T::ISA, Core_T, prologue_a::gemm::ActivationF32KBlockQuantize, Wei,
                                         epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    kernel.mProB.packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto quanA = kernel.mProA.createStorage(m, k, blocksize, false);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    tm.start();
    while (tm.stop() < timems) {
      for (int i = 0; i < batch; i++) {
        log.start();
        GemmProblem gp(1, m, n, k, blocksize);
        typename Launcher::Param args{gp, {A + i * m * k, k, &quanA}, {&packBs[i]}, {C + i * m * n, n}};
        parallel::GemmRunWithA<Parallel>(kernel, args, UT_Threading::get());
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / threads, band);
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(int m, int n, int k, BTLA_DTYPE qtype) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, qtype, BTLA_DTYPE::F32);
    int batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(qtype),
           bestla_dtype_str(BTLA_DTYPE::F32));
    avector<float> A(size_t(m) * k * batch);
    avector<float> B(size_t(k) * n);
    avector<float> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (int i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      for (auto blocksize : {32, 128}) {
        if (_cd->AMX_INT8()) {
          benchmark<gemm::ICoreRowNAmxint8KBlock<64, 16>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
        }
        if (_cd->AVX512_VNNI()) {
          benchmark<gemm::ICoreRowNAvx512vnniKBlock<48, 4>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
          benchmark<gemm::ICoreRowNAvx512vnniKBlock<96, 2>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
        }
        if (_cd->AVX_VNNI()) {
          benchmark<gemm::ICoreRowNAvxvnniKBlock<24, 2>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
          benchmark<gemm::ICoreRowNAvxvnniKBlock<48, 1>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
        }
      }
    }
  }
};
#ifdef BTLA_UT_PROLOGUE_B
#endif
static UTWOQ_CompInt8 sUTWOQ_CompInt8;

typedef struct {
  float d;             // delta
  uint8_t qs[32 / 2];  // nibbles / quants
} block_q4_0;
typedef struct {
  float d;         // delta
  uint8_t qs[32];  // nibbles / quants
} block_q8_0;
#define __AVX2__
// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
  const __m128i tmp = _mm_loadu_si128((const __m128i*)rsi);
  const __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
  const __m256i lowMask = _mm256_set1_epi8(0xF);
  return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
  const __m256i ones = _mm256_set1_epi16(1);
  const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
  return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if __AVXVNNI__
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Perform multiplication and create 16-bit values
  const __m256i dot = _mm256_maddubs_epi16(ax, sy);
  return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
  const __m256i zero = _mm256_setzero_si256();
  const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
  return _mm256_cvtepi32_ps(summed_pairs);
#else
  // Get absolute values of x vectors
  const __m256i ax = _mm256_sign_epi8(x, x);
  // Sign the values of the y vectors
  const __m256i sy = _mm256_sign_epi8(y, x);
  return mul_sum_us8_pairs_float(ax, sy);
#endif
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
  __m128 res = _mm256_extractf128_ps(x, 1);
  res = _mm_add_ps(res, _mm256_castps256_ps128(x));
  res = _mm_add_ps(res, _mm_movehl_ps(res, res));
  res = _mm_add_ss(res, _mm_movehdup_ps(res));
  return _mm_cvtss_f32(res);
}

static void ne_vec_dot_q4_0_q8_0(const int n, float* s, const void* vx, const void* vy) {
  const int qk = 32;
  const int nb = n / qk;

  assert(n % qk == 0);
  assert(nb % 2 == 0);

  const block_q4_0* x = (const block_q4_0*)vx;
  const block_q8_0* y = (const block_q8_0*)vy;

#if defined(__AVX2__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    /* Compute combined scale for the block */
    const __m256 d = _mm256_set1_ps((x[i].d) * (y[i].d));

    __m256i bx = bytes_from_nibbles_32(x[i].qs);

    // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
    const __m256i off = _mm256_set1_epi8(8);
    bx = _mm256_sub_epi8(bx, off);

    __m256i by = _mm256_loadu_si256((const __m256i*)y[i].qs);

    const __m256 q = mul_sum_i8_pairs_float(bx, by);

    /* Multiply q with scale and accumulate */
    acc = _mm256_fmadd_ps(d, q, acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__AVX__)
  // Initialize accumulator with zeros
  __m256 acc = _mm256_setzero_ps();

  // Main loop
  for (int i = 0; i < nb; ++i) {
    // Compute combined scale for the block
    const __m256 d = _mm256_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    const __m128i tmp = _mm_loadu_si128((const __m128i*)x[i].qs);

    __m128i bx = _mm_and_si128(lowMask, tmp);
    __m128i by = _mm_loadu_si128((const __m128i*)y[i].qs);
    bx = _mm_sub_epi8(bx, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx, by);

    bx = _mm_and_si128(lowMask, _mm_srli_epi64(tmp, 4));
    by = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
    bx = _mm_sub_epi8(bx, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx, by);

    // Convert int32_t to float
    __m256 p = _mm256_cvtepi32_ps(_mm256_set_m128i(i32_0, i32_1));

    // Apply the scale, and accumulate
    acc = _mm256_add_ps(_mm256_mul_ps(d, p), acc);
  }

  *s = hsum_float_8(acc);
#elif defined(__SSSE3__)
  // set constants
  const __m128i lowMask = _mm_set1_epi8(0xF);
  const __m128i off = _mm_set1_epi8(8);

  // Initialize accumulator with zeros
  __m128 acc_0 = _mm_setzero_ps();
  __m128 acc_1 = _mm_setzero_ps();
  __m128 acc_2 = _mm_setzero_ps();
  __m128 acc_3 = _mm_setzero_ps();

  // First round without accumulation
  {
    _mm_prefetch(&x[0] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[0] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 0 and 1
    const __m128 d_0_1 = _mm_set1_ps(NE_FP16_TO_FP32(x[0].d) * NE_FP16_TO_FP32(y[0].d));

    const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i*)x[0].qs);

    __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
    __m128i by_0 = _mm_loadu_si128((const __m128i*)y[0].qs);
    bx_0 = _mm_sub_epi8(bx_0, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

    __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
    __m128i by_1 = _mm_loadu_si128((const __m128i*)(y[0].qs + 16));
    bx_1 = _mm_sub_epi8(bx_1, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

    _mm_prefetch(&x[1] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[1] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 2 and 3
    const __m128 d_2_3 = _mm_set1_ps(NE_FP16_TO_FP32(x[1].d) * NE_FP16_TO_FP32(y[1].d));

    const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i*)x[1].qs);

    __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
    __m128i by_2 = _mm_loadu_si128((const __m128i*)y[1].qs);
    bx_2 = _mm_sub_epi8(bx_2, off);
    const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

    __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
    __m128i by_3 = _mm_loadu_si128((const __m128i*)(y[1].qs + 16));
    bx_3 = _mm_sub_epi8(bx_3, off);
    const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = _mm_cvtepi32_ps(i32_0);
    __m128 p1 = _mm_cvtepi32_ps(i32_1);
    __m128 p2 = _mm_cvtepi32_ps(i32_2);
    __m128 p3 = _mm_cvtepi32_ps(i32_3);

    // Apply the scale
    acc_0 = _mm_mul_ps(d_0_1, p0);
    acc_1 = _mm_mul_ps(d_0_1, p1);
    acc_2 = _mm_mul_ps(d_2_3, p2);
    acc_3 = _mm_mul_ps(d_2_3, p3);
  }

  // Main loop
  for (int i = 2; i < nb; i += 2) {
    _mm_prefetch(&x[i] + sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[i] + sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 0 and 1
    const __m128 d_0_1 = _mm_set1_ps(NE_FP16_TO_FP32(x[i].d) * NE_FP16_TO_FP32(y[i].d));

    const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i*)x[i].qs);

    __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
    __m128i by_0 = _mm_loadu_si128((const __m128i*)y[i].qs);
    bx_0 = _mm_sub_epi8(bx_0, off);
    const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

    __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
    __m128i by_1 = _mm_loadu_si128((const __m128i*)(y[i].qs + 16));
    bx_1 = _mm_sub_epi8(bx_1, off);
    const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

    _mm_prefetch(&x[i] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
    _mm_prefetch(&y[i] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

    // Compute combined scale for the block 2 and 3
    const __m128 d_2_3 = _mm_set1_ps(NE_FP16_TO_FP32(x[i + 1].d) * NE_FP16_TO_FP32(y[i + 1].d));

    const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i*)x[i + 1].qs);

    __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
    __m128i by_2 = _mm_loadu_si128((const __m128i*)y[i + 1].qs);
    bx_2 = _mm_sub_epi8(bx_2, off);
    const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

    __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
    __m128i by_3 = _mm_loadu_si128((const __m128i*)(y[i + 1].qs + 16));
    bx_3 = _mm_sub_epi8(bx_3, off);
    const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

    // Convert int32_t to float
    __m128 p0 = _mm_cvtepi32_ps(i32_0);
    __m128 p1 = _mm_cvtepi32_ps(i32_1);
    __m128 p2 = _mm_cvtepi32_ps(i32_2);
    __m128 p3 = _mm_cvtepi32_ps(i32_3);

    // Apply the scale
    __m128 p0_d = _mm_mul_ps(d_0_1, p0);
    __m128 p1_d = _mm_mul_ps(d_0_1, p1);
    __m128 p2_d = _mm_mul_ps(d_2_3, p2);
    __m128 p3_d = _mm_mul_ps(d_2_3, p3);

    // Acummulate
    acc_0 = _mm_add_ps(p0_d, acc_0);
    acc_1 = _mm_add_ps(p1_d, acc_1);
    acc_2 = _mm_add_ps(p2_d, acc_2);
    acc_3 = _mm_add_ps(p3_d, acc_3);
  }

  *s = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#else
  // scalar
  float sumf = 0.0;

  for (int i = 0; i < nb; i++) {
    int sumi = 0;

    for (int j = 0; j < qk / 2; ++j) {
      const int v0 = (x[i].qs[j] & 0x0F) - 8;
      const int v1 = (x[i].qs[j] >> 4) - 8;

      sumi += (v0 * y[i].qs[j]) + (v1 * y[i].qs[j + qk / 2]);
    }
    sumf += sumi * x[i].d * y[i].d;
  }

  *s = sumf;
#endif
}

class UTWOQ_GGML {
 public:
  UTWOQ_GGML() {
    UT_START();
    ut_q40();
  }

  void ut_q40() { benchmark_all<prologue_b::gemm::WeightKBlockNInteger, float>(1, 4096, 4096, BTLA_DTYPE::S4_CLIP); }

  template <typename Core_T, typename LOG_T, template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark(int m, int n, int k, int batch, int blocksize, float* A, float* B, float* C, float timems, int threads,
                 BTLA_DTYPE qtype) {
    LOG_T log;
    using Parallel = parallel::gemm::SchedulerKBlockS<Core_T>;
    using Launcher =
        wrapper::gemm::LauncherIntKBlock<Core_T::ISA, Core_T, prologue_a::gemm::ActivationF32KBlockQuantize, Wei,
                                         epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher kernel;
    UT_Threading::set_threads(threads);
    auto corestr = gemm::CoreAttr::to_str(Core_T::ID);
    utils::timer<std::chrono::milliseconds> tm;
    using WType = typename Wei<Core_T, Core_T::ISA>::StorageWeight;
    WType tmpB = kernel.mProB.createStorage(n, k, blocksize, qtype, bestla_dtype<Scale_T>, bestla_dtype<float>, false);
    std::vector<WType> packBs(batch, 0);
    avector<int8_t> bufB(tmpB.mSize * batch);
    for (size_t i = 0; i < batch; i++) {
      packBs[i] = tmpB;
      packBs[i].assign(bufB.data() + i * tmpB.mSize);
    }
    kernel.mProB.packWeight(n, k, B, n, &packBs[0], UT_Threading::get());
    for (size_t i = 1; i < batch; i++) {
      memcpy(packBs[i].template WPtr<void>(), packBs[0].template WPtr<void>(), packBs[0].template WSize<char>());
      memcpy(packBs[i].template SPtr<void>(), packBs[0].template SPtr<void>(), packBs[0].CSize() * sizeof(Scale_T));
    }
    auto quanA = kernel.mProA.createStorage(m, k, blocksize, false);
    utils::avector<int8_t> bufferA(quanA.mSize);
    quanA.assign(bufferA.data());
    auto psize = (size_t)m * n * k * 2;
    auto memsize = (size_t)packBs[0].mSize + (m * k + m * n) * sizeof(float);
    int blks = updiv(k, blocksize);
    std::vector<block_q4_0> QB(batch * n * blks);
    std::vector<block_q8_0> QA(batch * m * blks);
    int dr = updiv(n, threads);
    tm.start();
    while (tm.stop() < timems) {
      for (int i = 0; i < batch; i++) {
        log.start();
        auto qb = QB.data() + i * n * blks;
        auto qa = QA.data() + i * m * blks;
        auto cptr = C + i * m * n;
        UT_Threading::get()->parallel_for([&](int idx) {
          const int ir10 = dr * idx;
          const int ir11 = std::min(ir10 + dr, n);
          for (int ir = ir10; ir < ir11; ir++) {
            ne_vec_dot_q4_0_q8_0(k, cptr + ir, qb + ir * blks, qa);
          }
        });
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    double band = double(memsize) / log.min_val / 1e6;
    printf("Threads %d Block %d %s %s Flops:%.3fG PerCoreFlops:%.3fG MemoryBandwidth:%.3fGB/s\n", threads, blocksize,
           corestr, log.get_log_str(), flops, flops / threads, band);
  }

  template <template <class _T, BTLA_ISA> class Wei, typename Scale_T>
  void benchmark_all(int m, int n, int k, BTLA_DTYPE qtype) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, qtype, BTLA_DTYPE::F32);
    int batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(qtype),
           bestla_dtype_str(BTLA_DTYPE::F32));
    avector<float> A(size_t(m) * k * batch);
    avector<float> B(size_t(k) * n);
    avector<float> C(size_t(m) * n * batch);
    fill_buffer_randn(A.data(), k * m, (-0.5f), (0.5f));
    fill_buffer_randn(B.data(), k * n, (-0.5f), (0.5f));
    for (int i = 1; i < batch; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(float));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    GetCPUDevice();
    auto threads_cfg = UT_Threading::get_threads_config();
    for (auto threads : threads_cfg) {
      for (auto blocksize : {32}) {
        if (_cd->AMX_INT8()) {
          benchmark<gemm::ICoreRowNAmxint8KBlock<64, 16>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
        }
        if (_cd->AVX512_VNNI()) {
          benchmark<gemm::ICoreRowNAvx512vnniKBlock<48, 4>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
          benchmark<gemm::ICoreRowNAvx512vnniKBlock<96, 2>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
        }
        if (_cd->AVX_VNNI()) {
          benchmark<gemm::ICoreRowNAvxvnniKBlock<24, 2>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
          benchmark<gemm::ICoreRowNAvxvnniKBlock<48, 1>, LOG, Wei, Scale_T>(
              m, n, k, batch, blocksize, A.data(), B.data(), C.data(), testtime, threads, qtype);
        }
      }
    }
  }
};
static UTWOQ_GGML sUTWOQ_GGML;
}  // namespace ut
}  // namespace bestla
int main() {
  printf("BesTLA Benchmark done\n");
  return 0;
}
