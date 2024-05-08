#include "bestla_wrapper.h"
#include "bestla_ut.h"
namespace bestla {
using namespace utils;
namespace ut {
class UT_Fp32Fp32 {
 public:
  UT_Fp32Fp32() {
    UT_START();
    CheckISA(AVX2);
    ut<sAVX2>(1, 1, 1);
    ut<sAVX2>(8, 48, 2);
    ut<sAVX2>(8, 4096, 4096);
    ut<sAVX2>(384, 768, 768);
    ut<sAVX2>(1024, 1024, 1024);
    ut<sAVX2>(1024, 1536, 1536);

    CheckISA(AVX512F);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(384, 768, 768);
    ut<gemm::SCoreRowNAvx512f<48, 8>>(1024, 1024, 1024);
    ut<sAVX512F>(1, 1, 1);
    ut<sAVX512F>(8, 48, 2);
    ut<sAVX512F>(8, 4096, 4096);
    ut<sAVX512F>(384, 768, 768);
    ut<sAVX512F>(1024, 1024, 1024);
    ut<sAVX512F>(1024, 1536, 1536);
  }
  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d Core:%s\n", m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    fill_buffer_randn(matB.data(), matB.size(), -0.5f, 0.5f);
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                    prologue_b::gemm::WeightPack, epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    launcher.mProB.packWeight(n, k, {matB.data(), n, &packw}, UT_Threading::get());
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{gp, {matA.data(), k}, {matB.data(), n, &packw}, {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, UT_Threading::get());
    ut::buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
#ifdef BTLA_UT_WRAPPER
static UT_Fp32Fp32 sUT_Fp32Fp32;
#endif

class UT_U8S8S32 {
 public:
  UT_U8S8S32() {
    UT_START();
    GetCPUDevice();
    if (_cd->AVX512_VNNI()) {
      ut<sAVX512_VNNI>(4, 48, 4);
      ut<sAVX512_VNNI>(1, 1, 1);
      ut<sAVX512_VNNI>(8, 48, 2);
      ut<sAVX512_VNNI>(8, 4096, 4096);
      ut<sAVX512_VNNI>(384, 768, 768);
      ut<sAVX512_VNNI>(1024, 1024, 1024);
      ut<sAVX512_VNNI>(1024, 1536, 1536);
    }
    if (_cd->AVX_VNNI()) {
      ut<sAVX_VNNI>(1, 1, 1);
      ut<sAVX_VNNI>(8, 48, 2);
      ut<sAVX_VNNI>(8, 4096, 4096);
      ut<sAVX_VNNI>(384, 768, 768);
      ut<sAVX_VNNI>(1024, 1024, 1024);
      ut<sAVX_VNNI>(1024, 1536, 1536);
    }
    if (_cd->AMX_INT8()) {
      request_perm_xtile_data();
      ut<sAMX_INT8_US>(1, 1, 1);
      ut<sAMX_INT8_US>(8, 48, 2);
      ut<sAMX_INT8_US>(8, 4096, 4096);
      ut<sAMX_INT8_US>(384, 768, 768);
      ut<sAMX_INT8_US>(1024, 1024, 1024);
      ut<sAMX_INT8_US>(1024, 1536, 1536);
    }
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d Core:%s\n", m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n);
    avector<uint8_t> matAu8(m * k), zpAu8(m);
    avector<int8_t> matBs8(k * n);
    avector<float> scaleAf32(m), scaleBf32(n);
    fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpAu8.data(), zpAu8.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(matBs8.data(), matBs8.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    fill_buffer_randn(scaleBf32.data(), scaleBf32.size(), 0.001f, 0.005f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] = (float(matAu8[i * k + j]) - zpAu8[i]) * scaleAf32[i];
      }
    }
    avector<float> reduceB(n, 0);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matBf32[i * n + j] = (float(matBs8[i * n + j])) * scaleBf32[j];
        reduceB[j] += matBf32[i * n + j];
      }
    }
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                                 prologue_b::gemm::WeightPack, epilogue::gemm::ZpDequantInt32ToFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    launcher.mProB.packWeight(n, k, {matBs8.data(), n, &packw}, UT_Threading::get());
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{
        gp,
        {matAu8.data(), k},
        {matBs8.data(), n, &packw},
        {matC.data(), n, 1, scaleAf32.data(), scaleBf32.data(), zpAu8.data(), reduceB.data()}};
    parallel::GemmRun<Parallel>(launcher, args, UT_Threading::get());
    ut::buffer_error(refC.data(), matC.data(), refC.size(), 0.001f);
  }
};
#ifdef BTLA_UT_WRAPPER
static UT_U8S8S32 sUT_U8S8S32;
#endif

class UT_S8S8S32 {
 public:
  UT_S8S8S32() {
    UT_START();
    GetCPUDevice();
    if (_cd->AVX_VNNI()) {
      ut<sAVX_VNNI_SS>(1, 1, 1);
      ut<sAVX_VNNI_SS>(8, 48, 2);
      ut<sAVX_VNNI_SS>(8, 4096, 4096);
      ut<sAVX_VNNI_SS>(384, 768, 768);
    }
    if (_cd->AMX_INT8()) {
      ut<sAMX_INT8_SS>(1, 1, 1);
      ut<sAMX_INT8_SS>(8, 48, 2);
      ut<sAMX_INT8_SS>(8, 4096, 4096);
      ut<sAMX_INT8_SS>(384, 768, 768);
      ut<sAMX_INT8_SS>(1024, 1024, 1024);
      ut<sAMX_INT8_SS>(1024, 1536, 1536);
    }
  }
  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case: %d %d %d Core:%s\n", m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    avector<float> matBf32(k * n), matAf32(m * k), matC(m * n), refC(m * n);
    avector<int8_t> matAu8(m * k);
    avector<int8_t> matBs8(k * n);
    avector<float> scaleAf32(m), scaleBf32(n);
    fill_buffer_randn(matAu8.data(), matAu8.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(matBs8.data(), matBs8.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(scaleAf32.data(), scaleAf32.size(), 0.001f, 0.005f);
    fill_buffer_randn(scaleBf32.data(), scaleBf32.size(), 0.001f, 0.005f);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        matAf32[i * k + j] = (float(matAu8[i * k + j])) * scaleAf32[i];
      }
    }
    avector<float> reduceB(n, 0);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matBf32[i * n + j] = (float(matBs8[i * n + j])) * scaleBf32[j];
        reduceB[j] += matBf32[i * n + j];
      }
    }
    gemmref_fp32fp32fp32(m, n, k, matAf32.data(), matBf32.data(), refC.data(), k, n, n);
    using Launcher = wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                                 prologue_b::gemm::WeightPack, epilogue::gemm::DequantInt32ToFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;

    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    launcher.mProB.packWeight(n, k, {matBs8.data(), n, &packw}, UT_Threading::get());
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{
        gp, {matAu8.data(), k}, {matBs8.data(), n, &packw}, {matC.data(), n, 1, scaleAf32.data(), scaleBf32.data()}};
    parallel::GemmRun<Parallel>(launcher, args, UT_Threading::get());
    ut::buffer_error(refC.data(), matC.data(), refC.size(), 0.001f);
  }
};
#ifdef BTLA_UT_WRAPPER
static UT_S8S8S32 sUT_S8S8S32;
#endif

class UT_Bf16Bf16Fp32 {
 public:
  UT_Bf16Bf16Fp32() {
    UT_START();
    CheckISA(AMX_BF16);
    ut<sAMX_BF16>(1, 1, 1);
    ut<sAMX_BF16>(8, 48, 2);
    ut<sAMX_BF16>(8, 4096, 4096);
    ut<sAMX_BF16>(384, 768, 768);
    ut<sAMX_BF16>(1024, 1024, 1024);
    ut<sAMX_BF16>(1024, 1536, 1536);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case %s: %d %d %d core:%s\n", __FUNCTION__, m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                    prologue_b::gemm::WeightPack, epilogue::gemm::AccumulatorWriteBackFp32>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    avector<utils::bf16> matAbf16(m * k), matBbf16(k * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::bf16(-0.5f), utils::bf16(0.5f));
    avector<float> matC(m * n), refC(m * n);
    launcher.mProB.packWeight(n, k, {matBbf16.data(), n, &packw}, UT_Threading::get());
    gemmref_bf16bf16fp32(m, n, k, matAbf16.data(), matBbf16.data(), refC.data(), k, n, n);
    utils::GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{gp, {matAbf16.data(), k}, {matBbf16.data(), n, &packw}, {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, UT_Threading::get());
    buffer_error(refC.data(), matC.data(), refC.size(), 0.05f);
  }
};
#ifdef BTLA_UT_WRAPPER
static UT_Bf16Bf16Fp32 sUT_Bf16Bf16Fp32;
#endif

class UT_Fp16Fp16Fp16 {
 public:
  UT_Fp16Fp16Fp16() {
    UT_START();
    CheckISA(AVX512_FP16);
    ut<sAVX512_FP16>(1, 1, 1);
    ut<sAVX512_FP16>(8, 48, 2);
    ut<sAVX512_FP16>(8, 4096, 4096);
    ut<sAVX512_FP16>(384, 768, 768);
    ut<sAVX512_FP16>(1024, 1024, 1024);
    ut<sAVX512_FP16>(1024, 1536, 1536);
  }

  template <class GemmCore_T>
  void ut(int m, int n, int k) {
    printf("Test Case %s: %d %d %d core:%s\n", __FUNCTION__, m, n, k, gemm::CoreAttr::to_str(GemmCore_T::ID));
    using Launcher =
        wrapper::gemm::LauncherBase<GemmCore_T::ISA, GemmCore_T, prologue_a::gemm::ActivationBase,
                                    prologue_b::gemm::WeightPack, epilogue::gemm::AccumulatorWriteBackFp16>;
    Launcher launcher;
    using Parallel = parallel::gemm::SchedulerBase<GemmCore_T>;
    auto packw = launcher.mProB.createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    avector<utils::fp16> matAbf16(m * k), matBbf16(k * n), matC(m * n), refC(m * n);
    fill_buffer_randn(matAbf16.data(), matAbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matBbf16.data(), matBbf16.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    launcher.mProB.packWeight(n, k, {matBbf16.data(), n, &packw}, UT_Threading::get());
    gemmref_fp16fp16fp16(m, n, k, matAbf16.data(), matBbf16.data(), refC.data(), k, n, n);
    GemmProblem gp(1, m, n, k);
    typename Launcher::Param args{gp, {matAbf16.data(), k}, {matBbf16.data(), n, &packw}, {matC.data(), n}};
    parallel::GemmRun<Parallel>(launcher, args, UT_Threading::get());
    buffer_error(refC.data(), matC.data(), refC.size(), utils::fp16(0.0002f * k));
  }
};
#ifdef BTLA_UT_WRAPPER
static UT_Fp16Fp16Fp16 sUT_Fp16Fp16Fp16;
#endif
}  // namespace ut
}  // namespace bestla
