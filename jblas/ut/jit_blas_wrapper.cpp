#include "../jit_blas_wrapper.h"

#include "jit_blas_ut.h"
namespace jblas {
using namespace utils;

namespace wrapper {
namespace gemm {

class UT_AVX512F_NN {
 public:
  UT_AVX512F_NN() {
    UT_START();
    CheckISA(AVX512F);
    ut(1024, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
    ut(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 48, 1.f, 1.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);
    using GemmKernel = jblas::wrapper::gemm_default::avx512f::GemmKernel;
    GemmKernel kernelavx512f;
    auto packw = kernelavx512f.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernelavx512f.getWeightPtr()->packWeight(n, k, {_data.matB.data(), ldb, &packw});
    GemmKernel::Arguments args{
        m,     n,    k,   _data.matA.data(), lda, NULL, 0, &packw, _data.matC.data(), _data.matD.data(), ldc, ldd,
        alpha, beta, NULL};
    kernelavx512f.compute(args);
    _data.calc_ref(alpha, beta);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AVX512F_NN sUT_AVX512F_NN;
#endif

class UT_AVX512_FP16_NN {
 public:
  UT_AVX512_FP16_NN() {
    UT_START();
    CheckISA(AVX512_FP16);
    ut(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 64, 2, 2, 64, 64, 0, 1.f, 0.f);
    ut(8, 64, 128, 128, 64, 64, 0, 1.f, 0.f);
    ut(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f);
    ut(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut(1024, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);

    ut_96(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut_96(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut_96(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    if (beta != 0.f && alpha != 1.f) {
      printf("No alpha beta support!\n");
      return;
    }
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    ut::UT_GEMMData_Row_fp16 _data(m, n, k, lda, ldb, ldc, ldd);
    using GemmKernel = jblas::wrapper::gemm_default::avx512_fp16::GemmKernel;
    GemmKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_data.matB.data(), ldb, &packw});
    aligned_vector<fp16> ret(_data.matC.size());
    _data.calc_ref(alpha, beta);
    GemmKernel::Arguments args{m, n, k, _data.matA.data(), lda, NULL, 0, &packw, ret.data(), ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(_data.matC.data(), ret.data(), _data.matC.size(), fp16(0.001f));
  }

  void ut_96(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    if (beta != 0.f && alpha != 1.f) {
      printf("No alpha beta support!\n");
      return;
    }
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    ut::UT_GEMMData_Row_fp16 _data(m, n, k, lda, ldb, ldc, ldd);
    using GemmKernel = jblas::wrapper::gemm_default::avx512_fp16::GemmKernel_96;
    GemmKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_data.matB.data(), ldb, &packw});
    aligned_vector<fp16> ret(_data.matC.size());
    _data.calc_ref(alpha, beta);
    GemmKernel::Arguments args{m, n, k, _data.matA.data(), lda, NULL, 0, &packw, ret.data(), ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(_data.matC.data(), ret.data(), _data.matC.size(), fp16(0.001f));
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AVX512_FP16_NN sUT_AVX512_FP16_NN;
#endif

class UT_AVX512VNNI_NN {
 public:
  UT_AVX512VNNI_NN() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 48, 1.f, 1.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 4, 4, 48, 48, 0, 1.f, 0.f);
    ut(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f);
    ut(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    beta = 0.f;  // no beta
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, ldd);
    _quan.calc_ref(alpha, beta);
    using GemmKernel = jblas::wrapper::gemm_default::avx512_vnni::GemmKernel;
    GemmKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_quan.matB.data(), ldb, &packw});
    utils::aligned_vector<uint8_t> kernlret(m * ldc);
    GemmKernel::Arguments args{m,
                               n,
                               k,
                               _quan.matA.data(),
                               lda,
                               NULL,
                               0,
                               &packw,
                               kernlret.data(),
                               ldc,
                               alpha,
                               _quan.matA.scales[0] * _quan.matB.scales[0],
                               _quan.matC.scales[0],
                               _quan.matC.zeropoints[0],
                               NULL};
    kernel.compute(args);
    ut::buffer_error(_quan.matC.data(), kernlret.data(), kernlret.size(), (uint8_t)1);
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AVX512VNNI_NN sUT_AVX512VNNI_NN;
#endif

class UT_AVXVNNI_NN {
 public:
  UT_AVXVNNI_NN() {
    UT_START();
    CheckISA(AVX_VNNI);
    ut(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 48, 1.f, 1.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 4, 4, 48, 48, 0, 1.f, 0.f);
    ut(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f);
    ut(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    beta = 0.f;  // no beta
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, ldd);
    _quan.calc_ref(alpha, beta);
    using GemmKernel = jblas::wrapper::gemm_default::avx_vnni::GemmKernel48;
    GemmKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_quan.matB.data(), ldb, &packw});
    utils::aligned_vector<uint8_t> kernlret(m * ldc);
    GemmKernel::Arguments args{m,
                               n,
                               k,
                               _quan.matA.data(),
                               lda,
                               NULL,
                               0,
                               &packw,
                               kernlret.data(),
                               ldc,
                               alpha,
                               _quan.matA.scales[0] * _quan.matB.scales[0],
                               _quan.matC.scales[0],
                               _quan.matC.zeropoints[0],
                               NULL};
    kernel.compute(args);
    ut::buffer_error(_quan.matC.data(), kernlret.data(), kernlret.size(), (uint8_t)1);
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AVXVNNI_NN sUT_AVXVNNI_NN;
#endif

class UT_AVX512VNNI_NN_DynamicQuantNew {
 public:
  UT_AVX512VNNI_NN_DynamicQuantNew() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut(1, 1, 1, 1, 1, 1);
    ut(8, 48, 2, 2, 48, 48);
    ut(8, 48, 4, 4, 48, 48);
    ut(8, 4096, 4096, 4096, 4096, 4096);
    ut(8, 32, 32, 32, 32, 32, true);
    ut(8, 4096, 4096, 4096, 4096, 4096, true);
    ut(1, 4096, 4, 4, 4096, 4096);
    ut(1024, 1024, 1024, 1024, 1024, 1024);
    ut(32, 48, 1024, 1024, 48, 48);
    ut(64, 48, 1024, 1024, 48, 48);
    ut(8, 1536, 1024, 1024, 1536, 1536);
    ut(1536, 1536, 1536, 1536, 1536, 1536);
    ut_ab(1024, 1024, 1024, 1024, 1024, 1024);
    ut_ab(1536, 1536, 1536, 1536, 1536, 1536);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, bool asym = false) {
    printf("Test Case %s: %d %d %d %d %d %s\n", __FUNCTION__, m, n, k, lda, ldc, asym ? "asym" : "sym");
    ut::UT_GEMMData_Row_f32 rawf32(m, n, k, lda, ldb, ldc, 0);
    aligned_vector<uint8_t> matAu8(k * m);
    ut::fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    aligned_vector<uint8_t> matAzp(m);
    ut::fill_buffer_randn(matAzp.data(), matAzp.size(), uint8_t(0), uint8_t(255));
    aligned_vector<float> Ascales(m);
    ut::fill_buffer_randn(Ascales.data(), Ascales.size(), 0.01f, 0.015f);

    aligned_vector<int8_t> matBs8(k * n);
    ut::fill_buffer_randn(matBs8.data(), matBs8.size(), int8_t(-127), int8_t(127));
    aligned_vector<int8_t> matBzp(n);
    ut::fill_buffer_randn(matBzp.data(), matBzp.size(), int8_t(-127), int8_t(127));
    aligned_vector<float> Bscales(n), matC(m * n);
    ut::fill_buffer_randn(Bscales.data(), Bscales.size(), 0.01f, 0.015f);
    avector<float> Breduce(n, 0.f);
    avector<float> Areduce(m, 0.f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        if (i == 0) matBs8[i * n + j] = -128;
        if (i == 1) matBs8[i * n + j] = 127;
        float matBf32 = float(matBs8[i * n + j]);
        if (asym) matBf32 -= float(matBzp[j]);
        rawf32.matB[i * n + j] = matBf32 * Bscales[j];
        Breduce[j] += rawf32.matB[i * n + j];
      }
    }
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        if (j == 0) matAu8[i * k + j] = (uint8_t)0;
        if (j == 1) matAu8[i * k + j] = (uint8_t)255;
        rawf32.matA[i * k + j] = (float(matAu8[i * k + j]) - float(matAzp[i])) * Ascales[i];
        Areduce[i] += rawf32.matA[i * k + j];
      }
    }

    auto constexpr RuntimeISA = JblasAVX512_VNNI;
    using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
            RuntimeISA,                                            //
            jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,         //
            jblas::prologue::gemm::ActivationFp32AsymU8Quantize,   //
            jblas::prologue::gemm::WeightPack,                     //
            jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
        jblas::utils::parallel::Parallel2DGemm>;
    GemmKernel kernel;
    auto dq = kernel.getActivationPtr();
    auto refquanA = dq->createStorage(m, k);
    avector<int8_t> refbufA(refquanA.mSize);
    refquanA.assign(refbufA.data());
    auto quanA = dq->createStorage(m, k);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());

    rawf32.calc_ref(1.f, 0.f);
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {matBs8.data(), ldb, &packw});
    GemmKernel::Arguments args{m,
                               n,
                               k,
                               rawf32.matA.data(),
                               lda,
                               &quanA,
                               NULL,
                               0,
                               &packw,
                               matC.data(),
                               ldc,
                               quanA.mCStep,
                               quanA.mSPtr,
                               Bscales.data(),
                               quanA.mZPtr,
                               Breduce.data(),
                               asym ? matBzp.data() : nullptr,
                               asym ? Areduce.data() : nullptr,
                               k,
                               NULL};
    kernel.template compute<true, false>(args);
    ut::buffer_error(rawf32.matRef.data(), matC.data(), matC.size(), 0.6f);
  }

  void ut_ab(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case %s: %d %d %d %d %d\n", __FUNCTION__, m, n, k, lda, ldc);
    ut::UT_GEMMData_Row_f32 rawf32(m, n, k, lda, ldb, ldc, 0);
    aligned_vector<float> matAf32(m * k);
    ut::fill_buffer_randn(matAf32.data(), matAf32.size(), 0.05f, 0.1f);
    aligned_vector<int8_t> matBs8(k * n);
    ut::fill_buffer_randn(matBs8.data(), matBs8.size(), int8_t(-127), int8_t(127));
    aligned_vector<float> Bscales(n), matC(m * n);
    ut::fill_buffer_randn(Bscales.data(), Bscales.size(), 0.01f, 0.02f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        rawf32.matB[i * n + j] = matBs8[i * n + j] * Bscales[j];
      }
    }
    auto constexpr RuntimeISA = JblasAVX512_VNNI;
    using GemmKernel = jblas::wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<  //
            RuntimeISA,                                            //
            jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI,         //
            jblas::prologue::gemm::ActivationFp32AsymU8Quantize,   //
            jblas::prologue::gemm::WeightPack,                     //
            jblas::epilogue::gemm::DequantInt32ToFp32>,
        jblas::utils::parallel::Parallel2DGemm>;
    GemmKernel kernel;
    auto dq = kernel.getActivationPtr();
    auto refquanA = dq->createStorage(m, k);
    avector<int8_t> refbufA(refquanA.mSize);
    refquanA.assign(refbufA.data());
    auto quanA = dq->createStorage(m, k);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    dq->quantize({matAf32.data(), lda, &refquanA}, m, k);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        rawf32.matA[i * k + j] =
            (int(refquanA.template get<uint8_t>()[i * refquanA.lda + j]) - refquanA.mZPtr[i * quanA.mCStep]) *
            refquanA.mSPtr[i * quanA.mCStep];
      }
    }
    rawf32.calc_ref(1.f, 0.f);
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    GemmKernel::Arguments args{
        m,      n,           k,   matAf32.data(), lda,          &quanA,         matBs8.data(), ldb,
        &packw, matC.data(), ldc, quanA.mSPtr,    quanA.mCStep, Bscales.data(), NULL};
    kernel.compute<true, true>(args);
    ut::buffer_error(rawf32.matRef.data(), matC.data(), matC.size(), 0.01f);
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AVX512VNNI_NN_DynamicQuantNew sUT_AVX512VNNI_NN_DynamicQuantNew;
#endif

class UT_AMX_INT8ss_NN_DynamicQuant {
 public:
  UT_AMX_INT8ss_NN_DynamicQuant() {
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    ut(1, 1, 1, 1, 1, 1);
    ut(8, 48, 2, 2, 48, 48);
    ut(8, 48, 4, 4, 48, 48);
    ut(8, 4096, 4096, 4096, 4096, 4096);
    ut(1, 4096, 4, 4, 4096, 4096);
    ut(1024, 1024, 1024, 1024, 1024, 1024);
    ut(32, 48, 1024, 1024, 48, 48);
    ut(64, 48, 1024, 1024, 48, 48);
    ut(8, 1536, 1024, 1024, 1536, 1536);
    ut(1536, 1536, 1536, 1536, 1536, 1536);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    ut::UT_GEMMData_Row_f32 rawf32(m, n, k, lda, ldb, ldc, 0);
    aligned_vector<float> matAf32(m * k);
    ut::fill_buffer_randn(matAf32.data(), matAf32.size(), 0.f, 0.1f);
    aligned_vector<int8_t> matBs8(k * n);
    ut::fill_buffer_randn(matBs8.data(), matBs8.size(), int8_t(-127), int8_t(127));
    aligned_vector<float> Bscales(n), matC(m * n);
    ut::fill_buffer_randn(Bscales.data(), Bscales.size(), 0.01f, 0.02f);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        rawf32.matB[i * n + j] = matBs8[i * n + j] * Bscales[j];
      }
    }
    using GemmKernel = jblas::wrapper::gemm_default::amx_int8::GemmKernelDynamicQuant;
    GemmKernel kernel;
    auto dq = kernel.getActivationPtr();
    auto quanA = dq->createStorage(m, k);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    dq->quantize({matAf32.data(), lda, &quanA}, m, k);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        rawf32.matA[i * k + j] = (quanA.template get<int8_t>()[i * quanA.lda + j]) * quanA.mSPtr[i * quanA.mCStep];
      }
    }
    rawf32.calc_ref(1.f, 0.f);
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {matBs8.data(), ldb, &packw});
    GemmKernel::Arguments args{m,      n,           k,   matAf32.data(), lda,          &quanA,         NULL, 0,
                               &packw, matC.data(), ldc, quanA.mSPtr,    quanA.mCStep, Bscales.data(), NULL};
    kernel.compute<true, false>(args);
    ut::buffer_error(rawf32.matRef.data(), matC.data(), matC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AMX_INT8ss_NN_DynamicQuant sUT_AMX_INT8ss_NN_DynamicQuant;
#endif

class UT_AMXINT8_NN {
 public:
  UT_AMXINT8_NN() {
    UT_START();
    CheckISA(AMX_INT8);
    utils::request_perm_xtile_data();
    ut(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 48, 1.f, 1.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 4, 4, 48, 48, 0, 1.f, 0.f);
    ut(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f);
    ut(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    beta = 0.f;  // no beta
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, ldd);
    _quan.calc_ref(alpha, beta);
    using GemmKernel = jblas::wrapper::gemm_default::amx_int8::GemmKernel;
    GemmKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_quan.matB.data(), ldb, &packw});
    utils::aligned_vector<uint8_t> kernlret(m * ldc);
    GemmKernel::Arguments args{m,
                               n,
                               k,
                               _quan.matA.data(),
                               lda,
                               NULL,
                               0,
                               &packw,
                               kernlret.data(),
                               ldc,
                               alpha,
                               _quan.matA.scales[0] * _quan.matB.scales[0],
                               _quan.matC.scales[0],
                               _quan.matC.zeropoints[0],
                               NULL};
    kernel.compute(args);
    ut::buffer_error(_quan.matC.data(), kernlret.data(), kernlret.size(), (uint8_t)1);
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AMXINT8_NN sUT_AMXINT8_NN;
#endif

class UT_AMXINT8_48_NN {
 public:
  UT_AMXINT8_48_NN() {
    UT_START();
    CheckISA(AMX_INT8);
    utils::request_perm_xtile_data();
    ut(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 48, 1.f, 1.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 4, 4, 48, 48, 0, 1.f, 0.f);
    ut(8, 64, 2, 2, 64, 64, 0, 1.f, 0.f);
    ut(8, 64, 2, 2, 64, 64, 64, 1.f, 1.f);
    ut(8, 64, 2, 2, 64, 64, 0, 1.f, 0.f);
    ut(8, 64, 4, 4, 64, 64, 0, 1.f, 0.f);
    ut(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f);
    ut(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    beta = 0.f;  // no beta
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, ldd);
    _quan.calc_ref(alpha, beta);
    using GemmKernel = jblas::wrapper::gemm_default::amx_int8::GemmKernel48;
    GemmKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_quan.matB.data(), ldb, &packw});
    utils::aligned_vector<uint8_t> kernlret(m * ldc);
    GemmKernel::Arguments args{m,
                               n,
                               k,
                               _quan.matA.data(),
                               lda,
                               NULL,
                               0,
                               &packw,
                               kernlret.data(),
                               ldc,
                               alpha,
                               _quan.matA.scales[0] * _quan.matB.scales[0],
                               _quan.matC.scales[0],
                               _quan.matC.zeropoints[0],
                               NULL};
    kernel.compute(args);
    ut::buffer_error(_quan.matC.data(), kernlret.data(), kernlret.size(), (uint8_t)1);
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AMXINT8_48_NN sUT_AMXINT8_48_NN;
#endif

class UT_AMX_BF16_NN_PackWeight {
 public:
  UT_AMX_BF16_NN_PackWeight() {
    UT_START();
    CheckISA(AMX_BF16);
    utils::request_perm_xtile_data();
    ut(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 2, 2, 48, 48, 48, 1.f, 1.f);
    ut(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut(8, 48, 4, 4, 48, 48, 0, 1.f, 0.f);
    ut(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f);
    ut(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);

    ut_48(1, 1, 1, 1, 1, 1, 0, 1.f, 0.f);
    ut_48(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut_48(8, 48, 2, 2, 48, 48, 48, 1.f, 1.f);
    ut_48(8, 48, 2, 2, 48, 48, 0, 1.f, 0.f);
    ut_48(8, 48, 4, 4, 48, 48, 0, 1.f, 0.f);
    ut_48(8, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f);
    ut_48(1024, 1024, 1024, 1024, 1024, 1024, 0, 1.f, 0.f);
    ut_48(8, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut_48(32, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut_48(64, 48, 1024, 1024, 48, 48, 0, 1.f, 0.f);
    ut_48(8, 1536, 1024, 1024, 1536, 1536, 0, 1.f, 0.f);
    ut_48(1536, 1536, 1536, 1536, 1536, 1536, 0, 1.f, 0.f);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    beta = 0.f;
    ut::UT_GEMMData_Row_bf16 _data(m, n, k, lda, ldb, ldc, ldd);
    _data.calc_ref(alpha, beta);
    using GEMMKernel = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN;
    GEMMKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_data.matB.data(), n, &packw});
    utils::aligned_vector<utils::bf16> kernlret(m * ldc);
    GEMMKernel::Arguments args{m, n, k, _data.matA.data(), lda, NULL, 0, &packw, kernlret.data(), ldc, NULL};
    kernel.compute(args);
    utils::bf16 thres;
    thres.fromfloat(0.1f);
    ut::buffer_error(_data.matC.data(), kernlret.data(), _data.matC.size(), thres);
  }

  void ut_48(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta) {
    printf("Test Case: %d %d %d %d %d %d %f %f\n", m, n, k, lda, ldc, ldd, alpha, beta);
    beta = 0.f;
    ut::UT_GEMMData_Row_bf16 _data(m, n, k, lda, ldb, ldc, ldd);
    _data.calc_ref(alpha, beta);
    using GEMMKernel = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN_48;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packw = kernel.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer(packw.mSize);
    packw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, {_data.matB.data(), n, &packw});
    utils::aligned_vector<uint16_t> kernlret(m * ldc);
    GEMMKernel::Arguments args{
        m,   n,   k, _data.matA.data(), lda, NULL, 0, &packw, reinterpret_cast<jblas::utils::bf16*>(kernlret.data()),
        ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(reinterpret_cast<jblas::utils::bf16*>(_data.matC.data()),
                     reinterpret_cast<jblas::utils::bf16*>(kernlret.data()), _data.matC.size(),
                     jblas::utils::bf16(0x3f80));
  }
};
#ifdef JBLAS_UT_WRAPPER
static UT_AMX_BF16_NN_PackWeight sUT_AMX_BF16_NN_PackWeight;
#endif

}  // namespace gemm
}  // namespace wrapper
}  // namespace jblas
