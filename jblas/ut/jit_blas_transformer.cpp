#include "../jit_blas_transformer.h"
#include "../jit_blas_weight_compression.h"
#include "jit_blas_ut.h"

namespace jblas {
using namespace utils;
using CompType = jblas::prologue::weight_comp::gemm_kblcok::PrologueBIDs;
namespace wrapper {
namespace transformer {
class UT_AVX512VNNI_NN_QKV_INT4_BLOCK {
 public:
  UT_AVX512VNNI_NN_QKV_INT4_BLOCK() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 64);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 32);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);

    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;
        }
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
      }
    }

    utils::aligned_vector<uint8_t> matAu8(m * lda), matAzp(m * kblk_num);
    ut::fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    ut::fill_buffer_randn(matAzp.data(), matAzp.size(), uint8_t(100), uint8_t(150));
    utils::aligned_vector<float> AScales(kblk_num * m);
    ut::fill_buffer_randn(AScales.data(), AScales.size(), 0.001f, 0.005f);
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        if (i % blocksize == 0) {
          matAu8.data()[i + j * lda] = 255;
        }
        if (i % blocksize == blocksize - 1) {
          matAu8.data()[i + j * lda] = 0;
        }
      }
    }
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        _data.matB[j * ldb + i] = float(quanW.data()[j * ldb + i]) * scales[j / blocksize * n + i];
      }
    }
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        _data.matA[j * lda + i] = (float(matAu8.data()[j * lda + i]) - matAzp[i / blocksize + j * kblk_num]) *
                                  AScales[i / blocksize + j * kblk_num];
      }
    }
    _data.calc_ref(alpha, beta);

    using GEMMKernel = wrapper::transformer_default::weight_comp::avx512_vnni::QKVGemmDynamicS4Fp32KBlock;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), nullptr, &packedw);
    std::vector<PrologueB::Param> bparams{{&packedw}, {&packedw}, {&packedw}};
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({_data.matA.data(), lda, &quanA}, m, k);

    ut::buffer_error(quanA.template get<uint8_t>(), matAu8.data(), matAu8.size());
    ut::buffer_error(quanA.mZPtr, matAzp.data(), matAzp.size());
    ut::buffer_error(quanA.mSPtr, AScales.data(), AScales.size(), 0.001f);
    int batch = 3;
    aligned_vector<float> matCBatch(3 * m * n);
    std::vector<GEMMKernel::CParam> cparams(batch);
    for (size_t i = 0; i < batch; i++) {
      cparams[i].ldc = ldc;
      cparams[i].C = matCBatch.data() + i * m * n;
    }
    GEMMKernel::Arguments args{m, n, k, batch, _data.matA.data(), lda, &quanA, bparams.data(), cparams.data(), NULL};

    kernel.compute(args);

    ut::buffer_error(_data.matRef.data(), cparams[0].C, _data.matRef.size(), 0.001f);
    for (size_t i = 1; i < batch; i++) {
      ut::buffer_error(cparams[0].C, cparams[i].C, _data.matRef.size(), 0.f);
    }
  }
};
#ifdef JBLAS_UT_TRANSFORMER
static UT_AVX512VNNI_NN_QKV_INT4_BLOCK sUT_AVX512VNNI_NN_QKV_INT4_BLOCK;
#endif

class UT_AMX_INT8_NN_QKV_INT4_BLOCK {
 public:
  UT_AMX_INT8_NN_QKV_INT4_BLOCK() {
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    jblas::utils::parallel::CpuDevice::getInstance()->setThreads(-1);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 256);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);

    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), -0.005f, 0.005f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;
        }
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
      }
    }

    utils::aligned_vector<int8_t> matAs8(m * lda);
    ut::fill_buffer_randn(matAs8.data(), matAs8.size(), int8_t(-127), int8_t(127));
    utils::aligned_vector<float> AScales(kblk_num * m);
    ut::fill_buffer_randn(AScales.data(), AScales.size(), 0.001f, 0.005f);
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        if (i % blocksize == 0) {
          matAs8.data()[i + j * lda] = 127;
        }
      }
    }
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        _data.matB[j * ldb + i] = float(quanW.data()[j * ldb + i]) * scales[j / blocksize * n + i];
      }
    }
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        _data.matA[j * lda + i] = (float(matAs8.data()[j * lda + i])) * AScales[i / blocksize + j * kblk_num];
      }
    }
    _data.calc_ref(alpha, beta);

    using GEMMKernel = wrapper::transformer_default::weight_comp::amx_int8::QKVGemmDynamicS4Fp32KBlock;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), nullptr, &packedw);
    std::vector<PrologueB::Param> bparams{{&packedw}, {&packedw}, {&packedw}};
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({_data.matA.data(), lda, &quanA}, m, k);

    ut::buffer_error(quanA.template get<int8_t>(), matAs8.data(), matAs8.size());
    ut::buffer_error(quanA.mSPtr, AScales.data(), AScales.size(), 0.001f);
    int batch = 3;
    aligned_vector<float> matCBatch(3 * m * n);
    std::vector<GEMMKernel::CParam> cparams(batch);
    for (size_t i = 0; i < batch; i++) {
      cparams[i].ldc = ldc;
      cparams[i].C = matCBatch.data() + i * m * n;
    }
    GEMMKernel::Arguments args{m, n, k, batch, _data.matA.data(), lda, &quanA, bparams.data(), cparams.data(), NULL};

    kernel.compute(args);

    ut::buffer_error(_data.matRef.data(), cparams[0].C, _data.matRef.size(), 0.001f);
    for (size_t i = 1; i < batch; i++) {
      ut::buffer_error(cparams[0].C, cparams[i].C, _data.matRef.size(), 0.f);
    }
  }
};
#ifdef JBLAS_UT_TRANSFORMER
static UT_AMX_INT8_NN_QKV_INT4_BLOCK sUT_AMX_INT8_NN_QKV_INT4_BLOCK;
#endif

class UT_QKVGemmInterfacePackWeight {
 public:
  UT_QKVGemmInterfacePackWeight() {
    UT_START();
    CheckISA(AVX512F);  // low ISA first
    utfp32(2, 4096, 4096, 32);
    utfp32(2, 4096, 4096, 64);
    utfp32(2, 4096, 4096, 64, true);
    utfp32(2, 4096, 4096, 128);
    utfp32(2, 4096, 4096, 128, true);
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    utbf16(2, 4096, 4096, 128);
    utbf16(2, 4096, 4096, 64);
    utbf16(2, 4096, 4096, 64, true);
  }

  void utbf16(int m, int n, int k, int blocksize, bool asym = false) {
    printf("Test Case %s: %d %d %d-%d %s\n", __FUNCTION__, m, n, k, blocksize, asym ? "asym" : "sym");
    int lda = k;
    int ldb = n;
    int ldc = n;
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));

    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;
        }
        if ((i % blocksize == 1) && asym) {
          quanW.data()[i * n + j] = -128;
        }
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
      }
    }

    utils::aligned_vector<uint8_t> matAu8(m * lda), matAzp(m * kblk_num);
    ut::fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    ut::fill_buffer_randn(matAzp.data(), matAzp.size(), uint8_t(100), uint8_t(150));
    utils::aligned_vector<float> AScales(kblk_num * m);
    ut::fill_buffer_randn(AScales.data(), AScales.size(), 0.001f, 0.005f);
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        if (i % blocksize == 0) {
          matAu8.data()[i + j * lda] = 255;
        }
        if (i % blocksize == blocksize - 1) {
          matAu8.data()[i + j * lda] = 0;
        }
      }
    }
    avector<bf16> matA(m * lda), matB(k * ldb);
    avector<float> refC(m * ldc), matAf32(m * lda);
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        float wei = float(quanW.data()[j * ldb + i]);
        if (asym) wei -= float(zero_points[j / blocksize * n + i]);
        matB[j * ldb + i] = bf16(wei * scales[j / blocksize * n + i]);
      }
    }
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        matAf32[j * lda + i] = (float(matAu8.data()[j * lda + i]) - matAzp[i / blocksize + j * kblk_num]) *
                               AScales[i / blocksize + j * kblk_num];
        matA[j * lda + i] = bf16(matAf32[j * lda + i]);
      }
    }
    ut::gemmref_bf16bf16fp32(m, n, k, matA.data(), matB.data(), refC.data(), lda, ldb, ldc);

    using GEMMKernel = wrapper::transformer_default::weight_comp::amx_bf16::QKVGemmS4Fp32Kblock;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), asym ? zero_points.data() : nullptr,
                                       &packedw);
    std::vector<PrologueB::Param> bparams{{&packedw}, {&packedw}, {&packedw}};
    int batch = 3;
    aligned_vector<float> matCBatch(3 * m * n);
    std::vector<GEMMKernel::CParam> cparams(batch);
    for (size_t i = 0; i < batch; i++) {
      cparams[i].ldc = ldc;
      cparams[i].C = matCBatch.data() + i * m * n;
    }
    GEMMKernel::Arguments args{m, n, k, batch, matAf32.data(), lda, bparams.data(), cparams.data(), NULL};

    kernel.compute(args);

    ut::buffer_error(refC.data(), cparams[0].C, refC.size(), 0.001f);
    for (size_t i = 1; i < batch; i++) {
      ut::buffer_error(cparams[0].C, cparams[i].C, refC.size(), 0.f);
    }
  }

  void utfp32(int m, int n, int k, int blocksize, bool asym = false) {
    printf("Test Case %s: %d %d %d-%d %s\n", __FUNCTION__, m, n, k, blocksize, asym ? "asym" : "sym");
    int lda = k;
    int ldb = n;
    int ldc = n;
    int ldd = n;
    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;
        }
        if (i % blocksize == 1 && asym) {
          quanW.data()[i * n + j] = -128;
        }
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
      }
    }

    utils::aligned_vector<uint8_t> matAu8(m * lda), matAzp(m * kblk_num);
    ut::fill_buffer_randn(matAu8.data(), matAu8.size(), uint8_t(0), uint8_t(255));
    ut::fill_buffer_randn(matAzp.data(), matAzp.size(), uint8_t(100), uint8_t(150));
    utils::aligned_vector<float> AScales(kblk_num * m);
    ut::fill_buffer_randn(AScales.data(), AScales.size(), 0.001f, 0.005f);
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        if (i % blocksize == 0) {
          matAu8.data()[i + j * lda] = 255;
        }
        if (i % blocksize == blocksize - 1) {
          matAu8.data()[i + j * lda] = 0;
        }
      }
    }
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        float wei = float(quanW.data()[j * ldb + i]);
        if (asym) wei -= float(zero_points[j / blocksize * n + i]);
        _data.matB[j * ldb + i] = wei * scales[j / blocksize * n + i];
      }
    }
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        _data.matA[j * lda + i] = (float(matAu8.data()[j * lda + i]) - matAzp[i / blocksize + j * kblk_num]) *
                                  AScales[i / blocksize + j * kblk_num];
      }
    }
    _data.calc_ref(1.0f, 0.f);

    using GEMMKernel = wrapper::transformer_default::weight_comp::avx512f::QKVGemmS4Fp32Kblock;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), asym ? zero_points.data() : nullptr,
                                       &packedw);
    std::vector<PrologueB::Param> bparams{{&packedw}, {&packedw}, {&packedw}};
    int batch = 3;
    aligned_vector<float> matCBatch(3 * m * n);
    std::vector<GEMMKernel::CParam> cparams(batch);
    for (size_t i = 0; i < batch; i++) {
      cparams[i].ldc = ldc;
      cparams[i].C = matCBatch.data() + i * m * n;
    }
    GEMMKernel::Arguments args{m, n, k, batch, _data.matA.data(), lda, bparams.data(), cparams.data(), NULL};

    kernel.compute(args);

    ut::buffer_error(_data.matRef.data(), cparams[0].C, _data.matRef.size(), 0.001f);
    for (size_t i = 1; i < batch; i++) {
      ut::buffer_error(cparams[0].C, cparams[i].C, _data.matRef.size(), 0.f);
    }
  }
};
#ifdef JBLAS_UT_TRANSFORMER
static UT_QKVGemmInterfacePackWeight sUT_QKVGemmInterfacePackWeight;
#endif
}  // namespace transformer
}  // namespace wrapper
}  // namespace jblas
