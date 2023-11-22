#include "../jit_blas_gemm.h"

#include "../jit_blas_utils.h"
#include "jit_blas_ut.h"

namespace jblas {
using namespace utils;

template <int NTILE>
void ref_bf16(utils::bf16* matA, utils::bf16* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride,
              int _cstride, int kpos) {
  int lda = _astride / sizeof(utils::bf16);
  int ldb = _bstride / sizeof(utils::bf16);
  int ldc = _cstride / sizeof(float);
  int constexpr KPack = 4 / sizeof(utils::bf16);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        float tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<utils::bf16, float>(utils::bf16{matA[i * lda + k + ik]});
            auto tmpB = utils::cast<utils::bf16, float>(utils::bf16{matB[k * NTILE + ij * KPack + ik + j * ldb]});
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE>
void ref_fp32(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
              int kpos) {
  int lda = _astride / sizeof(float);
  int ldb = _bstride / sizeof(float);
  int ldc = _cstride / sizeof(float);
  int constexpr KPack = 4 / sizeof(float);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        float tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = matA[i * lda + k + ik];
            auto tmpB = matB[k * NTILE + ij * KPack + ik + j * ldb];
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE, class T_A_, class T_B_>
void ref_int8(T_A_* matA, T_B_* matB, int32_t* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
              int kpos) {
  int lda = _astride / sizeof(T_A_);
  int ldb = _bstride / sizeof(T_B_);
  int ldc = _cstride / sizeof(int32_t);
  int constexpr KPack = 4 / sizeof(T_B_);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          continue;
        }
        int32_t tmp = 0;
        for (int k = 0; k < _k; k += KPack) {
          for (int ik = 0; ik < KPack; ik++) {
            if (k + ik >= _k) {
              continue;
            }
            auto tmpA = utils::cast<T_A_, int32_t>(matA[i * lda + k + ik]);
            auto tmpB = utils::cast<T_B_, int32_t>(matB[k * NTILE + ij * KPack + ik + j * ldb]);
            tmp += tmpA * tmpB;
          }
        }
        matC[i * ldc + j + ij] = tmp;
      }
    }
  }
}

template <int NTILE, typename T_SB_>
void ref_kblock_int8(uint8_t* matA, int8_t* matB, float* matC, uint8_t* zpA, float* scaleA, int _ldsa, T_SB_* scaleB,
                     float* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride,
                     int _cstride, int kpos) {
  int lda = _astride / sizeof(matA[0]);
  int ldb = _bstride / sizeof(matB[0]);
  int ldc = _cstride / sizeof(matC[0]);
  for (int i = 0; i < _m; i++) {
    for (int j = 0; j < _n; j += NTILE) {
      for (int ij = 0; ij < NTILE; ij++) {
        if (j + ij >= _n) {
          break;
        }
        float tmpf = 0.f;
        for (int k = 0; k < _k; k += _kblock) {
          int tmp = 0;
          int zpval = int(zpA[i * _ldsa + k / _kblock]);
          for (int ik = 0; ik < _kblock; ik += 4) {
            if (k + ik >= _k) {
              break;
            }
            for (int ikk = 0; ikk < 4; ikk++) {
              tmp += (int(matA[i * lda + k + ik + ikk])) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
            }
          }
          tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * float(scaleB[j + ij + k / _kblock * _ldsb]);
          tmpf -= zpval * scaleA[i * _ldsa + k / _kblock] * reduceB[j + ij + k / _kblock * _ldsb];
        }
        matC[i * ldc + j + ij] = tmpf;
      }
    }
  }
}
namespace ut {
class UT_GEMM_AVX512F {
 public:
  UT_GEMM_AVX512F() {
    UT_START();
    CheckISA(AVX512F);
    ut(8, 48, 2, 2, 48, 48);
    ut(1, 48, 2, 2, 48, 48);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    gemm::GemmCore_Row_NN_8x48_AVX512F gemm;
    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, 0);
    _data.calc_ref(1.0f, 0.f);

    gemm.forward(_data.matA.data(), _data.matB.data(), _data.matC.data(), m, n, k, lda * 4, ldb * 4, ldc * 4, 0);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512F sUT_GEMM_AVX512F;
#endif

class UT_GEMM_4x24_AVX2 {
 public:
  UT_GEMM_4x24_AVX2() {
    UT_START();
    CheckISA(AVX2);
    ut(4, 24, 2, 2, 24, 24);
    ut(1, 24, 2, 2, 24, 24);
    ut(3, 24, 101, 101, 24, 24);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    gemm::GemmCore_Row_NN_4x24_AVX2 gemm;
    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, 0);
    _data.calc_ref(1.0f, 0.f);

    gemm.forward(_data.matA.data(), _data.matB.data(), _data.matC.data(), m, n, k, lda * 4, ldb * 4, ldc * 4, 0);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_4x24_AVX2 sUT_GEMM_4x24_AVX2;
#endif

class UT_GEMM_2x48_AVX2 {
 public:
  UT_GEMM_2x48_AVX2() {
    UT_START();
    CheckISA(AVX2);
    ut(1, 48, 2, 2, 48, 48);
    ut(2, 48, 101, 101, 48, 48);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    gemm::GemmCore_Row_NN_2x48_AVX2 gemm;
    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, 0);
    _data.calc_ref(1.0f, 0.f);

    gemm.forward(_data.matA.data(), _data.matB.data(), _data.matC.data(), m, n, k, lda * 4, ldb * 4, ldc * 4, 0);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_2x48_AVX2 sUT_GEMM_2x48_AVX2;
#endif

class UT_GemmCore_Row_NN_AVX512_FP16 {
 public:
  UT_GemmCore_Row_NN_AVX512_FP16() {
    UT_START();
    CheckISA(AVX512_FP16);

    ut(4, 64, 2, 2, 64, 64);
    ut(1, 64, 2, 2, 64, 64);
    ut(3, 64, 101, 101, 64, 64);

    ut_96(4, 96, 2, 2, 96, 96);
    ut_96(1, 96, 2, 2, 96, 96);
    ut_96(3, 96, 101, 101, 96, 96);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    gemm::GemmCore_Row_NN_8x64_AVX512_FP16 gemm;
    ut::UT_GEMMData_Row_fp16 _data(m, n, k, lda, ldb, ldc, 0);
    _data.calc_ref(1.0f, 0.f);
    aligned_vector<fp16> ret(_data.matC.size());
    gemm.forward(_data.matA.data(), _data.matB.data(), ret.data(), m, n, k, lda * sizeof(fp16), ldb * sizeof(fp16),
                 ldc * sizeof(fp16), 0);
    ut::buffer_error(_data.matC.data(), ret.data(), _data.matC.size(), fp16(0.001f));
    printf("\n");
  }

  void ut_96(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    gemm::GemmCore_Row_NN_8x96_AVX512_FP16 gemm;
    ut::UT_GEMMData_Row_fp16 _data(m, n, k, lda, ldb, ldc, 0);
    _data.calc_ref(1.0f, 0.f);
    aligned_vector<fp16> ret(_data.matC.size());
    gemm.forward(_data.matA.data(), _data.matB.data(), ret.data(), m, n, k, lda * sizeof(fp16), ldb * sizeof(fp16),
                 ldc * sizeof(fp16), 0);
    ut::buffer_error(_data.matC.data(), ret.data(), _data.matC.size(), fp16(0.001f));
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GemmCore_Row_NN_AVX512_FP16 sUT_GemmCore_Row_NN_AVX512_FP16;
#endif

class UT_GEMM_AMX_BF16 {
 public:
  UT_GEMM_AMX_BF16() {
    UT_START();
    GetCPUDevice();
    if (!_cd->AMX_BF16()) {
      printf("Error Device\n");
      return;
    }
    request_perm_xtile_data();
    ut(16, 64, 64, 64, 64, 64);
    ut(16, 64, 32, 32, 64, 64);
    ut(16, 64, 16, 16, 64, 64);
    ut(1, 64, 16, 16, 64, 64);
    ut(1, 16, 2, 2, 16, 16);

    ut_48(16, 48, 64, 64, 48, 48);
    ut_48(16, 48, 32, 32, 48, 48);
    ut_48(16, 48, 16, 16, 48, 48);
    ut_48(1, 48, 16, 16, 48, 48);
    ut_48(1, 16, 2, 2, 16, 16);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("%s: %d %d %d %d %d %d\n", __FUNCTION__, m, n, k, lda, ldb, ldc);

    gemm::GemmCore_Row_NN_16x64_AMX_BF16 gemm;
    ut::UT_GEMMData_Row_bf16 bf16(m, n, k, lda, ldb, ldc, 0);
    utils::aligned_vector<float> resC(m * ldc);
    aligned_vector<float> refC(m * ldc, 0);
    int reordered_bstride = k * 2;
    ref_bf16<gemm::GemmCore_Row_NN_16x64_AMX_BF16::NTILE>(bf16.matA.data(), bf16.matB.data(), refC.data(), m, n, k,
                                                          lda * 2, reordered_bstride, ldc * 4, 0);
    gemm.forward(bf16.matA.data(), bf16.matB.data(), resC.data(), m, n, k, lda * 2, reordered_bstride, ldc * 4, 0);
    ut::buffer_error(refC.data(), resC.data(), resC.size(), 0.001f);
    printf("\n");
  }

  void ut_48(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("%s: %d %d %d %d %d %d\n", __FUNCTION__, m, n, k, lda, ldb, ldc);
    gemm::GemmCore_Row_NN_16x48_AMX_BF16 gemm;
    ut::UT_GEMMData_Row_bf16 bf16(m, n, k, lda, ldb, ldc, 0);
    utils::aligned_vector<float> resC(m * ldc);
    aligned_vector<float> refC(m * ldc, 0);
    int reordered_bstride = k * 2;
    ref_bf16<gemm::GemmCore_Row_NN_16x48_AMX_BF16::NTILE>(bf16.matA.data(), bf16.matB.data(), refC.data(), m, n, k,
                                                          lda * 2, reordered_bstride, ldc * 4, 0);
    gemm.forward(bf16.matA.data(), bf16.matB.data(), resC.data(), m, n, k, lda * 2, reordered_bstride, ldc * 4, 0);
    ut::buffer_error(refC.data(), resC.data(), resC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AMX_BF16 sUT_GEMM_AMX_BF16;
#endif

class UT_GEMM_AVX512_VNNI {
 public:
  UT_GEMM_AVX512_VNNI() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut(8, 48, 4, 4, 48, 48);
    ut(1, 48, 4, 4, 48, 48);
    ut(8, 144, 4, 4, 4, 144);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    gemm::GemmCore_Row_NN_8x48_AVX512_VNNI gemm;
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, 0, n, true);
    aligned_vector<int32_t> matRef(m * ldc, 0);
    aligned_vector<int32_t> matC(m * ldc, 0);
    ref_int8<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI::NTILE>(_quan.matA.data(), _quan.matB.data(), matRef.data(), m, n,
                                                            k, lda, ldb, ldc * 4, 0);
    gemm.forward(_quan.matA.data(), _quan.matB.data(), matC.data(), m, n, k, lda, ldb, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size());
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512_VNNI sUT_GEMM_AVX512_VNNI;
#endif

class UT_GEMM_AVX_VNNI {
 public:
  UT_GEMM_AVX_VNNI() {
    UT_START();
    CheckISA(AVX_VNNI);
    ut(1, 48, 4, 4, 48, 48);
    ut(2, 144, 4, 4, 4, 144);
    ut(1, 48, 20, 4, 48, 48);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d\n", m, n, k, lda, ldc);
    gemm::GemmCore_Row_NN_2x48_AVX_VNNI gemm;
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, 0, n, true);
    aligned_vector<int32_t> matRef(m * ldc, 0);
    aligned_vector<int32_t> matC(m * ldc, 0);
    ref_int8<gemm::GemmCore_Row_NN_2x48_AVX_VNNI::NTILE>(_quan.matA.data(), _quan.matB.data(), matRef.data(), m, n, k,
                                                         lda, ldb, ldc * 4, 0);
    gemm.forward(_quan.matA.data(), _quan.matB.data(), matC.data(), m, n, k, lda, ldb, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size());
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX_VNNI sUT_GEMM_AVX_VNNI;
#endif

class UT_GEMM_16x64_AMX_INT8 {
 public:
  UT_GEMM_16x64_AMX_INT8() {
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    ut(8, 48, 4, 4, 48, 48);
    ut(1, 48, 4, 4, 48, 48);
    ut(8, 144, 4, 4, 144, 144);
    ut(1, 16, 4, 4, 16, 16);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d %d\n", m, n, k, lda, ldb, ldc);
    gemm::GemmCore_Row_NN_16x64_AMX_U8S8 gemm;
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, 0, n, true);
    int reordered_bstride = k;
    aligned_vector<int32_t> matRef(m * ldc, 0);
    aligned_vector<int32_t> matC(m * ldc, 0);
    ref_int8<gemm::GemmCore_Row_NN_16x64_AMX_U8S8::NTILE>(_quan.matA.data(), _quan.matB.data(), matRef.data(), m, n, k,
                                                          lda, reordered_bstride, ldc * 4, 0);
    gemm.forward(_quan.matA.data(), _quan.matB.data(), matC.data(), m, n, k, lda, reordered_bstride, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size());
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_16x64_AMX_INT8 sUT_GEMM_16x64_AMX_INT8;
#endif

class UT_GEMM_16x48_AMX_INT8 {
 public:
  UT_GEMM_16x48_AMX_INT8() {
    UT_START();
    CheckISA(AMX_INT8);

    request_perm_xtile_data();
    ut(8, 48, 4, 4, 48, 48);
    ut(1, 48, 4, 4, 48, 48);
    ut(8, 64, 4, 4, 64, 64);
    ut(1, 64, 4, 4, 64, 64);
    ut(8, 144, 4, 4, 144, 144);
    ut(1, 16, 4, 4, 16, 16);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d %d\n", m, n, k, lda, ldb, ldc);

    gemm::GemmCore_Row_NN_16x48_AMX_U8S8 gemm;
    ut::UT_GEMMData_Row_u8s8 _quan(m, n, k, lda, ldb, ldc, 0, n, true);
    int reordered_bstride = k;
    aligned_vector<int32_t> matRef(m * ldc, 0);
    aligned_vector<int32_t> matC(m * ldc, 0);
    ref_int8<gemm::GemmCore_Row_NN_16x48_AMX_U8S8::NTILE>(_quan.matA.data(), _quan.matB.data(), matRef.data(), m, n, k,
                                                          lda, reordered_bstride, ldc * 4, 0);
    gemm.forward(_quan.matA.data(), _quan.matB.data(), matC.data(), m, n, k, lda, reordered_bstride, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size());
    printf("\n");
  }
};
#if defined(JBLAS_UT_GEMM)
static UT_GEMM_16x48_AMX_INT8 sUT_GEMM_16x48_AMX_INT8;
#endif

class UT_GEMM_16x48_AMX_S8S8 {
 public:
  UT_GEMM_16x48_AMX_S8S8() {
    UT_START();
    CheckISA(AMX_INT8);

    request_perm_xtile_data();
    ut(8, 48, 4, 4, 48, 48);
    ut(1, 48, 4, 4, 48, 48);
    ut(8, 64, 4, 4, 64, 64);
    ut(1, 64, 4, 4, 64, 64);
    ut(8, 144, 4, 4, 144, 144);
    ut(1, 16, 4, 4, 16, 16);
  }

  void ut(int m, int n, int k, int lda, int ldb, int ldc) {
    printf("Test Case: %d %d %d %d %d %d\n", m, n, k, lda, ldb, ldc);

    gemm::GemmCore_Row_NN_16x48_AMX_S8S8 gemm;
    aligned_vector<int8_t> matA(m * lda), matB(k * ldb);
    aligned_vector<int32_t> matC(m * ldc, 0), matRef(m * ldc, 0);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    int reordered_bstride = k;
    ref_int8<gemm::GemmCore_Row_NN_16x48_AMX_S8S8::NTILE>(matA.data(), matB.data(), matRef.data(), m, n, k, lda,
                                                          reordered_bstride, ldc * 4, 0);
    gemm.forward(matA.data(), matB.data(), matC.data(), m, n, k, lda, reordered_bstride, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size());
    printf("\n");
  }
};
#if defined(JBLAS_UT_GEMM)
static UT_GEMM_16x48_AMX_S8S8 sUT_GEMM_16x48_AMX_S8S8;
#endif

class UT_GEMM_AVX512_VNNI_KBLOCK {
 public:
  using GemmCore = gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK;
  UT_GEMM_AVX512_VNNI_KBLOCK() {
    UT_START();
    CheckISA(AVX512_VNNI);

    ut_f32_f32(1, 16, 8, 32);
    ut_f32_f32(1, 64, 8, 32);
    ut_f32_f32(1, 48, 8, 32);
    ut_f32_f32(2, 48, 32, 32);
    ut_f32_f32(2, 48, 128, 32);
    ut_f32_f32(2, 48, 1024, 128);
    ut_f32_bf16(1, 16, 8, 32);
  }

  void ut_f32_f32(int m, int n, int k, int kblock) {
    printf("Test Case %s: %d %d %d %d\n", __FUNCTION__, m, n, k, kblock);
    GemmCore gemm;
    int kblk = updiv(k, kblock);
    int npad = padto(n, gemm.NTILE);
    int lda = k, ldb = k, ldc = npad;
    aligned_vector<float> matRef(m * ldc, 0);
    aligned_vector<float> matC(m * ldc, 0);
    aligned_vector<int8_t> matB(k * npad);
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    aligned_vector<uint8_t> matA(m * k), zpA(m * kblk);
    fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(70), uint8_t(155));
    aligned_vector<float> scaleA(m * kblk), scaleB(kblk * npad);
    fill_buffer_randn(scaleA.data(), scaleA.size(), float(0.01f), float(0.03f));
    fill_buffer_randn(scaleB.data(), scaleB.size(), float(0.01f), float(0.03f));
    int ldsa = kblk;
    int ldsb = npad;

    gemm.reference(matA.data(), matB.data(), matRef.data(), zpA.data(), scaleA.data(), ldsa, scaleB.data(), ldsb, m, n,
                   k, kblock, lda, ldb, ldc * 4, 0);
    gemm.forward(matA.data(), matB.data(), matC.data(), zpA.data(), scaleA.data(), ldsa, scaleB.data(), ldsb, m, n, k,
                 kblock, lda, ldb, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size(), 0.001f);
    printf("\n");
  }

  void ut_f32_bf16(int m, int n, int k, int kblock) {
    printf("Test Case %s: %d %d %d %d\n", __FUNCTION__, m, n, k, kblock);
    GemmCore gemm;
    int kblk = updiv(k, kblock);
    int npad = padto(n, gemm.NTILE);
    int lda = k, ldb = k, ldc = npad;
    aligned_vector<float> matRef(m * ldc, 0);
    aligned_vector<float> matC(m * ldc, 0);
    aligned_vector<int8_t> matB(k * npad);
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    aligned_vector<uint8_t> matA(m * k), zpA(m * kblk);
    fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(70), uint8_t(155));
    aligned_vector<float> scaleA(m * kblk), scaleB(kblk * npad);
    fill_buffer_randn(scaleA.data(), scaleA.size(), float(0.01f), float(0.03f));
    fill_buffer_randn(scaleB.data(), scaleB.size(), float(0.01f), float(0.03f));
    aligned_vector<bf16> scaleBb(kblk * npad);

    for (size_t i = 0; i < scaleB.size(); i++) {
      scaleBb[i].fromfloat(scaleB[i]);
    }
    int ldsa = kblk;
    int ldsb = npad;

    gemm.reference(matA.data(), matB.data(), matRef.data(), zpA.data(), scaleA.data(), ldsa, scaleBb.data(), ldsb, m, n,
                   k, kblock, lda, ldb, ldc * 4, 0);
    gemm.forward(matA.data(), matB.data(), matC.data(), zpA.data(), scaleA.data(), ldsa, scaleBb.data(), ldsb, m, n, k,
                 kblock, lda, ldb, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512_VNNI_KBLOCK sUT_GEMM_AVX512_VNNI_KBLOCK;
#endif

class UT_GEMM_AVX512_VNNI_KBLOCK_New {
 public:
  using GemmCore = gemm::kblock::GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK;
  UT_GEMM_AVX512_VNNI_KBLOCK_New() {
    UT_START();
    CheckISA(AVX512_VNNI);

    ut_f32_f32(1, 48, 8, 32);
    ut_f32_f32(2, 48, 32, 32);
    ut_f32_f32(2, 48, 128, 32);
    ut_f32_f32(2, 48, 1024, 128);
  }

  void ut_f32_f32(int m, int n, int k, int kblock) {
    printf("Test Case %s: %d %d %d %d\n", __FUNCTION__, m, n, k, kblock);
    GemmCore gemm;
    int kblk = updiv(k, kblock);
    int npad = padto(n, gemm.NTILE);
    int lda = k, ldb = k, ldc = npad;
    aligned_vector<float> matRef(m * ldc, 0);
    aligned_vector<float> matC(m * ldc, 0);
    aligned_vector<int8_t> matB(k * npad);
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    aligned_vector<uint8_t> matA(m * k), zpA(m * kblk);
    fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(100), uint8_t(155));
    aligned_vector<float> scaleA(m * kblk), scaleB(kblk * npad);
    fill_buffer_randn(scaleA.data(), scaleA.size(), float(0.01f), float(0.03f));
    fill_buffer_randn(scaleB.data(), scaleB.size(), float(0.01f), float(0.03f));
    avector<float> reduceB(npad * kblk);
    for (int i = 0; i < npad; i++) {
      for (int j = 0; j < k; j += kblock) {
        float red = 0.f;
        int kidx = j / kblock;
        for (int jj = 0; jj < kblock; jj++) {
          if (jj + j < k) {
            int iidx = jj % 4;
            int i1idx = (jj + j) / 4;
            red += float(matB[i1idx * npad * 4 + iidx + i * 4]) * scaleB[kidx * npad + i];
          }
        }
        reduceB[kidx * npad + i] = red;
      }
    }
    int ldsa = kblk;
    int ldsb = npad;

    ref_kblock_int8<GemmCore::NTILE>(matA.data(), matB.data(), matRef.data(), zpA.data(), scaleA.data(), ldsa,
                                     scaleB.data(), reduceB.data(), ldsb, m, n, k, kblock, lda, ldb, ldc * 4, 0);
    gemm.forward(matA.data(), matB.data(), matC.data(), zpA.data(), scaleA.data(), ldsa, scaleB.data(), reduceB.data(),
                 ldsb, m, n, k, kblock, lda, ldb, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX512_VNNI_KBLOCK_New sUT_GEMM_AVX512_VNNI_KBLOCK_New;
#endif

class UT_GEMM_AVX_VNNI_KBLOCK_New {
 public:
  using GemmCore = gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK;
  UT_GEMM_AVX_VNNI_KBLOCK_New() {
    UT_START();
    CheckISA(AVX_VNNI);
    ut_f32_f32(1, 48, 8, 32);
    ut_f32_f32(1, 48, 32, 32);
    ut_f32_f32(1, 48, 128, 32);
    ut_f32_f32(1, 48, 1024, 128);
  }

  void ut_f32_f32(int m, int n, int k, int kblock) {
    printf("Test Case %s: %d %d %d %d\n", __FUNCTION__, m, n, k, kblock);
    GemmCore gemm;
    int kblk = updiv(k, kblock);
    int npad = padto(n, gemm.NTILE);
    int lda = k, ldb = k, ldc = npad;
    aligned_vector<float> matRef(m * ldc, 0);
    aligned_vector<float> matC(m * ldc, 0);
    aligned_vector<int8_t> matB(k * npad);
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    aligned_vector<uint8_t> matA(m * k), zpA(m * kblk);
    fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(100), uint8_t(155));
    aligned_vector<float> scaleA(m * kblk), scaleB(kblk * npad);
    fill_buffer_randn(scaleA.data(), scaleA.size(), float(0.01f), float(0.03f));
    fill_buffer_randn(scaleB.data(), scaleB.size(), float(0.01f), float(0.03f));
    avector<float> reduceB(npad * kblk);
    for (int i = 0; i < npad; i++) {
      for (int j = 0; j < k; j += kblock) {
        float red = 0.f;
        int kidx = j / kblock;
        for (int jj = 0; jj < kblock; jj++) {
          if (jj + j < k) {
            int iidx = jj % 4;
            int i1idx = (jj + j) / 4;
            red += float(matB[i1idx * npad * 4 + iidx + i * 4]) * scaleB[kidx * npad + i];
          }
        }
        reduceB[kidx * npad + i] = red;
      }
    }
    int ldsa = kblk;
    int ldsb = npad;

    ref_kblock_int8<GemmCore::NTILE>(matA.data(), matB.data(), matRef.data(), zpA.data(), scaleA.data(), ldsa,
                                     scaleB.data(), reduceB.data(), ldsb, m, n, k, kblock, lda, ldb, ldc * 4, 0);
    gemm.forward(matA.data(), matB.data(), matC.data(), zpA.data(), scaleA.data(), ldsa, scaleB.data(), reduceB.data(),
                 ldsb, m, n, k, kblock, lda, ldb, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AVX_VNNI_KBLOCK_New sUT_GEMM_AVX_VNNI_KBLOCK_New;
#endif

class UT_GEMM_AMXINT8_48_KBLOCK {
 public:
  UT_GEMM_AMXINT8_48_KBLOCK() {
    UT_START();
    CheckISA(AMX_INT8);

    request_perm_xtile_data();
    ut_f32_f32(1, 48, 128, 128);
    ut_f32_f32(2, 48, 128, 128);
    ut_f32_f32(16, 48, 256, 128);
  }

  void ut_f32_f32(int m, int n, int k, int kblock) {
    printf("Test Case %s: %d %d %d %d\n", __FUNCTION__, m, n, k, kblock);
    gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK gemm;
    int kblk = updiv(k, kblock);
    int npad = padto(n, gemm.NTILE);
    int lda = k, ldb = k, ldc = npad;
    aligned_vector<float> matRef(m * ldc, 0);
    aligned_vector<float> matC(m * ldc, 0);
    aligned_vector<int8_t> matB(k * npad);
    fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    aligned_vector<int8_t> matA(m * k);
    fill_buffer_randn(matA.data(), matA.size(), int8_t(-127), int8_t(127));
    aligned_vector<float> scaleA(m * kblk), scaleB(kblk * npad);
    fill_buffer_randn(scaleA.data(), scaleA.size(), float(0.01f), float(0.03f));
    fill_buffer_randn(scaleB.data(), scaleB.size(), float(0.01f), float(0.03f));
    int ldsa = kblk;
    int ldsb = npad;

    gemm.reference(matA.data(), matB.data(), matRef.data(), scaleA.data(), ldsa, scaleB.data(), ldsb, m, n, k, kblock,
                   lda, ldb, ldc * 4, 0);
    gemm.forward(matA.data(), matB.data(), matC.data(), NULL, scaleA.data(), ldsa, scaleB.data(), ldsb, m, n, k, kblock,
                 lda, ldb, ldc * 4, 0);
    ut::buffer_error(matRef.data(), matC.data(), matC.size(), 0.001f);
    printf("\n");
  }
};
#ifdef JBLAS_UT_GEMM
static UT_GEMM_AMXINT8_48_KBLOCK sUT_GEMM_AMXINT8_48_KBLOCK;
#endif
}  // namespace ut
}  // namespace jblas
