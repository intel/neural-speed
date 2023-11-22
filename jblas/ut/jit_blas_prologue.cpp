#include "../jit_blas_prologue.h"
#include <cassert>
#include "jblas/jit_blas.h"
#include "jblas/jit_blas_utils.h"
#include "jit_blas_ut.h"

using namespace jblas::utils;
namespace jblas {
namespace ut {
class UT_ActivationBase {
 public:
  UT_ActivationBase() {
    UT_START();
    ut<gemm::GemmCore_Row_NN_8x48_AVX512F>(8, 3, 128);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512F>(8, 48, 128);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512F>(8, 3, 128);
  }
  template <typename _T>
  void ut(int m, int k, int lda) {
    int kpad = padto(k, _T::KTILE);
    using BType = typename _T::BType;
    printf("Test Case %dB NTile%d: %d %d %d %d \n", int(sizeof(BType)), _T::NTILE, m, k, lda, kpad);
    std::vector<BType> src(m * lda);
    std::vector<BType> dst(m * kpad), dstref(m * kpad);
    for (int i = 0; i < src.size(); i++) {
      src[i] = static_cast<BType>(i);
    }
    jblas::prologue::gemm::ActivationBase<_T, JblasNoSIMD> reorderref;
    jblas::prologue::gemm::ActivationBase<_T, JblasAVX512F> reorderavx512;
    auto dstrefptr = dstref.data();
    auto dstptr = dst.data();
    int dststride = 0;
    reorderref.getActivation(&dstrefptr, &dststride, {src.data(), lda}, m, k, 0, 0);
    GetCPUDevice();
    if (_cd->AVX512F()) {
      reorderavx512.getActivation(&dstptr, &dststride, {src.data(), lda}, m, k, 0, 0);
      ut::buffer_error(dst.data(), dstref.data(), dst.size());
    }
  }
};
#ifdef JBLAS_UT_PROLOGUE
static UT_ActivationBase sUT_ActivationBase;
#endif

class UT_ActivationConverter {
 public:
  UT_ActivationConverter() {
    UT_START();
    CheckISA(AMX_BF16);
    ut<float, gemm::GemmCore_Row_NN_16x64_AMX_BF16>(8, 3, 128);
    ut<float, gemm::GemmCore_Row_NN_16x64_AMX_BF16>(8, 48, 128);
    ut<utils::bf16, gemm::GemmCore_Row_NN_8x48_AVX512F>(8, 3, 128);
    ut<utils::bf16, gemm::GemmCore_Row_NN_8x48_AVX512F>(8, 48, 128);
  }

  template <typename SRC_T, typename _T>
  void ut(int m, int k, int lda) {
    using SrcType = SRC_T;
    using AType = typename _T::AType;
    int kpad = padto(k, _T::KTILE);
    printf("Test Case %dB NTile%d: %d %d %d %d \n", int(sizeof(AType)), _T::NTILE, m, k, lda, kpad);
    std::vector<SrcType> src(m * lda);
    std::vector<AType> dst(m * kpad), dstref(m * kpad);
    for (int i = 0; i < src.size(); i++) {
      src[i] = static_cast<SrcType>(float(i));
    }
    jblas::prologue::gemm::ActivationConverter<_T, JblasNoSIMD, SRC_T> reorderref;
    jblas::prologue::gemm::ActivationConverter<_T, JblasAVX512F, SRC_T> reorderavx512;
    auto dstptr = dstref.data();
    int dststride = 0;
    auto ret = reorderref.getActivation(&dstptr, &dststride, {src.data(), lda}, m, k, 0, 0);
    assert(ret == JblasSuccess);
    dstptr = dst.data();
    ret = reorderavx512.getActivation(&dstptr, &dststride, {src.data(), lda}, m, k, 0, 0);
    assert(ret == JblasSuccess);
    ut::buffer_error(dst.data(), dstref.data(), dst.size(), AType{0});
    aligned_vector<SrcType> revert(dst.size());
    for (size_t i = 0; i < revert.size(); i++) {
      revert[i] = utils::cast<AType, SrcType>(dst[i]);
    }
    for (size_t i = 0; i < src.size(); i++) {
      auto tmp = utils::cast<SrcType, AType>(src[i]);
      src[i] = utils::cast<AType, SrcType>(tmp);
    }
    buffer_error_2d(src.data(), revert.data(), m, k, lda, kpad);
  }
};
#ifdef JBLAS_UT_PROLOGUE
static UT_ActivationConverter sUT_ActivationConverter;
#endif

class UT_ActivationU8KBlockQuantize {
 public:
  UT_ActivationU8KBlockQuantize() {
    UT_START();
    ut<gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK, JblasAVX512F>(1, 4096, 4096, 128);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX512F>(2, 4096, 4096, 128);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX512F>(2, 4096, 4096, 32);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX512F>(2, 11040, 11040, 32);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX512F>(1024, 4096, 4096, 32);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX512F>(1024, 11040, 11040, 32);
    ut<gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK, JblasAVX2>(1, 4096, 4096, 128);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX2>(2, 4096, 4096, 128);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX2>(2, 4096, 4096, 32);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX2>(2, 11040, 11040, 32);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX2>(1024, 4096, 4096, 32);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, JblasAVX2>(1024, 11040, 11040, 32);
  }
  template <typename _T, JBLAS_ISA ISA>
  void ut(int m, int k, int lda, int kblock) {
    int kpad = padto(k, _T::KTILE);
    printf("Test Case NTile%d: %d %d %d %d %d \n", _T::KTILE, m, k, lda, kblock, kpad);
    int kcount = updiv(kpad, kblock);
    utils::aligned_vector<float> raw(m * lda), scales(m * kcount);
    ut::fill_buffer_randn(raw.data(), raw.size(), -1.f, 1.f);
    utils::aligned_vector<uint8_t> q, zp;
    q.resize(m * lda);
    zp.resize(m * kcount);
    kernel::ref::quantize_f32_u8_colblock(m, k, raw.data(), lda, q.data(), lda, scales.data(), kcount, zp.data(),
                                          kblock);
    jblas::prologue::gemm::ActivationF32U8KBlockQuantize<_T, ISA> actA;
    auto quanAct = actA.createStorage(m, k, kblock);
    avector<int8_t> bufA(quanAct.mSize);
    quanAct.assign(bufA.data());
    actA.quantize({raw.data(), lda, &quanAct}, m, k);

    ut::buffer_error(quanAct.template get<uint8_t>(), q.data(), q.size(), uint8_t(1));
    ut::buffer_error(quanAct.mZPtr, zp.data(), zp.size(), uint8_t(1));
  }
};
#ifdef JBLAS_UT_PROLOGUE
static UT_ActivationU8KBlockQuantize sUT_ActivationU8KBlockQuantize;
#endif

class UT_ActivationS8KBlockQuantize {
 public:
  UT_ActivationS8KBlockQuantize() {
    UT_START();
    ut<gemm::GemmCore_Row_NN_16x48_AMX_S8S8>(2, 11040, 32);
    ut<gemm::kblock::GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK>(1, 4096, 128);
    ut<gemm::GemmCore_Row_NN_16x48_AMX_S8S8>(2, 4096, 128);
    ut<gemm::GemmCore_Row_NN_16x48_AMX_S8S8>(2, 4096, 32);
    ut<gemm::GemmCore_Row_NN_16x48_AMX_S8S8>(1024, 4096, 32);
    ut<gemm::GemmCore_Row_NN_16x48_AMX_S8S8>(1024, 11040, 32);
  }

  template <typename _T>
  void ut(int m, int k, int kblock) {
    int kpad = padto(k, _T::KTILE);
    int lda = kpad;
    printf("Test Case NTile%d: %d %d %d %d %d \n", _T::KTILE, m, k, lda, kblock, kpad);
    int kcount = updiv(kpad, kblock);
    utils::aligned_vector<float> raw(m * k), scales(m * kcount);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.1f, 0.1f);
    utils::aligned_vector<int8_t> q;
    q.resize(m * lda);
    kernel::ref::quantize_f32_s8_colblock(m, k, raw.data(), k, q.data(), lda, scales.data(), kcount, kblock);
    jblas::prologue::gemm::ActivationF32S8KBlockQuantize<_T, JblasAVX512F> actA;
    auto quanAct = actA.createStorage(m, k, kblock);
    avector<int8_t> bufA(quanAct.mSize);
    quanAct.assign(bufA.data());
    actA.quantize({raw.data(), k, &quanAct}, m, k);
    ut::buffer_error(quanAct.template get<int8_t>(), q.data(), q.size(), int8_t(1));
  }
};
#ifdef JBLAS_UT_PROLOGUE
static UT_ActivationS8KBlockQuantize sUT_ActivationS8KBlockQuantize;
#endif

class UT_ActivationU8PerChannelNQuantize {
 public:
  UT_ActivationU8PerChannelNQuantize() {
    UT_START();
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(2, 4096);
    ut<gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK>(1, 4096);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(2, 4096);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(2, 11040);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(1024, 4096);
    ut<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(1024, 11040);
    ut_ws<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(1024, 4096);
    ut_ws<gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(1024, 11040);
  }

  template <typename _T>
  void ut(int m, int k) {
    int kpad = padto(k, _T::KTILE);
    int lda = kpad;
    printf("Test Case %s NTile%d: %d %d %d %d\n", __FUNCTION__, _T::KTILE, m, k, lda, kpad);
    utils::aligned_vector<float> raw(m * k), scales(m);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.1f, 0.1f);
    utils::aligned_vector<uint8_t> q, zp;
    q.resize(m * lda);
    zp.resize(m);
    kernel::ref::quantize_f32_u8_colblock(m, k, raw.data(), k, q.data(), lda, scales.data(), 1, zp.data(), k);
    jblas::prologue::gemm::ActivationFp32AsymU8Quantize<_T, JblasAVX512F> actA;
    auto quan = actA.createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto ret = actA.quantize({raw.data(), k, &quan}, m, k);
    assert(ret == JblasSuccess);
    (void)ret;
    ut::buffer_error(quan.template get<uint8_t>(), q.data(), q.size(), uint8_t(1));
  }

  template <typename _T>
  void ut_ws(int m, int k) {
    int kpad = padto(k, _T::KTILE);
    int lda = kpad;
    printf("Test Case %s NTile%d: %d %d %d %d\n", __FUNCTION__, _T::KTILE, m, k, lda, kpad);
    utils::aligned_vector<float> raw(m * k), scales(m);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.1f, 0.1f);
    utils::aligned_vector<uint8_t> q, zp;
    q.resize(m * lda);
    zp.resize(m);
    kernel::ref::quantize_f32_u8_colblock(m, k, raw.data(), k, q.data(), lda, scales.data(), 1, zp.data(), k);
    jblas::prologue::gemm::ActivationFp32AsymU8Quantize<_T, JblasAVX512F> actA;
    auto quan = actA.createStorage(m, k);
    avector<int8_t> ws(quan.mSize);
    quan.assign(ws.data());
    auto ret = actA.quantize({raw.data(), k, &quan}, m, k);
    auto quantmp = actA.createStorage(1, 1);
    quantmp.deserialize(ws.data());
    assert(ret == JblasSuccess);
    (void)ret;
    ut::buffer_error(quan.template get<uint8_t>(), q.data(), q.size(), uint8_t(1));
  }
};
#ifdef JBLAS_UT_PROLOGUE
static UT_ActivationU8PerChannelNQuantize sUT_ActivationU8PerChannelNQuantize;
#endif
}  // namespace ut
}  // namespace jblas
