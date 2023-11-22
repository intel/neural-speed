#include "../jit_blas_weight_compression.h"
#include "jblas/jit_blas.h"
#include "jit_blas_ut.h"

namespace jblas {
using namespace utils;
using CompType = jblas::prologue::weight_comp::gemm_kblcok::PrologueBIDs;
namespace wrapper {
namespace gemm {
class UT_BlockQunatize_INT8 {
 public:
  UT_BlockQunatize_INT8() {
    UT_START();
    CheckISA(AVX512F);
    ut(1024, 1024, 32);
    ut(1024, 1024, 32, true);
    ut(4128, 4096, 32);
    ut(4128, 4096, 32, true);
    ut(1024, 4096, 32);
    ut(4096, 1024, 32);

    ut_transpose(4096, 4096, 32);
    ut_transpose(4096, 4096, 32, true);
    ut_transpose(4128, 4096, 32);
    ut_transpose(4128, 4096, 32, true);
    ut_transpose(1024, 4096, 32);
    ut_transpose(4096, 1024, 32);
  }

  void ut(int n, int k, int blocksize, bool asym = false) {
    printf("%s: %d %d %d %s\n", __FUNCTION__, n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.003f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;  // make sure each block has maximum value to quantize
        }
        if (i % blocksize == 1 && asym) {
          quanW.data()[i * n + j] = -128;  // make sure each block has minimum value to quantize if asym
        }
      }
    }

    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        if (asym) {
          dequanRef[j * ldb + i] = (float(quanW.data()[j * ldb + i]) - float(zero_points[j / blocksize * n + i])) *
                                   scales[j / blocksize * n + i];
        } else {
          dequanRef[j * ldb + i] = float(quanW.data()[j * ldb + i]) * scales[j / blocksize * n + i];
        }
      }
    }

    auto constexpr RuntimeISA = JblasAVX512F;
    using PrologueB =
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                     RuntimeISA>;
    PrologueB kernel;
    auto ptr = kernel.createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    kernel.packWeight(n, k, dequanRef.data(), ldb, &ptr);
    avector<float> dequant(n * k);
    kernel.unpackWeight(n, k, &ptr, dequant.data(), n);
    avector<int8_t> ws8(n * k);
    kernel.unpackWeight(n, k, &ptr, ws8.data(), n);
    ut::buffer_error(quanW.data(), ws8.data(), ws8.size(), (int8_t)1);
    ut::buffer_error(dequanRef.data(), dequant.data(), dequanRef.size(), 0.01f);
  }

  void ut_transpose(int n, int k, int blocksize, bool asym = false) {
    printf("%s: %d %d %d %s\n", __FUNCTION__, n, k, blocksize, asym ? "asym" : "sym");
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.003f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;  // make sure each block has maximum value to quantize
        }
        if (i % blocksize == 1 && asym) {
          quanW.data()[i * n + j] = -128;  // make sure each block has minimum value to quantize if asym
        }
      }
    }

    avector<float> dequanT(k * n);
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        if (asym) {
          dequanRef[j * n + i] = (float(quanW.data()[j * n + i]) - float(zero_points[j / blocksize * n + i])) *
                                 scales[j / blocksize * n + i];
        } else {
          dequanRef[j * n + i] = float(quanW.data()[j * n + i]) * scales[j / blocksize * n + i];
        }
        dequanT[j + i * k] = dequanRef[j * n + i];
      }
    }

    auto constexpr RuntimeISA = JblasAVX512F;
    using PrologueB =
        jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                     RuntimeISA>;
    PrologueB kernel;
    auto ptr = kernel.createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    kernel.packTransposeWeight(n, k, dequanT.data(), k, &ptr);
    avector<float> dequant(n * k), tardequanT(k * n);
    kernel.unpackWeight(n, k, &ptr, dequant.data(), n);
    kernel.unpackTransposeWeight(n, k, &ptr, tardequanT.data(), k);
    ut::buffer_error(dequanT.data(), tardequanT.data(), tardequanT.size(), 0.01f);
    avector<int8_t> ws8(n * k);
    kernel.unpackWeight(n, k, &ptr, ws8.data(), n);
    ut::buffer_error(quanW.data(), ws8.data(), ws8.size(), (int8_t)1);
    ut::buffer_error(dequanRef.data(), dequant.data(), dequanRef.size(), 0.01f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_BlockQunatize_INT8 sUT_BlockQunatize_INT8;
#endif

class UT_TransposeBlockQuantize_F4 {
 public:
  UT_TransposeBlockQuantize_F4() {
    UT_START();
    CheckISA(AVX512F);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(4096, 4096, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(1024, 4096, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(4096, 1024, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(48, 32, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(32, 32, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(48, 32, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(48, 32, 32);
  }

  template <template <class _GemmCore_T, JBLAS_ISA ISA_T> class Wei, JBLAS_F4_TYPE F4_T>
  void ut(int n, int k, int blocksize) {
    printf("Test Case: %d %d %d\n", n, k, blocksize);
    int ldb = n;
    utils::aligned_vector<float> dequanRef(n * k);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 1.f, 5.f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(0, 16);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          switch (F4_T) {
            case FP4_E2M1:
              quanW.data()[i * n + j] = 7;  // make sure each block has maximum fp4e2m1 value(0b111) to quantize
              break;
            case FP4_BNB:
              quanW.data()[i * n + j] = 3;  // make sure each block has maximum fp4bnb value(0b011) to quantize
              break;
            case NF4:
              quanW.data()[i * n + j] = 15;  // make sure each block has maximum nf4 value(0b1111) to quantize
              break;
            default:
              break;
          }
        }
      }
    }

    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        dequanRef[j + i * k] =
            jblas::kernel::ref::f4_dequantize<F4_T>(quanW.data()[j * ldb + i], scales[j / blocksize * n + i]);
        quanW.data()[j * ldb + i] =
            jblas::kernel::ref::f4_quantize<F4_T>(dequanRef[j + i * k] / scales[j / blocksize * n + i]);
      }
    }

    auto constexpr RuntimeISA = JblasAVX512F;
    using PrologueB = Wei<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, RuntimeISA>;
    PrologueB kernel;
    auto packedW = kernel.createStorage(n, k, blocksize);
    auto packedW1 = kernel.createStorage(n, k, blocksize);
    avector<int8_t> buf(packedW.mSize), buf1(packedW1.mSize);
    packedW.assign(buf.data());
    packedW1.assign(buf1.data());
    kernel.packTransposeWeight(n, k, dequanRef.data(), k, &packedW);
    kernel.packQWeight(n, k, quanW.data(), ldb, scales.data(), &packedW1);
    ut::buffer_error(packedW.mSPtr, packedW1.mSPtr, packedW1.mCSize);
    ut::buffer_error(packedW.template get<int8_t>(), packedW1.template get<int8_t>(), packedW1.template size<int8_t>());
    avector<float> dequant(n * k);
    kernel.unpackTransposeWeight(n, k, &packedW1, dequant.data(), k);
    ut::buffer_error(dequanRef.data(), dequant.data(), dequant.size());
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_TransposeBlockQuantize_F4 sUT_TransposeBlockQuantize_F4;
#endif

class UT_BlockQuantize_INT4 {
 public:
  UT_BlockQuantize_INT4() {
    UT_START();
    CheckISA(AVX2);
    ut_2<jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32>(4096, 4096, 128);
    CheckISA(AVX512F);
    ut_512vnni<jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32>(4096, 4096, 128);
  }
  template <template <class _T, JBLAS_ISA> class Wei>
  void ut_2(int n, int k, int blocksize, bool asym = false) {
    printf("Test Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.005f, 0.01f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    avector<float> dequant(quanW.size());
    avector<float> reduce(scales.size(), 0.f);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
        if (!asym) {
          dequant[i * n + j] = quanW.data()[i * n + j] * scales[i / blocksize * n + j];
        } else {
          dequant[i * n + j] =
              float(quanW.data()[i * n + j] - zero_points[i / blocksize * n + j]) * scales[i / blocksize * n + j];
        }
        reduce[i / blocksize * n + j] += dequant[i * n + j];
      }
    }

    auto constexpr RuntimeISA = JblasAVX2;
    using PrologueB = Wei<jblas::gemm::GemmCore_Row_NN_2x48_AVX2, JblasAVX2>;
    using PrologueB512 = Wei<jblas::gemm::GemmCore_Row_NN_2x48_AVX2, JblasAVX512F>;
    PrologueB kernel;
    PrologueB512 kernel512;
    utils::aligned_vector<int8_t> retW(n * k);
    auto packedW = kernel.createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(packedW.mSize);
    packedW.assign(buffer.data());
    kernel.packWeight(n, k, dequant.data(), ldb, &packedW);
    avector<float> unpackf32(dequant.size());
    avector<float> unpack512f32(dequant.size());
    kernel.unpackWeight(n, k, &packedW, unpackf32.data(), n);
    kernel512.unpackWeight(n, k, &packedW, unpack512f32.data(), n);
    ut::buffer_error(unpackf32.data(), unpack512f32.data(), unpackf32.size(), 0.01f);
  }
  template <template <class _T, JBLAS_ISA> class Wei>
  void ut_512vnni(int n, int k, int blocksize, bool asym = false) {
    printf("Test Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.005f, 0.01f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    avector<float> dequant(quanW.size());
    avector<float> reduce(scales.size(), 0.f);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        // quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0; //anyway there will be a float-rounding error about
        // 1 LSB.
        if (!asym) {
          dequant[i * n + j] = quanW.data()[i * n + j] * scales[i / blocksize * n + j];
        } else {
          dequant[i * n + j] =
              float(quanW.data()[i * n + j] - zero_points[i / blocksize * n + j]) * scales[i / blocksize * n + j];
        }
        reduce[i / blocksize * n + j] += dequant[i * n + j];
      }
    }

    auto constexpr RuntimeISA = JblasAVX512_VNNI;
    using PrologueB = Wei<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, RuntimeISA>;

    PrologueB kernel;
    utils::aligned_vector<int8_t> retW(n * k);
    auto packedW = kernel.createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(packedW.mSize);
    packedW.assign(buffer.data());
    kernel.packWeight(n, k, dequant.data(), ldb, &packedW);
    avector<float> unpackf32(dequant.size());
    kernel.unpackWeight(n, k, &packedW, unpackf32.data(), n);
    int lsb = 16;
    float err_thres = lsb * 0.01f;  // lsb*max_scale
    ut::buffer_error(dequant.data(), unpackf32.data(), dequant.size(), err_thres);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_BlockQuantize_INT4 sUT_BlockQuantize_INT4;
#endif

class UT_SerDes_INT4 {
 public:
  UT_SerDes_INT4() {
    UT_START();
    CheckISA(AVX512F);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32>(4096, 4096, 32);
    ut_c<jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32>(4096, 4096, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightS4FullRangeScaleFp32>(4096, 4096, 32);
    ut_c<jblas::prologue::weight_comp::gemm_kblcok::WeightS4FullRangeScaleFp32>(4096, 4096, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightS4FullRangeScaleFp32>(4096, 4096, 32, true);
    ut_c<jblas::prologue::weight_comp::gemm_kblcok::WeightS4FullRangeScaleFp32>(4096, 4096, 32, true);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32>(4096, 4096, 32, true);
    ut_c<jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32>(4096, 4096, 32, true);
  }

  template <template <class _T, JBLAS_ISA> class Wei>
  void ut(int n, int k, int blocksize, bool asym = false) {
    printf("Test Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.005f, 0.01f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
      }
    }

    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMLaunch_F32_AVX512F =
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                 prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AlphaBetaProcessFp32>;
    using PrologueB = typename GEMMLaunch_F32_AVX512F::PrologueB;
    using GEMMKernel =
        wrapper::gemm_pack_weight::GemmInterfacePackWeight<GEMMLaunch_F32_AVX512F, utils::parallel::Parallel2DGemm>;
    GEMMKernel kernel;
    utils::aligned_vector<int8_t> retW(n * k);
    auto packedW = kernel.getWeightPtr()->createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(packedW.mSize);
    packedW.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), asym ? zero_points.data() : nullptr,
                                       &packedW);

    packedW.serialize(buffer.data());
    auto tarptr =
        new jblas::prologue::weight_comp::gemm_kblcok::StorageWeightS4ScaleFp32(jblas::gemm::GemmCoreType::Undef);
    tarptr->deserialize(buffer.data());
    auto refptr = (prologue::weight_comp::gemm_kblcok::StorageWeightS4ScaleFp32*)&packedW;
    ut::buffer_error((int8_t*)refptr->get<int8_t>(), (int8_t*)tarptr->get<int8_t>(), tarptr->size<int8_t>());
    ut::buffer_error(refptr->mSPtr, tarptr->mSPtr, tarptr->mCSize);
    prologue::weight_comp::gemm_kblcok::StorageWeightF4ScaleFp32 test(jblas::gemm::GemmCoreType::Undef);
    test.deserialize(buffer.data());
    ut::buffer_error((int8_t*)refptr->get<int8_t>(), (int8_t*)test.get<int8_t>(), tarptr->size<int8_t>(), (int8_t)1);
    delete tarptr;
  }

  template <template <class _T, JBLAS_ISA> class Wei>
  void ut_c(int n, int k, int blocksize, bool asym = false) {
    printf("Test C type Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.01f, 0.09f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
      }
    }

    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMLaunch_F32_AVX512F =
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                 prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AlphaBetaProcessFp32>;
    using PrologueB = typename GEMMLaunch_F32_AVX512F::PrologueB;
    using GEMMKernel =
        wrapper::gemm_pack_weight::GemmInterfacePackWeight<GEMMLaunch_F32_AVX512F, utils::parallel::Parallel2DGemm>;
    GEMMKernel kernel;
    utils::aligned_vector<int8_t> retW(n * k);
    utils::aligned_vector<float> retScales(kblk_num * n);
    auto packedW = kernel.getWeightPtr()->createStorage(n, k, blocksize, asym);
    utils::avector<int8_t> buffer(packedW.mSize);
    packedW.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), asym ? zero_points.data() : nullptr,
                                       &packedW);
    utils::aligned_vector<int8_t> tbuf(packedW.mSize);
    packedW.serialize(tbuf.data());
    auto refptr = (prologue::weight_comp::gemm_kblcok::StorageWeightS4ScaleFp32*)&packedW;
    auto dptr = prologue::weight_comp::gemm_kblcok::PackedWeightParser::deserialBuffer(tbuf.data());
    auto tarptr = reinterpret_cast<prologue::weight_comp::gemm_kblcok::StorageWeightS4ScaleFp32*>(dptr);
    ut::buffer_error((int8_t*)refptr->get<int8_t>(), (int8_t*)tarptr->get<int8_t>(), tarptr->size<int8_t>());
    ut::buffer_error(refptr->mSPtr, tarptr->mSPtr, tarptr->mCSize);
    delete tarptr;
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_SerDes_INT4 sUT_SerDes_INT4;
#endif

class UT_Fp32_NN_INT4_BLOCK {
 public:
  UT_Fp32_NN_INT4_BLOCK() {
    UT_START();
    using namespace jblas::prologue::weight_comp::gemm_kblcok;
    using sAVX512F = jblas::gemm::GemmCore_Row_NN_8x48_AVX512F;
    using sAVX2 = jblas::gemm::GemmCore_Row_NN_2x48_AVX2;

    CheckISA(AVX2);
    // keep this for local dump file test
    // ut_file<WeightS4ClipScaleFp32, sAVX2>(7, 4096, 4096, 128);
    ut_transpose<WeightS4FullRangeScaleFp32, sAVX2>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut<WeightS4FullRangeScaleFp32, sAVX2>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 1024);
    ut_transpose<WeightS4ClipScaleFp32, sAVX2>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut<WeightS4ClipScaleFp32, sAVX2>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 1024);

    CheckISA(AVX512F);
    ut_transpose<WeightS4FullRangeScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut_transpose<WeightS4FullRangeScaleFp32, sAVX512F>(2, 32000, 4096, 4096, 32000, 32000, 0, 1.f, 1.f, 32);
    ut_transpose<WeightS4ClipScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut_transpose<WeightS4ClipScaleFp32, sAVX512F>(2, 32000, 4096, 4096, 32000, 32000, 0, 1.f, 1.f, 32);
    ut<WeightS4FullRangeScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, -1);
    ut<WeightS4FullRangeScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 128);
    ut<WeightS4FullRangeScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 1024);
    ut<WeightS4FullRangeScaleFp32, sAVX512F>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 1024);
    ut<WeightS4ClipScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, -1);
    ut<WeightS4ClipScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 128);
    ut<WeightS4ClipScaleFp32, sAVX512F>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 1024);
    ut<WeightS4ClipScaleFp32, sAVX512F>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 1024);
    ut_bf16<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut_bf16<WeightS4FullRangeScaleFp32>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 788);
    ut_bf16<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut_bf16<WeightS4ClipScaleFp32>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 788);
  }

  template <template <class _T, JBLAS_ISA> class Wei, class GemmCore_T>
  void ut_file(int m, int n, int k, int blocksize) {
    printf("Test Case %s: %d %d %d-%d\n", __FUNCTION__, m, n, k, blocksize);
    auto constexpr ISA = GemmCore_T::ISA;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<ISA, GemmCore_T, prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto weibuf = ut::readFile2Buffer<int8_t>("weight");
    // auto actbuf = ut::readFile2Buffer<float>("activation");
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    packedw.deserialize(weibuf.data());
    avector<float> kerC(m * n);
    avector<float> matA(m * k);
    ut::fill_buffer_randn(matA.data(), matA.size(), -0.1f, 0.1f);
    typename GEMMKernel::Arguments args{m, n, k, matA.data(), k, &packedw, kerC.data(), n, NULL};
    kernel.compute(args);
  }

  template <template <class _T, JBLAS_ISA> class Wei, class GemmCore_T>
  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);
    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);
    auto constexpr ISA = GemmCore_T::ISA;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<ISA, GemmCore_T, prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AlphaBetaProcessFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, _data.matB.data(), ldb, &packedw);
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, _data.matB.data(), ldb);
    _data.calc_ref(alpha, beta);
    typename GEMMKernel::Arguments args{
        m, n, k, _data.matA.data(), lda, &packedw, _data.matC.data(), _data.matD.data(), ldc, ldd, alpha, beta, NULL};

    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }

  template <template <class _T, JBLAS_ISA> class Wei, class GemmCore_T>
  void ut_transpose(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);

    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);

    utils::aligned_vector<float> BT(n * k);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        BT[i + j * k] = _data.matB[i * ldb + j];
      }
    }
    auto constexpr ISA = GemmCore_T::ISA;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<ISA, GemmCore_T, prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AlphaBetaProcessFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packTransposeWeight(n, k, BT.data(), k, &packedw);
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, _data.matB.data(), n);
    _data.calc_ref(alpha, beta);
    typename GEMMKernel::Arguments args{
        m, n, k, _data.matA.data(), lda, &packedw, _data.matC.data(), _data.matD.data(), ldc, ldd, alpha, beta, NULL};
    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }

  template <template <class _T, JBLAS_ISA> class Wei>
  void ut_bf16(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);

    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);

    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                 prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AlphaBetaProcessFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, _data.matB.data(), ldb, &packedw);
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, _data.matB.data(), n);
    _data.calc_ref(alpha, beta);
    typename GEMMKernel::Arguments args{
        m, n, k, _data.matA.data(), lda, &packedw, _data.matC.data(), _data.matD.data(), ldc, ldd, alpha, beta, NULL};
    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_Fp32_NN_INT4_BLOCK sUT_Fp32_NN_INT4_BLOCK;
#endif

class UT_AVX512F_NN_F4_BLOCK {
 public:
  UT_AVX512F_NN_F4_BLOCK() {
    UT_START();
    CheckISA(AVX512F);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(2, 4096, 4096, 4096, 4096, 4096, 0,
                                                                                   1.f, 1.f, 128);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f,
                                                                           128);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(2, 4096, 4096, 4096, 4096, 4096, 0,
                                                                                    1.f, 1.f, 128);
    ut_bf16<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(2, 4096, 4096, 4096, 4096, 4096,
                                                                                       0, 1.f, 1.f, 32);
    ut_bf16<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f,
                                                                                1.f, 32);
    ut_bf16<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(2, 4096, 4096, 4096, 4096,
                                                                                         4096, 0, 1.f, 1.f, 32);
  }
  template <template <class _GemmCore_T, JBLAS_ISA ISA_T> class Wei, JBLAS_F4_TYPE F4_T>
  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);

    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.f, 1.f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(0, 16);

    // scaling each channel's B value,make channel scale is unique
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        _data.matB[j * ldb + i] =
            jblas::kernel::ref::f4_dequantize<F4_T>(quanW.data()[j * ldb + i], scales[j / blocksize * n + i]);
      }
    }
    _data.calc_ref(alpha, beta);

    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                 prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AlphaBetaProcessFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), &packedw);
    typename GEMMKernel::Arguments args{
        m, n, k, _data.matA.data(), lda, &packedw, _data.matC.data(), _data.matD.data(), ldc, ldd, alpha, beta, NULL};

    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }
  template <template <class _GemmCore_T, JBLAS_ISA ISA_T> class Wei, JBLAS_F4_TYPE F4_T>
  void ut_bf16(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);

    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, ldd);
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.f, 1.f);
    utils::aligned_vector<utils::bf16> scalesbf16(kblk_num * n, utils::bf16{0});
    for (size_t i = 0; i < scales.size(); i++) {
      scalesbf16[i] = utils::cast<float, utils::bf16>(scales[i]);
    }
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(0, 16);

    for (size_t i = 0; i < scales.size(); i++) {
      scales[i] = utils::cast<utils::bf16, float>(scalesbf16[i]);
    }

    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        _data.matB[j * ldb + i] =
            jblas::kernel::ref::f4_dequantize<F4_T>(quanW.data()[j * ldb + i], scales[j / blocksize * n + i]);
      }
    }
    _data.calc_ref(alpha, beta);

    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMLaunch_F32_AVX512F =
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F,
                                                                 prologue::gemm::ActivationBase, Wei,
                                                                 jblas::epilogue::gemm::AlphaBetaProcessFp32>;
    using PrologueB = typename GEMMLaunch_F32_AVX512F::PrologueB;
    using GEMMKernel =
        wrapper::gemm_pack_weight::GemmInterfacePackWeight<GEMMLaunch_F32_AVX512F, utils::parallel::Parallel2DGemm>;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), &packedw);
    typename GEMMKernel::Arguments args{
        m, n, k, _data.matA.data(), lda, &packedw, _data.matC.data(), _data.matD.data(), ldc, ldd, alpha, beta, NULL};
    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX512F_NN_F4_BLOCK sUT_AVX512F_NN_F4_BLOCK;
#endif

class UT_AMXBF16_NN_INT4_BLOCK {
 public:
  UT_AMXBF16_NN_INT4_BLOCK() {
    UT_START();
    using namespace wrapper::gemm_default::weight_comp::amx_bf16;
    reorder_ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(64, 64, 32);
    reorder_ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(64, 32, 32);
    reorder_ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(64, 32, 32, true);
    reorder_ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(128, 32, 32);
    reorder_ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(128, 64, 32);
    reorder_ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(128, 128, 32);
    reorder_ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(64, 64, 32);
    reorder_ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(64, 32, 32);
    reorder_ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(128, 32, 32);
    reorder_ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(128, 64, 32);
    reorder_ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(128, 128, 32);
    reorder_ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(128, 128, 32);
    reorder_ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(128, 128, 32, true);
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(2, 64, 64, 64, 64, 64, 0, 1.f, 1.f, 32, true);
    ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32, true);
    ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(255, 1023, 33, 33, 1023, 1023, 0, 1.f, 1.f, 64);
    ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(255, 1023, 33, 33, 1023, 1023, 0, 1.f, 1.f, 64, true);
    ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 128);
    ut<GemmKernelS4ClipFp32KBlock, S4_CLIP>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 256);
    ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 128);
    ut<GemmKernelS4FullRangeFp32KBlock, S4_FULLRANGE>(2, 4096, 3136, 3136, 4096, 4096, 0, 1.f, 1.f, 32);
  }

  template <typename GEMMKernel, JBLAS_SIGN_INT_TYPE S4_T>
  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize,
          bool asym = false) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f %s\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha,
           beta, asym ? "asym" : "sym");

    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)-5, (int8_t)5);
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
    avector<bf16> matB(k * ldb), matA(m * lda);
    avector<float> matAf32(m * lda), refC(m * ldc), matBfp32(k * n);
    ut::fill_buffer_randn(matA.data(), matA.size(), bf16(0.0f), bf16(1.f));
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        if (asym) {
          matBfp32[j * n + i] = (float(jblas::kernel::ref::get_s8<S4_T>(quanW.data()[j * ldb + i] >> 4)) -
                                 float(zero_points[j / blocksize * n + i])) *
                                scales[j / blocksize * n + i];
        } else {
          matBfp32[j * n + i] =
              float(jblas::kernel::ref::get_s8<S4_T>(quanW.data()[j * ldb + i] >> 4)) * scales[j / blocksize * n + i];
        }
        matB[j * ldb + i] = bf16(matBfp32[j * n + i]);
      }
    }
    for (size_t i = 0; i < matA.size(); i++) {
      matAf32[i] = float(matA[i]);
    }
    ut::gemmref_bf16bf16fp32(m, n, k, matA.data(), matB.data(), refC.data(), lda, ldb, ldc);
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize, asym);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), asym ? zero_points.data() : nullptr,
                                       &packedw);
    avector<float> retC(m * ldc);
    typename GEMMKernel::Arguments args{m, n, k, matAf32.data(), lda, &packedw, retC.data(), ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(refC.data(), retC.data(), retC.size(), 0.003f);
  }

  template <typename GEMMKernel, JBLAS_SIGN_INT_TYPE S4_T>
  void reorder_ut(int n, int k, int blocksize, bool asym = false) {
    printf("Test Case %s: %d %d %d %s\n", __FUNCTION__, n, k, blocksize, asym ? "asym" : "sym");

    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);
    utils::aligned_vector<int8_t> zero_points(kblk_num * n);
    ut::fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(0), (int8_t)(0));
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(-127, 127);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        if (i % blocksize == 0) {
          quanW.data()[i * n + j] = 127;
        }
        if (i % blocksize == 1) {
          quanW.data()[i * n + j] = -128;
        }
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xf0;
      }
    }
    aligned_vector<utils::bf16> B(k * n);
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        float qw = float(jblas::kernel::ref::get_s8<S4_T>(quanW.data()[j * n + i] >> 4));
        if (asym) {
          qw = qw - float(zero_points[j / blocksize * n + i]);
        }
        B[j * n + i] = utils::bf16(qw * scales[j / blocksize * n + i]);
      }
    }

    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto wptr = kernel.getWeightPtr();
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize, asym);
    utils::avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), n, scales.data(), asym ? zero_points.data() : nullptr,
                                       &packedw);
    int kpad = padto(k, GEMMKernel::GemmCore::KTILE);
    int npad = padto(n, GEMMKernel::GemmCore::NTILE);

    using GEMMKernel_D = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN;
    GEMMKernel_D kernel_d;
    auto rawpack = kernel_d.getWeightPtr()->createStorage(n, k);
    utils::avector<int8_t> rawbuffer(rawpack.mSize);
    rawpack.assign(rawbuffer.data());
    kernel_d.getWeightPtr()->packWeight(n, k, {B.data(), n, &rawpack});
    aligned_vector<utils::bf16> deqB(kpad * npad);
    auto Bptr = deqB.data();
    int dstep = 0;
    wptr->getWeight(&Bptr, &dstep, kpad, npad, 0, 0, {&packedw});
    ut::buffer_error(rawpack.template get<utils::bf16>(), Bptr, rawpack.template size<utils::bf16>(),
                     utils::bf16(0.005f));
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AMXBF16_NN_INT4_BLOCK sUT_AMXBF16_NN_INT4_BLOCK;
#endif

class UT_AMXBF16_NN_F4_BLOCK {
 public:
  UT_AMXBF16_NN_F4_BLOCK() {
    UT_START();
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(64, 32, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(64, 64, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(128, 32, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(128, 64, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(128, 128, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(64, 32, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(64, 64, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(128, 32, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(128, 64, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(128, 128, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(64, 32, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(64, 64, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(128, 32, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(128, 64, 32);
    reorder_ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(128, 128, 32);
    CheckISA(AMX_BF16);
    request_perm_xtile_data();
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(2, 4096, 4096, 4096, 4096, 4096, 0,
                                                                                  1.f, 0.f, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(2, 4096, 4096, 4096, 4096, 4096, 0,
                                                                                    1.f, 0.f, 32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f,
                                                                           32);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(2, 4096, 4096, 4096, 4096, 4096, 0,
                                                                                  1.f, 0.f, 128);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4BnbScaleFp32, FP4_BNB>(2, 4096, 3128, 3128, 4096, 4096, 0,
                                                                                  1.f, 0.f, 256);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightFp4E2M1ScaleFp32, FP4_E2M1>(2, 4096, 3128, 3128, 4096, 4096, 0,
                                                                                    1.f, 0.f, 256);
    ut<jblas::prologue::weight_comp::gemm_kblcok::WeightNf4ScaleFp32, NF4>(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 0.f,
                                                                           256);
  }
  template <template <class _GemmCore_T, JBLAS_ISA ISA_T> class Wei, JBLAS_F4_TYPE F4_T>
  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);
    if (alpha != 1.f && beta != 0.f) {
      printf("No alpha beta support\n");
      return;
    }
    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.f, 1.f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(0, 16);
    avector<bf16> matA(m * lda), matB(k * ldb);
    avector<float> matAf32(m * lda), refC(m * ldc), retC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), bf16(-0.5f), bf16(0.5f));
    for (size_t i = 0; i < matA.size(); i++) {
      matAf32[i] = float(matA[i]);
    }

    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        matB[j * ldb + i] =
            bf16(kernel::ref::f4_dequantize<F4_T>(quanW.data()[j * ldb + i], scales[j / blocksize * n + i]));
      }
    }
    ut::gemmref_bf16bf16fp32(m, n, k, matA.data(), matB.data(), refC.data(), lda, ldb, ldc);
    auto constexpr RuntimeISA = JblasAMX_BF16;
    using GEMMKernel = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
            jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
            Wei,
            jblas::epilogue::gemm::AccumulatorWriteBackFp32>,  // output fp32->fp32
        jblas::wrapper::gemm_default::DefaultParallel>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), &packedw);
    typename GEMMKernel::Arguments args{m, n, k, matAf32.data(), lda, &packedw, retC.data(), ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(refC.data(), retC.data(), retC.size(), 0.001f);
  }
  template <template <class _GemmCore_T, JBLAS_ISA ISA_T> class Wei, JBLAS_F4_TYPE F4_T>
  void reorder_ut(int n, int k, int blocksize) {
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, n, k, blocksize);

    int kblk_num = utils::updiv(k, blocksize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.f, 1.f);
    ut::UT_vector_s8 quanW;
    quanW.resize(k * n);
    quanW.fill_rand(0, 16);

    aligned_vector<utils::bf16> B(k * n);
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        B[j * n + i] =
            utils::bf16(kernel::ref::f4_dequantize<F4_T>(quanW.data()[j * n + i], scales[j / blocksize * n + i]));
      }
    }

    auto constexpr RuntimeISA = JblasAMX_BF16;
    using GEMMKernel = jblas::wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16,
            jblas::prologue::gemm::ActivationConverterFp32,  // activation fp32->bf16
            Wei,
            jblas::epilogue::gemm::AccumulatorWriteBackFp32>,  // output fp32->fp32
        jblas::wrapper::gemm_default::DefaultParallel>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto wptr = kernel.getWeightPtr();
    auto packedw = wptr->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    wptr->packQWeight(n, k, quanW.data(), n, scales.data(), &packedw);
    using GEMMKernel_D = jblas::wrapper::gemm_default::amx_bf16::GemmKernelPackedWeightNN;
    GEMMKernel_D kernel_d;
    auto rawpackw = kernel_d.getWeightPtr()->createStorage(n, k);
    avector<int8_t> buffer1(rawpackw.mSize);
    rawpackw.assign(buffer1.data());
    kernel_d.getWeightPtr()->packWeight(n, k, {B.data(), n, &rawpackw});
    aligned_vector<utils::bf16> deqB(B.size());
    auto Bptr = deqB.data();
    int dstep = 0;
    wptr->getWeight(&Bptr, &dstep, k, n, 0, 0, {&packedw});
    ut::buffer_error(rawpackw.get<utils::bf16>(), Bptr, rawpackw.size<utils::bf16>(), utils::bf16(0.00001f));
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AMXBF16_NN_F4_BLOCK sUT_AMXBF16_NN_F4_BLOCK;
#endif

class UT_AVX512F_NN_INT8_BLOCK {
 public:
  UT_AVX512F_NN_INT8_BLOCK() {
    UT_START();
    CheckISA(AVX512F);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut_transpose(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 32);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 128);
    ut(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 1.f, 1024);
    ut(2, 4096, 3128, 3128, 4096, 4096, 0, 1.f, 1.f, 1024);
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

    // scaling each channel's B value,make channel scale is unique
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        _data.matB[j * ldb + i] = float(quanW.data()[j * ldb + i]) * scales[j / blocksize * n + i];
      }
    }
    _data.calc_ref(alpha, beta);

    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, prologue::gemm::ActivationBase,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), nullptr, &packedw);
    GEMMKernel::Arguments args{
        m, n, k, _data.matA.data(), lda, &packedw, _data.matC.data(), _data.matD.data(), ldc, ldd, alpha, beta, NULL};

    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }

  void ut_transpose(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
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
        quanW.data()[i * n + j] = quanW.data()[i * n + j] & 0xff;
      }
    }

    // scaling each channel's B value,make channel scale is unique
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        _data.matB[j * ldb + i] = float(quanW.data()[j * ldb + i]) * scales[j / blocksize * n + i];
      }
    }
    utils::aligned_vector<float> BT(n * k);
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        BT[i + j * k] = _data.matB[i * ldb + j];
      }
    }

    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, prologue::gemm::ActivationBase,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32, jblas::epilogue::gemm::AlphaBetaProcessFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packTransposeWeight(n, k, BT.data(), k, &packedw);
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, _data.matB.data(), n);
    _data.calc_ref(alpha, beta);
    GEMMKernel::Arguments args{
        m, n, k, _data.matA.data(), lda, &packedw, _data.matC.data(), _data.matD.data(), ldc, ldd, alpha, beta, NULL};

    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX512F_NN_INT8_BLOCK sUT_AVX512F_NN_INT8_BLOCK;
#endif

class UT_AVX512_VNNI_NN_INT8_PerChannel {
 public:
  UT_AVX512_VNNI_NN_INT8_PerChannel() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut_dynamicA_zp(2, 4096, 4096);
    ut_dynamicA_zp(128, 4096, 4096);
    ut_dynamicA(2, 4096, 3128);
    ut_dynamicA(2, 4096, 4096);
  }

  void ut_dynamicA(int m, int n, int k) {
    float alpha = 1.f, beta = 0.f;
    int ldb = n;
    int lda = k;
    int ldc = n;
    int ldd = 0;
    printf("Test Case %s: %d %d %d-%d %d %d %f %f\n", __FUNCTION__, m, n, k, lda, ldc, ldd, alpha, beta);
    avector<uint8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    for (size_t i = 0; i < m; i++) {
      matA[i * lda + 1] = 255;  // make sure kernel has the same quantization result
      matA[i * lda + 2] = 0;    // make sure kernel has the same quantization result
    }
    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    utils::aligned_vector<float> scales(n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_u8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
      }
    }
    avector<float> deqA(m * lda);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        deqA[i * lda + j] = matA[i * lda + j] * AScale[i];
      }
    }
    auto constexpr RuntimeISA = JblasAVX512_VNNI;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, prologue::gemm::ActivationFp32AsymU8Quantize,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN,
            jblas::epilogue::gemm::DequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto quan = kernel.getActivationPtr()->createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,          n,   k,          deqA.data(), lda,           &quan, &packedw,
                               ret.data(), ldc, quan.mSPtr, quan.mCStep, packedw.mSPtr, NULL};
    kernel.compute<true, false>(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.001f);
  }

  void ut_dynamicA_zp(int m, int n, int k) {
    int ldb = n;
    int lda = k;
    int ldc = n;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, m, n, k);
    avector<uint8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    avector<uint8_t> zpA(m);
    ut::fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(100), uint8_t(150));
    for (size_t i = 0; i < m; i++) {
      matA[i * lda + 1] = 255;  // make sure kernel has the same quantization result
      matA[i * lda + 2] = 0;    // make sure kernel has the same quantization result
    }
    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    utils::aligned_vector<float> scales(n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_u8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> Breduce(n, 0.f), deqB(n * k);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        deqB[i * n + j] = matB[i * n + j] * scales[j];
        Breduce[j] += deqB[i * n + j];
      }
    }

    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
        deqC[i * ldc + j] -= float(zpA[i]) * AScale[i] * Breduce[j];
      }
    }
    avector<float> deqA(m * lda);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        deqA[i * lda + j] = (matA[i * lda + j] - zpA[i]) * AScale[i];
      }
    }
    auto constexpr RuntimeISA = JblasAVX512_VNNI;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, prologue::gemm::ActivationFp32AsymU8Quantize,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN,
            jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto quan = kernel.getActivationPtr()->createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,          n,    k,           deqA.data(), lda,           &quan,      &packedw,
                               ret.data(), ldc,  quan.mCStep, quan.mSPtr,  packedw.mSPtr, quan.mZPtr, packedw.mRPtr,
                               NULL,       NULL, 1,           NULL};
    kernel.compute<true, false>(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX512_VNNI_NN_INT8_PerChannel sUT_AVX512_VNNI_NN_INT8_PerChannel;
#endif

class UT_AMX_INT8_NN_INT8_PerChannel {
 public:
  UT_AMX_INT8_NN_INT8_PerChannel() {
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    ut_dynamicA(2, 4096, 3136);  // TODO(Yi): handel case of 3128
    ut_dynamicA(2, 4096, 4096);
    ut(2, 4096, 4096, -1);
    ut(2, 4096, 3136, -1);
  }

  void ut_dynamicA(int m, int n, int k) {
    float alpha = 1.f, beta = 0.f;
    int ldb = n;
    int lda = k;
    int ldc = n;
    int ldd = 0;
    printf("Test Case %s: %d %d %d-%d %d %d %f %f\n", __FUNCTION__, m, n, k, lda, ldc, ldd, alpha, beta);
    avector<int8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), int8_t(-127), int8_t(127));
    for (size_t i = 0; i < m; i++) {
      matA[i * lda + 1] = 127;  // make sure kernel has the same quantization result
    }
    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    utils::aligned_vector<float> scales(n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.003f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_s8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
      }
    }
    avector<float> deqA(m * lda);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        deqA[i * lda + j] = matA[i * lda + j] * AScale[i];
      }
    }
    auto constexpr RuntimeISA = JblasAMX_INT8;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8, prologue::gemm::ActivationFp32SymS8Quantize,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32PerChannelN,
            jblas::epilogue::gemm::DequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto quan = kernel.getActivationPtr()->createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,          n,   k,          deqA.data(), lda,           &quan, &packedw,
                               ret.data(), ldc, quan.mSPtr, quan.mCStep, packedw.mSPtr, NULL};
    kernel.compute<true, false>(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.003f);
  }

  void ut(int m, int n, int k, int blocksize) {
    float alpha = 1.f, beta = 0.f;
    int ldb = n;
    int lda = k;
    int ldc = n;
    int ldd = 0;
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);
    avector<uint8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));

    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    for (size_t i = 0; i < n; i++) {
      matB[i + 1 * n] = 127;  // make sure kernel has the same quantization result
    }
    int bsize = blocksize <= 0 ? k : blocksize;
    int kblk_num = utils::updiv(k, bsize);
    utils::aligned_vector<float> scales(kblk_num * n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_u8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
      }
    }

    auto constexpr RuntimeISA = JblasAMX_INT8;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_U8S8, prologue::gemm::ActivationBase,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS8ScaleFp32, jblas::epilogue::gemm::DequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,   n, k, matA.data(), lda, &packedw, ret.data(), ldc, AScale.data(), 1, packedw.mSPtr,
                               NULL};
    kernel.compute(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AMX_INT8_NN_INT8_PerChannel sUT_AMX_INT8_NN_INT8_PerChannel;
#endif

class UT_AVX512F_NN_INT4_PerChannel {
 public:
  UT_AVX512F_NN_INT4_PerChannel() {
    UT_START();
    CheckISA(AVX512F);
    ut(2, 4096, 4096);
    ut(128, 4096, 4096);
  }

  void ut(int m, int n, int k) {
    int ldb = n;
    int lda = k;
    int ldc = n;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, m, n, k);
    avector<float> matA(m * lda);
    avector<float> matB(k * ldb);
    avector<float> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    ut::fill_buffer_randn(matB.data(), matB.size(), -0.5f, 0.5f);
    avector<float> deqB(k * ldb);
    auto constexpr RuntimeISA = JblasAVX512F;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfacePackWeight<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512F, prologue::gemm::ActivationBase,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
            jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packWeight(n, k, matB.data(), ldb, &packedw);
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, deqB.data(), ldb);
    ut::gemmref_fp32fp32fp32(m, n, k, matA.data(), deqB.data(), matC.data(), lda, ldb, ldc);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{
        m, n, k, matA.data(), lda, &packedw, ret.data(), ldc,
    };
    kernel.compute(args);
    ut::buffer_error(matC.data(), ret.data(), ret.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX512F_NN_INT4_PerChannel sUT_AVX512F_NN_INT4_PerChannel;
#endif

class UT_AVX512_VNNI_NN_INT4_PerChannel {
 public:
  UT_AVX512_VNNI_NN_INT4_PerChannel() {
    UT_START();
    CheckISA(AVX512_VNNI);
    ut_dynamicA_zp(2, 4096, 4096);
    ut_dynamicA_zp(128, 4096, 4096);
  }

  void ut_dynamicA_zp(int m, int n, int k) {
    int ldb = n;
    int lda = k;
    int ldc = n;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, m, n, k);
    avector<uint8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    avector<uint8_t> zpA(m);
    ut::fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(100), uint8_t(150));
    for (size_t i = 0; i < m; i++) {
      matA[i * lda + 1] = 255;  // make sure kernel has the same quantization result
      matA[i * lda + 2] = 0;    // make sure kernel has the same quantization result
    }
    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matB[i * n + j] &= 0xf0;
      }
    }
    utils::aligned_vector<float> scales(n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_u8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> Breduce(n, 0.f), deqB(n * k);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        deqB[i * n + j] = matB[i * n + j] * scales[j];
        Breduce[j] += deqB[i * n + j];
      }
    }

    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
        deqC[i * ldc + j] -= float(zpA[i]) * AScale[i] * Breduce[j];
      }
    }
    avector<float> deqA(m * lda);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        deqA[i * lda + j] = (matA[i * lda + j] - zpA[i]) * AScale[i];
      }
    }
    auto constexpr RuntimeISA = JblasAVX512_VNNI;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI, prologue::gemm::ActivationFp32AsymU8Quantize,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
            jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto quan = kernel.getActivationPtr()->createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,          n,    k,           deqA.data(), lda,           &quan,      &packedw,
                               ret.data(), ldc,  quan.mCStep, quan.mSPtr,  packedw.mSPtr, quan.mZPtr, packedw.mRPtr,
                               NULL,       NULL, 1,           NULL};
    kernel.compute<true, false>(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX512_VNNI_NN_INT4_PerChannel sUT_AVX512_VNNI_NN_INT4_PerChannel;
#endif

class UT_AVX_VNNI_NN_INT4_PerChannel {
 public:
  UT_AVX_VNNI_NN_INT4_PerChannel() {
    UT_START();
    CheckISA(AVX_VNNI);
    ut_dynamicA_zp(2, 4096, 4096);
    ut_dynamicA_zp(128, 4096, 4096);
  }

  void ut_dynamicA_zp(int m, int n, int k) {
    int ldb = n;
    int lda = k;
    int ldc = n;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, m, n, k);
    avector<uint8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    avector<uint8_t> zpA(m);
    ut::fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(100), uint8_t(150));
    for (size_t i = 0; i < m; i++) {
      matA[i * lda + 1] = 255;  // make sure kernel has the same quantization result
      matA[i * lda + 2] = 0;    // make sure kernel has the same quantization result
    }
    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matB[i * n + j] &= 0xf0;
      }
    }
    utils::aligned_vector<float> scales(n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_u8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> Breduce(n, 0.f), deqB(n * k);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        deqB[i * n + j] = matB[i * n + j] * scales[j];
        Breduce[j] += deqB[i * n + j];
      }
    }

    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
        deqC[i * ldc + j] -= float(zpA[i]) * AScale[i] * Breduce[j];
      }
    }
    avector<float> deqA(m * lda);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        deqA[i * lda + j] = (matA[i * lda + j] - zpA[i]) * AScale[i];
      }
    }
    auto constexpr RuntimeISA = JblasAVX_VNNI;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_2x48_AVX_VNNI, prologue::gemm::ActivationFp32AsymU8Quantize,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
            jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto quan = kernel.getActivationPtr()->createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,          n,    k,           deqA.data(), lda,           &quan,      &packedw,
                               ret.data(), ldc,  quan.mCStep, quan.mSPtr,  packedw.mSPtr, quan.mZPtr, packedw.mRPtr,
                               NULL,       NULL, 1,           NULL};
    kernel.compute<true, false>(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX_VNNI_NN_INT4_PerChannel sUT_AVX_VNNI_NN_INT4_PerChannel;
#endif

class UT_AMX_INT8_INT4_PerChannel {
 public:
  UT_AMX_INT8_INT4_PerChannel() {
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    ut_dynamicA_u8s8(2, 4096, 4096);
    ut_dynamicA_u8s8(128, 4096, 4096);
    ut_dynamicA_s8s8(2, 4096, 4096);
    ut_dynamicA_s8s8(128, 4096, 4096);
  }

  void ut_dynamicA_u8s8(int m, int n, int k) {
    int ldb = n;
    int lda = k;
    int ldc = n;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, m, n, k);
    avector<uint8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), uint8_t(0), uint8_t(255));
    avector<uint8_t> zpA(m);
    ut::fill_buffer_randn(zpA.data(), zpA.size(), uint8_t(100), uint8_t(150));
    for (size_t i = 0; i < m; i++) {
      matA[i * lda + 1] = 255;  // make sure kernel has the same quantization result
      matA[i * lda + 2] = 0;    // make sure kernel has the same quantization result
    }
    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matB[i * n + j] &= 0xf0;
      }
    }
    utils::aligned_vector<float> scales(n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_u8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> Breduce(n, 0.f), deqB(n * k);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        deqB[i * n + j] = matB[i * n + j] * scales[j];
        Breduce[j] += deqB[i * n + j];
      }
    }

    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
        deqC[i * ldc + j] -= float(zpA[i]) * AScale[i] * Breduce[j];
      }
    }
    avector<float> deqA(m * lda);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        deqA[i * lda + j] = (matA[i * lda + j] - zpA[i]) * AScale[i];
      }
    }
    auto constexpr RuntimeISA = JblasAMX_INT8;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_U8S8, prologue::gemm::ActivationFp32AsymU8Quantize,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
            jblas::epilogue::gemm::ZpDequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto quan = kernel.getActivationPtr()->createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,          n,    k,           deqA.data(), lda,           &quan,      &packedw,
                               ret.data(), ldc,  quan.mCStep, quan.mSPtr,  packedw.mSPtr, quan.mZPtr, packedw.mRPtr,
                               NULL,       NULL, 1,           NULL};
    kernel.compute<true, false>(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.001f);
  }

  void ut_dynamicA_s8s8(int m, int n, int k) {
    int ldb = n;
    int lda = k;
    int ldc = n;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, m, n, k);
    avector<int8_t> matA(m * lda);
    avector<int8_t> matB(k * ldb);
    avector<int32_t> matC(m * ldc);
    ut::fill_buffer_randn(matA.data(), matA.size(), int8_t(-127), int8_t(127));

    for (size_t i = 0; i < m; i++) {
      matA[i * lda + 1] = 127;  // make sure kernel has the same quantization result
    }
    ut::fill_buffer_randn(matB.data(), matB.size(), int8_t(-127), int8_t(127));
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        matB[i * n + j] &= 0xf0;
      }
    }
    utils::aligned_vector<float> scales(n);
    ut::fill_buffer_randn(scales.data(), scales.size(), 0.001f, 0.005f);

    avector<float> AScale(m);
    ut::fill_buffer_randn(AScale.data(), AScale.size(), 0.001f, 0.003f);
    ut::gemmref_s8s8s32(m, n, k, matA.data(), matB.data(), matC.data(), lda, ldb, ldc);
    avector<float> Breduce(n, 0.f), deqB(n * k);
    for (size_t i = 0; i < k; i++) {
      for (size_t j = 0; j < n; j++) {
        deqB[i * n + j] = matB[i * n + j] * scales[j];
        Breduce[j] += deqB[i * n + j];
      }
    }

    avector<float> deqC(m * ldc);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        deqC[i * ldc + j] = matC[i * ldc + j] * AScale[i] * scales[j];
      }
    }
    avector<float> deqA(m * lda);
    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        deqA[i * lda + j] = (matA[i * lda + j]) * AScale[i];
      }
    }
    auto constexpr RuntimeISA = JblasAMX_INT8;
    using GEMMKernel = wrapper::gemm_pack_weight::GemmInterfaceParallelAB<
        jblas::wrapper::gemm_pack_weight::GemmLauncherPackWeight<
            RuntimeISA, jblas::gemm::GemmCore_Row_NN_16x48_AMX_S8S8, prologue::gemm::ActivationFp32SymS8Quantize,
            jblas::prologue::weight_comp::gemm_kblcok::WeightS4ClipScaleFp32PerN,
            jblas::epilogue::gemm::DequantInt32ToFp32>,
        utils::parallel::Parallel2DGemm>;
    using PrologueB = GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto quan = kernel.getActivationPtr()->createStorage(m, k);
    avector<int8_t> bufA(quan.mSize);
    quan.assign(bufA.data());
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, false);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, matB.data(), ldb, scales.data(), nullptr, &packedw);
    avector<float> ret(m * ldc);
    GEMMKernel::Arguments args{m,          n,   k,          deqA.data(), lda,           &quan, &packedw,
                               ret.data(), ldc, quan.mSPtr, quan.mCStep, packedw.mSPtr, NULL};

    kernel.compute<true, false>(args);
    ut::buffer_error(deqC.data(), ret.data(), ret.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AMX_INT8_INT4_PerChannel sUT_AMX_INT8_INT4_PerChannel;
#endif

class UT_AVX512VNNI_NN_INT4_BLOCK {
 public:
  UT_AVX512VNNI_NN_INT4_BLOCK() {
    UT_START();
    CheckISA(AVX512_VNNI);
    using namespace jblas::prologue::weight_comp::gemm_kblcok;
    ut_new<WeightS4ClipScaleFp32>(2, 48, 256, 256, 48, 48, 0, 1.f, 0.f, 128);
    ut_new<WeightS4ClipScaleFp32>(255, 1023, 16, 16, 1023, 1023, 0, 1.f, 0.f, 128);
    ut_new<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 64);
    ut<WeightS4ClipScaleFp32>(255, 1023, 16, 16, 1023, 1023, 0, 1.f, 0.f, 128);
    ut_bf16<WeightS4ClipScaleFp32>(255, 1023, 16, 16, 1023, 1023, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 48, 8, 8, 48, 48, 0, 1.f, 0.f, 8);
    ut<WeightS4ClipScaleFp32>(2, 48, 128, 128, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 48, 256, 256, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 64);
    ut<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 32);
    ut_bf16<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut_bf16<WeightS4ClipScaleFp32>(2, 48, 128, 128, 48, 48, 0, 1.f, 0.f, 128);
    ut_bf16<WeightS4ClipScaleFp32>(2, 48, 8, 8, 48, 48, 0, 1.f, 0.f, 8);

    ut<WeightS4FullRangeScaleFp32>(2, 48, 128, 128, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4FullRangeScaleFp32>(2, 48, 256, 256, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 64);
    ut<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 32);
    ut_bf16<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut_bf16<WeightS4FullRangeScaleFp32>(2, 48, 128, 128, 48, 48, 0, 1.f, 0.f, 128);
    ut_bf16<WeightS4FullRangeScaleFp32>(2, 48, 8, 8, 48, 48, 0, 1.f, 0.f, 8);
  }

  template <template <class _T, JBLAS_ISA> class Wei>
  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);
    using GEMMKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
        jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
            JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
            jblas::prologue::gemm::ActivationF32U8KBlockQuantize, Wei, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
        jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matB(k * ldb), matA(m * lda);
    ut::fill_buffer_randn(matB.data(), matB.size(), -1.f, 1.f);
    ut::fill_buffer_randn(matA.data(), matA.size(), 0.f, 1.f);
    kernel.getWeightPtr()->packWeight(n, k, matB.data(), ldb, &packedw);
    avector<int8_t> unpacked(matB.size());
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, unpacked.data(), ldb);
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({matA.data(), lda, &quanA}, m, k);
    avector<float> reff32(m * ldc);
    ut::kblockgemmref_u8zp_s8_f32(m, n, k, blocksize, quanA.template get<uint8_t>(), quanA.mZPtr, quanA.mSPtr,
                                  unpacked.data(), packedw.mSPtr, reff32.data(), lda, updiv(k, blocksize), ldb,
                                  packedw.mNPad, ldc);
    avector<float> retf32(m * ldc);
    float elt_const_v[] = {0.f, 0.f, 0.f, 2.f, 3.f};
    typename GEMMKernel::Arguments args{m,   n,           k,   matA.data(), lda, &quanA, &packedw, retf32.data(),
                                        ldc, elt_const_v, NULL};
    kernel.compute(args, GELU,
                   LINEAR);  // eltop-chain fusion via dynamic-template-param demo.
    auto gelu = [&](float x) {
      return 0.5f * x * (1.f + tanhf(0.7978845834732056f * (x + 0.044714998453855515f * x * x * x)));
    };
    auto linear = [&](float alpha, float beta, float x) { return x * alpha + beta; };
    for (int i = 0; i < m * n; i++) reff32[i] = linear(2.f, 3.f, gelu(reff32[i]));
    ut::buffer_error(reff32.data(), retf32.data(), retf32.size(), 0.001f);
  }

  template <template <class _T, JBLAS_ISA> class Wei>
  void ut_new(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);
    using GEMMKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
        jblas::wrapper::gemm_kblock::GemmLauncherKBlock<
            JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK,
            jblas::prologue::gemm::ActivationF32U8KBlockQuantize, Wei, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
        jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matB(k * ldb), matA(m * lda);
    ut::fill_buffer_randn(matB.data(), matB.size(), -1.f, 1.f);
    ut::fill_buffer_randn(matA.data(), matA.size(), 0.f, 1.f);
    kernel.getWeightPtr()->packWeight(n, k, matB.data(), ldb, &packedw);
    avector<int8_t> unpacked(matB.size());
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, unpacked.data(), ldb);
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({matA.data(), lda, &quanA}, m, k);
    avector<float> reff32(m * ldc);
    ut::kblockgemmref_u8zp_s8_f32(m, n, k, blocksize, quanA.template get<uint8_t>(), quanA.mZPtr, quanA.mSPtr,
                                  unpacked.data(), packedw.mSPtr, reff32.data(), lda, updiv(k, blocksize), ldb,
                                  packedw.mNPad, ldc);
    avector<float> retf32(m * ldc);
    float elt_const_v[] = {0.f, 0.f, 0.f, 2.f, 3.f};
    typename GEMMKernel::Arguments args{m,   n,           k,   matA.data(), lda, &quanA, &packedw, retf32.data(),
                                        ldc, elt_const_v, NULL};
    kernel.compute(args, GELU,
                   LINEAR);  // eltop-chain fusion via dynamic-template-param demo.
    auto gelu = [&](float x) {
      return 0.5f * x * (1.f + tanhf(0.7978845834732056f * (x + 0.044714998453855515f * x * x * x)));
    };
    auto linear = [&](float alpha, float beta, float x) { return x * alpha + beta; };
    for (int i = 0; i < m * n; i++) reff32[i] = linear(2.f, 3.f, gelu(reff32[i]));
    ut::buffer_error(reff32.data(), retf32.data(), retf32.size(), 0.001f);
  }

  template <template <class _T, JBLAS_ISA> class Wei>
  void ut_bf16(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);
    using GEMMKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
        jblas::wrapper::gemm_kblock::GemmSLauncherKBlockPackWeight<
            JblasAVX512_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK,
            jblas::prologue::gemm::ActivationF32U8KBlockQuantize, Wei, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
        jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matB(k * ldb), matA(m * lda);
    ut::fill_buffer_randn(matB.data(), matB.size(), -1.f, 1.f);
    ut::fill_buffer_randn(matA.data(), matA.size(), 0.f, 1.f);
    kernel.getWeightPtr()->packWeight(n, k, matB.data(), ldb, &packedw);
    avector<int8_t> unpacked(matB.size());
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, unpacked.data(), ldb);
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({matA.data(), lda, &quanA}, m, k);
    avector<float> reff32(m * ldc);
    ut::kblockgemmref_u8zp_s8_f32(m, n, k, blocksize, quanA.template get<uint8_t>(), quanA.mZPtr, quanA.mSPtr,
                                  unpacked.data(), packedw.mSPtr, reff32.data(), lda, updiv(k, blocksize), ldb,
                                  packedw.mNPad, ldc);
    avector<float> retf32(m * ldc);
    typename GEMMKernel::Arguments args{m, n, k, matA.data(), lda, &quanA, &packedw, retf32.data(), ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(reff32.data(), retf32.data(), retf32.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX512VNNI_NN_INT4_BLOCK sUT_AVX512VNNI_NN_INT4_BLOCK;
#endif

class UT_AVX_VNNI_NN_INT4_BLOCK {
 public:
  UT_AVX_VNNI_NN_INT4_BLOCK() {
    UT_START();
    CheckISA(AVX_VNNI);
    using namespace jblas::prologue::weight_comp::gemm_kblcok;
    ut<WeightS4ClipScaleFp32>(255, 1023, 16, 16, 1023, 1023, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 48, 8, 8, 48, 48, 0, 1.f, 0.f, 8);
    ut<WeightS4ClipScaleFp32>(2, 48, 128, 128, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 48, 256, 256, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 64);
    ut<WeightS4ClipScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 32);

    ut<WeightS4FullRangeScaleFp32>(2, 48, 128, 128, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4FullRangeScaleFp32>(2, 48, 256, 256, 48, 48, 0, 1.f, 0.f, 128);
    ut<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 128);
    ut<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 64);
    ut<WeightS4FullRangeScaleFp32>(2, 4096, 4096, 4096, 4096, 4096, 0, 1.f, 0.f, 32);
  }

  template <template <class _T, JBLAS_ISA> class Wei>
  void ut(int m, int n, int k, int lda, int ldb, int ldc, int ldd, float alpha, float beta, int blocksize) {
    printf("Test Case %s: %d %d %d-%d %d %d %d %f %f\n", __FUNCTION__, m, n, k, blocksize, lda, ldc, ldd, alpha, beta);
    using GEMMKernel = jblas::wrapper::gemm_kblock::GemmInterfaceKBlockPackWeight<
        jblas::wrapper::gemm_kblock::GemmLauncherKBlock<
            JblasAVX_VNNI, jblas::gemm::kblock::GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK,
            jblas::prologue::gemm::ActivationF32U8KBlockQuantize, Wei, jblas::epilogue::gemm::AccumulatorWriteBackFp32>,
        jblas::utils::parallel::Parallel2DGemmKBlockFixed>;
    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    avector<float> matB(k * ldb), matA(m * lda);
    ut::fill_buffer_randn(matB.data(), matB.size(), -1.f, 1.f);
    ut::fill_buffer_randn(matA.data(), matA.size(), 0.f, 1.f);
    kernel.getWeightPtr()->packWeight(n, k, matB.data(), ldb, &packedw);
    avector<int8_t> unpacked(matB.size());
    kernel.getWeightPtr()->unpackWeight(n, k, &packedw, unpacked.data(), ldb);
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({matA.data(), lda, &quanA}, m, k);
    avector<float> reff32(m * ldc);
    ut::kblockgemmref_u8zp_s8_f32(m, n, k, blocksize, quanA.template get<uint8_t>(), quanA.mZPtr, quanA.mSPtr,
                                  unpacked.data(), packedw.mSPtr, reff32.data(), lda, updiv(k, blocksize), ldb,
                                  packedw.mNPad, ldc);
    avector<float> retf32(m * ldc);
    float elt_const_v[] = {0.f, 0.f, 0.f, 2.f, 3.f};
    typename GEMMKernel::Arguments args{m,   n,           k,   matA.data(), lda, &quanA, &packedw, retf32.data(),
                                        ldc, elt_const_v, NULL};
    kernel.compute(args, GELU);  // eltop-chain fusion via dynamic-template-param demo.
    auto gelu = [&](float x) {
      return 0.5f * x * (1.f + tanhf(0.7978845834732056f * (x + 0.044714998453855515f * x * x * x)));
    };
    for (int i = 0; i < m * n; i++) reff32[i] = gelu(reff32[i]);
    ut::buffer_error(reff32.data(), retf32.data(), retf32.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AVX_VNNI_NN_INT4_BLOCK sUT_AVX_VNNI_NN_INT4_BLOCK;
#endif

class UT_AMXINT8_NN_INT4_BLOCK {
 public:
  UT_AMXINT8_NN_INT4_BLOCK() {
    using namespace wrapper::gemm_default::weight_comp::amx_int8;
    UT_START();
    CheckISA(AMX_INT8);
    request_perm_xtile_data();
    jblas::utils::parallel::CpuDevice::getInstance()->setThreads(-1);
    ut_s_bf16<GemmSKernelDynamicS4ClipFp32KBlock, S4_CLIP>(2, 4096, 4096, 128);
    ut_s_bf16<GemmSKernelDynamicS4ClipFp32KBlock, S4_CLIP>(2, 4096, 4096, 256);
    ut_s_bf16<GemmSKernelDynamicS4FullRangeFp32KBlock, S4_FULLRANGE>(2, 4096, 4096, 128);
    ut_s_bf16<GemmSKernelDynamicS4FullRangeFp32KBlock, S4_FULLRANGE>(2, 4096, 4096, 256);
    ut<GemmSKernelDynamicS4ClipFp32KBlock, S4_CLIP>(2, 4096, 4096, 128);
    ut<GemmSKernelDynamicS4ClipFp32KBlock, S4_CLIP>(2, 4096, 4096, 256);
    ut<GemmSKernelDynamicS4FullRangeFp32KBlock, S4_FULLRANGE>(2, 4096, 4096, 128);
    ut<GemmSKernelDynamicS4FullRangeFp32KBlock, S4_FULLRANGE>(2, 4096, 4096, 256);
  }

  template <typename GEMMKernel, JBLAS_SIGN_INT_TYPE S4_T>
  void ut(int m, int n, int k, int blocksize) {
    int lda = k;
    int ldb = n, ldc = n;
    printf("Test Case %s: %d %d %d-%d %d %d\n", __FUNCTION__, m, n, k, blocksize, lda, ldc);

    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, 0);
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
        _data.matB[j * ldb + i] =
            float(jblas::kernel::ref::get_s8<S4_T>(quanW.data()[j * ldb + i] >> 4)) * scales[j / blocksize * n + i];
      }
    }
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        _data.matA[j * lda + i] = (float(matAs8.data()[j * lda + i])) * AScales[i / blocksize + j * kblk_num];
      }
    }
    _data.calc_ref(1.0f, 0.f);

    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), nullptr, &packedw);
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({_data.matA.data(), lda, &quanA}, m, k);

    typename GEMMKernel::Arguments args{m,   n,   k, _data.matA.data(), lda, &quanA, &packedw, _data.matC.data(),
                                        ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }

  template <typename GEMMKernel, JBLAS_SIGN_INT_TYPE S4_T>
  void ut_s_bf16(int m, int n, int k, int blocksize) {
    int lda = k;
    int ldb = n, ldc = n;
    printf("Test Case %s: %d %d %d-%d %d %d\n", __FUNCTION__, m, n, k, blocksize, lda, ldc);
    ut::UT_GEMMData_Row_f32 _data(m, n, k, lda, ldb, ldc, 0);
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
    for (int i = 0; i < scales.size(); i++) {
      auto valbf16 = utils::bf16();
      valbf16.fromfloat(scales[i]);
      scales[i] = valbf16.tofloat();
    }
    for (int j = 0; j < k; j++) {
      for (int i = 0; i < n; i++) {
        _data.matB[j * ldb + i] =
            float(jblas::kernel::ref::get_s8<S4_T>(quanW.data()[j * ldb + i] >> 4)) * scales[j / blocksize * n + i];
      }
    }
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < k; i++) {
        _data.matA[j * lda + i] = (float(matAs8.data()[j * lda + i])) * AScales[i / blocksize + j * kblk_num];
      }
    }
    _data.calc_ref(1.f, 0.f);

    using PrologueB = typename GEMMKernel::WeightType;
    GEMMKernel kernel;
    auto packedw = kernel.getWeightPtr()->createStorage(n, k, blocksize);
    avector<int8_t> buffer(packedw.mSize);
    packedw.assign(buffer.data());
    kernel.getWeightPtr()->packQWeight(n, k, quanW.data(), ldb, scales.data(), nullptr, &packedw);
    auto quanA = kernel.getActivationPtr()->createStorage(m, k, blocksize);
    avector<int8_t> bufA(quanA.mSize);
    quanA.assign(bufA.data());
    kernel.getActivationPtr()->quantize({_data.matA.data(), lda, &quanA}, m, k);

    typename GEMMKernel::Arguments args{m,   n,   k, _data.matA.data(), lda, &quanA, &packedw, _data.matC.data(),
                                        ldc, NULL};
    kernel.compute(args);
    ut::buffer_error(_data.matRef.data(), _data.matC.data(), _data.matC.size(), 0.001f);
  }
};
#ifdef JBLAS_UT_WEIGHT_COMPRESSION
static UT_AMXINT8_NN_INT4_BLOCK sUT_AMXINT8_NN_INT4_BLOCK;
#endif

}  // namespace gemm
}  // namespace wrapper
}  // namespace jblas
