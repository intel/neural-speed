#include "../jit_blas_utils.h"
#include "../jit_blas_gemm.h"
#include "jit_blas_ut.h"

namespace jblas {
namespace utils {

namespace parallel {
class UT_GemmParallel {
 public:
  UT_GemmParallel() {
    UT_START();
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4, 4096, 4096,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4096, 4096, 4096,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 4096, 4096,
                                                      40);  // vnni
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 16384, 4096,
                                                      40);  // vnni
    ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_U8S8>(4096, 16384, 4096,
                                                    56);  // amxint8
    ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16>(4096, 4096, 4096,
                                                    40);  // vnni
  }
  template <class GemmCore>
  void ut(int m, int n, int k, int threads) {
    printf("case %s (size %d %d %d) (nthd %d)\n", __FUNCTION__, m, n, k, threads);
    utils::parallel::Parallel2DGemm<GemmCore> paral;
    paral.update(m, n, k, threads);
    paral.print();
  }
};
#ifdef JBLAS_UT_UTILS
static UT_GemmParallel sUT_GemmParallel;
#endif

class UT_GemmParallelKBlock {
 public:
  UT_GemmParallelKBlock() {
    UT_START();
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4, 4096, 4096, 32,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4096, 4096, 4096, 32,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 4096, 4096, 32,
                                                      40);  // vnni
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 16384, 4096, 32,
                                                      40);  // vnni
    // ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_U8S8>(4096, 16384, 4096, 32,
    //                                                 56);  // amxint8
    ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16>(4096, 4096, 4096, 32,
                                                    40);  // vnni

    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4, 4096, 4096, 128,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4096, 4096, 4096, 128,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 4096, 4096, 128,
                                                      40);  // vnni
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 16384, 4096, 128,
                                                      40);  // vnni
    ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_U8S8>(4096, 16384, 4096, 128,
                                                    56);  // amxint8
    ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16>(4096, 4096, 4096, 128,
                                                    40);  // vnni

    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4, 4096, 4096, 1024,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512F>(4096, 4096, 4096, 1024,
                                                  20);  // avx512f
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 4096, 4096, 1024,
                                                      40);  // vnni
    ut<jblas::gemm::GemmCore_Row_NN_8x48_AVX512_VNNI>(4096, 16384, 4096, 1024,
                                                      40);  // vnni
    ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_U8S8>(4096, 16384, 4096, 1024,
                                                    56);  // amxint8
    ut<jblas::gemm::GemmCore_Row_NN_16x64_AMX_BF16>(4096, 4096, 4096, 1024,
                                                    40);  // vnni
  }

  template <class GemmCore>
  void ut(int m, int n, int k, int blocksize, int threads) {
    printf("case %s (size %d %d %d) (nthd %d)\n", __FUNCTION__, m, n, k, threads);
    utils::parallel::Parallel2DGemmKBlock<GemmCore> paral;
    paral.update(m, n, k, blocksize, threads);
    paral.print();
  }
};
#ifdef JBLAS_UT_UTILS
static UT_GemmParallelKBlock sUT_GemmParallelKBlock;
#endif
class UT_RowmajorParallelKBlock {
 public:
  UT_RowmajorParallelKBlock() {
    UT_START();
    ut(2, 4096, 1, 4, 32, 56);
    ut(4096, 4096, 1, 4, 32, 56);
    ut(2, 4096, 1, 4, 128, 56);
    ut(4096, 4096, 1, 4, 128, 56);
    ut(2, 4096, 1, 4, 1024, 56);
    ut(4096, 4096, 1, 4, 1024, 56);

    ut(2, 4096, 1, 4, 32, 48);
    ut(4096, 4096, 1, 4, 32, 48);
    ut(2, 4096, 1, 4, 128, 48);
    ut(4096, 4096, 1, 4, 128, 48);
    ut(2, 4096, 1, 4, 1024, 48);
    ut(4096, 4096, 1, 4, 1024, 48);
  }

  void ut(int row, int col, int minrow, int mincol, int blocksize, int threads) {
    printf("case %s (size %d %d) (padding %d %d) (block %d) (nthd %d)\n", __FUNCTION__, row, col, minrow, mincol,
           blocksize, threads);
    utils::parallel::Parallel2DRowMajorColBlock paral;
    paral.update(row, col, minrow, mincol, blocksize, threads);
    paral.print();
  }
};
#ifdef JBLAS_UT_UTILS
static UT_RowmajorParallelKBlock sUT_RowmajorParallelKBlock;
#endif

}  // namespace parallel
}  // namespace utils
}  // namespace jblas
