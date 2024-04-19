#include "kernel_ut.h"
#include "kernel_avx2.h"
#include "kernel_avx512f.h"
namespace bestla {
using namespace utils;
namespace ut {
#if CompileAVX512F()
class UT_Avx512f_decompress_kblock_s4_fp {
 public:
  UT_Avx512f_decompress_kblock_s4_fp() {
    UT_START();
    CheckISA(AVX512F);
    ut<BTLA_DTYPE::S4_CLIP, 2, float, utils::bf16>(32, 128, 128, 128, 0, 32, 128);
    ut<BTLA_DTYPE::S4_CLIP, 2, float, utils::bf16>(32, 96, 96, 96, 0, 32, 96);
    ut<BTLA_DTYPE::S4_CLIP, 1, float, float>(32, 48, 48, 128, 0, 32, 128);
    ut<BTLA_DTYPE::S4_CLIP, 1, float, utils::bf16>(32, 48, 48, 128, 0, 32, 128);
  }

  template <BTLA_DTYPE S4_T, int PACK_ROW, typename ST_T, typename DST_T>
  void ut(int row, int col, int ld_src, int ld_dst, int k_offset, int kblock, int NPad, bool asym = false) {
    printf("Test Case %s_%d_%d: %d %d %d %d %d %d %d %d\n", __FUNCTION__, int(S4_T), PACK_ROW, row, col, ld_src, ld_dst,
           k_offset, kblock, NPad, asym);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<DST_T> bf16_wei(ld_dst * row);
    std::vector<DST_T> ref_wei(ld_dst * row);
    std::vector<ST_T> scales(col);
    std::vector<int8_t> zero_points(col);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-128), int8_t(127));
    fill_buffer_randn(scales.data(), scales.size(), ST_T(0.01f), ST_T(0.02f));
    fill_buffer_randn(zero_points.data(), zero_points.size(), (int8_t)(-5), (int8_t)(5));

    for (int i = 0; i < col * row; i += 2) {
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]);
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]);
    }
    kernel::ref::decompress_kblock_s4_fp<S4_T, DST_T, PACK_ROW, ST_T>(
        s4_wei.data(), ref_wei.data(), row, col, ld_src, ld_dst, scales.data(), asym ? zero_points.data() : nullptr,
        k_offset, kblock, NPad, cache, CacheSize);
    kernel::avx512f::decompress_kblock_s4_fp<S4_T, DST_T, PACK_ROW, ST_T>(
        s4_wei.data(), bf16_wei.data(), row, col, ld_src, ld_dst, scales.data(), asym ? zero_points.data() : nullptr,
        k_offset, kblock, NPad, cache, CacheSize);
    ut::buffer_error(ref_wei.data(), bf16_wei.data(), bf16_wei.size(), DST_T(BF16_ERR));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_Avx512f_decompress_kblock_s4_fp sUT_Avx512f_decompress_kblock_s4_fp;
#endif
#endif
#if CompileAVX2()
class UT_avx2_decompress_s4_s8 {
 public:
  UT_avx2_decompress_s4_s8() {
    UT_START();
    CheckISA(AVX2);
    ut<BTLA_DTYPE::S4_CLIP>(32, 128);
    ut<BTLA_DTYPE::S4_CLIP>(32, 96);
    ut<BTLA_DTYPE::S4_CLIP>(32, 48);
  }

  template <BTLA_DTYPE S4_T>
  void ut(int row, int col) {
    printf("Test Case %s_%s: %d %d\n", __FUNCTION__, bestla_dtype_str(S4_T), row, col);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-128), int8_t(127));

    for (int i = 0; i < col * row; i += 2) {
      s8_wei[i] = s8_wei[i] & 0xf0;
      s8_wei[i + 1] = s8_wei[i + 1] & 0xf0;
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]);
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]);
    }
    kernel::avx2::decompress_s4_s8<S4_T>(s4_wei.data(), rev.data(), row, col, col, col);

    ut::buffer_error(s8_wei.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
static UT_avx2_decompress_s4_s8 sUT_avx2_decompress_s4_s8;
#endif
class UT_avx2_gemv {
 public:
  UT_avx2_gemv() {
    UT_START();
    CheckISA(AVX2);
    ut_2bit(24, 32, 32, true);
  }

  void ut_2bit(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, n, k, kblock);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(blks);
    avector<int8_t> bzp(n * blks);
    avector<uint8_t> A(k), azp(blks);
    avector<float> Af32(k), Bf32(n * k), Cf32(n), Cref(n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-50), int8_t(50));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < 4; j++) {
        Af32[i + j] = (int(A[i + j]) - azp[bid]) * scalea[bid];
      }
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 4];
        Bf32[(i + 0) * n + j] = (int(b24.a << 6) - bzp[bid * n + j]) * scaleb[bid * n + j];
        Bf32[(i + 1) * n + j] = (int(b24.b << 6) - bzp[bid * n + j]) * scaleb[bid * n + j];
        Bf32[(i + 2) * n + j] = (int(b24.c << 6) - bzp[bid * n + j]) * scaleb[bid * n + j];
        Bf32[(i + 3) * n + j] = (int(b24.d << 6) - bzp[bid * n + j]) * scaleb[bid * n + j];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx2::gemv_2bit_u8s8_fp32<float, 24>({A.data(), scalea.data(), azp.data()},
                                                {nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), bzp.data(), 2},
                                                Cf32.data(), k, n, kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }
};
UT_avx2_gemv sUT_avx2_gemv;

}  // namespace ut
}  // namespace bestla
