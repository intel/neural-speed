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
    // kernel::ref::decompress_kblock_s4_fp<S4_T, DST_T, PACK_ROW, ST_T>(
    //     s4_wei.data(), ref_wei.data(), row, col, ld_src, ld_dst, scales.data(), asym ? zero_points.data() : nullptr,
    //     k_offset, kblock, NPad, cache, CacheSize);
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
    ut<1, 24>(32);
    ut<4, 24>(32);
    ut<1, 24>(32, true);
    ut<2, 24>(32, true);
    ut<4, 24>(32, true);
  }

  template <int PackRow, int NTILE>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, row, col, blocksize);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<int8_t> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = PackRow;
    std::vector<int8_t> zp(col * blks);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    std::vector<int8_t> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-8), int8_t(7));

    for (int i = 0; i < col * row; i += 2) {
      s8_ref[i] = s8_wei[i];
      s8_ref[i + 1] = s8_wei[i + 1];
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]) + 8;
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]) + 8;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] -= zp[i / blocksize * NTILE + j];
          }
        }
      }
    }

    kernel::avx2::decompress_kblock_s4_s8<PackRow, NTILE>(s4_wei.data(), isasym ? zp.data() : nullptr, rev.data(),
                                                          blocksize, NTILE, 0, 0, row_offset, NTILE, cache, CacheSize);
    kernel::avx2::decompress_kblock_s4_s8<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, isasym ? zp.data() : nullptr, rev.data() + row_offset * NTILE,
        blocksize, NTILE, 0, row_offset, row - row_offset, NTILE, cache, CacheSize);
    ut::buffer_error(s8_ref.data(), rev.data(), rev.size(), int8_t(0));
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
#endif
static UT_avx2_decompress_s4_s8 sUT_avx2_decompress_s4_s8;

class UT_avx2_decompress_s4_fp {
 public:
  UT_avx2_decompress_s4_fp() {
    UT_START();
    CheckISA(AVX2);
    ut<1, 24, float>(32);
    ut<2, 24, float>(32);
    ut<4, 24, float>(32);
    ut<1, 24, float>(32, true);
    ut<2, 24, float>(32, true);
    ut<4, 24, float>(32, true);
  }

  template <int PackRow, int NTILE, typename T>
  void ut(int blocksize, bool isasym = false) {
    int row = blocksize * 2;
    int constexpr col = NTILE;
    printf("Test Case %s: %d %d %d Asym:%d Pack:%d\n", __FUNCTION__, row, col, blocksize, isasym, PackRow);
    std::vector<utils::int4x2> s4_wei(row * col / 2);
    std::vector<int8_t> s8_wei(col * row);
    std::vector<T> s8_ref(col * row);
    int blks = row / blocksize;
    int row_offset = blocksize / 2;
    std::vector<int8_t> zp(col * blks);
    avector<float> scale(col * blks);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(zp.data(), zp.size(), int8_t(-8), int8_t(7));
    std::vector<T> rev(col * row);
    fill_buffer_randn(s8_wei.data(), s8_wei.size(), int8_t(-8), int8_t(7));

    for (int i = 0; i < col * row; i += 2) {
      s8_ref[i] = float(s8_wei[i]);
      s8_ref[i + 1] = float(s8_wei[i + 1]);
      s4_wei[i / 2].x = utils::int4x2::convert(s8_wei[i]) + 8;
      s4_wei[i / 2].y = utils::int4x2::convert(s8_wei[i + 1]) + 8;
    }
    if (isasym) {
      for (int i = 0; i < row; i += PackRow) {
        for (int j = 0; j < NTILE; j++) {
          int corr_offset = i / blocksize * NTILE + j;
          for (int ip = 0; ip < PackRow; ip++) {
            s8_ref[i * NTILE + j * PackRow + ip] = float(s8_ref[i * NTILE + j * PackRow + ip]) - float(zp[corr_offset]);
          }
        }
      }
    }

    for (int i = 0; i < row; i += PackRow) {
      for (int j = 0; j < NTILE; j++) {
        int corr_offset = i / blocksize * NTILE + j;
        for (int ip = 0; ip < PackRow; ip++) {
          s8_ref[i * NTILE + j * PackRow + ip] =
              float(s8_ref[i * NTILE + j * PackRow + ip]) * float(scale[corr_offset]);
        }
      }
    }

    kernel::avx2::decompress_kblock_s4_fp<PackRow, NTILE>(s4_wei.data(), rev.data(), row_offset, NTILE, scale.data(),
                                                          BTLA_DTYPE::F32, isasym ? zp.data() : nullptr, 0, 0,
                                                          blocksize, NTILE, cache, CacheSize);
    kernel::avx2::decompress_kblock_s4_fp<PackRow, NTILE>(
        s4_wei.data() + row_offset * NTILE / 2, rev.data() + row_offset * NTILE, row - row_offset, NTILE, scale.data(),
        BTLA_DTYPE::F32, isasym ? zp.data() : nullptr, row_offset, 0, blocksize, NTILE, cache, CacheSize);

    ut::buffer_error(s8_ref.data(), rev.data(), rev.size());
  }
};
#ifdef BTLA_UT_KERNEL_INTRIN
#endif
static UT_avx2_decompress_s4_fp sUT_avx2_decompress_s4_fp;

class UT_avx2_gemv {
 public:
  UT_avx2_gemv() {
    UT_START();
    CheckISA(AVX2);
    ut_4bit(24, 128, 32, true);
    ut_4bit(24, 128, 32, false);
    ut_4bit_s8s8(24, 128, 32, true);
    ut_4bit_s8s8(24, 128, 32, false);

    ut_2bit(24, 128, 32, true);
    ut_2bit(24, 128, 32, false);
    ut_2bit_s8s8(24, 128, 32, true);
    ut_2bit_s8s8(24, 128, 32, false);
  }

  void ut_4bit(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, n, k, kblock);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(blks);
    avector<int8_t> bzp(n * blks);
    avector<uint8_t> A(k), azp(blks);
    avector<float> Af32(k), Bf32(n * k), Cf32(n), Cref(n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-8), int8_t(7));
    fill_buffer_randn(azp.data(), azp.size(), uint8_t(100), uint8_t(150));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < 4; j++) {
        Af32[i + j] = (int(A[i + j]) - azp[bid]) * scalea[bid];
      }
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 2];
        auto b42 = b2[(i * n + j * 4 + 2) / 2];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(1, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx2::gemv_4bit_u8s8_fp32<float, 24>(
        {A.data(), scalea.data(), azp.data()},
        {(uint8_t*)b2.data(), nullptr, nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2}, Cf32.data(), k, n,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  void ut_4bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, n, k, kblock);
    int blks = k / kblock;
    avector<bit4x2> b2(n * k / 2);
    avector<float> scaleb(n * blks), scalea(blks);
    avector<int8_t> bzp(n * blks);
    avector<int8_t> A(k);
    avector<float> Af32(k), Bf32(n * k), Cf32(n), Cref(n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < 4; j++) {
        Af32[i + j] = int(A[i + j]) * scalea[bid];
      }
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 2];
        auto b42 = b2[(i * n + j * 4 + 2) / 2];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.y - 8)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b42.x - 8)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b42.y - 8)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(1, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx2::gemv_4bit_s8s8_fp32<float, 24>(
        {(uint8_t*)A.data(), scalea.data(), nullptr},
        {(uint8_t*)b2.data(), nullptr, nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2}, Cf32.data(), k, n,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
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
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-2), int8_t(1));
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
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(1, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx2::gemv_2bit_u8s8_fp32<float, 24>(
        {A.data(), scalea.data(), azp.data()},
        {nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2}, Cf32.data(), k, n,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }

  void ut_2bit_s8s8(int n, int k, int kblock, bool iasym) {
    printf("Test Case %s: %d %d %d\n", __FUNCTION__, n, k, kblock);
    int blks = k / kblock;
    avector<bit2x4> b2(n * k / 4);
    avector<float> scaleb(n * blks), scalea(blks);
    avector<int8_t> bzp(n * blks);
    avector<int8_t> A(k);
    avector<float> Af32(k), Bf32(n * k), Cf32(n), Cref(n);
    fill_buffer_randn((uint8_t*)b2.data(), b2.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(A.data(), A.size(), int8_t(-127), int8_t(127));
    fill_buffer_randn(bzp.data(), bzp.size(), int8_t(-2), int8_t(1));
    fill_buffer_randn(scaleb.data(), scaleb.size(), 0.01f, 0.02f);
    fill_buffer_randn(scalea.data(), scalea.size(), 0.01f, 0.02f);

    for (int i = 0; i < k; i += 4) {
      int bid = i / kblock;
      for (int j = 0; j < 4; j++) {
        Af32[i + j] = int(A[i + j]) * scalea[bid];
      }
      for (int j = 0; j < n; j += 1) {
        auto b24 = b2[(i * n + j * 4) / 4];
        if (iasym) {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2) - bzp[bid * n + j]) * scaleb[bid * n + j];
        } else {
          Bf32[(i + 0) * n + j] = (int(b24.a - 2)) * scaleb[bid * n + j];
          Bf32[(i + 1) * n + j] = (int(b24.b - 2)) * scaleb[bid * n + j];
          Bf32[(i + 2) * n + j] = (int(b24.c - 2)) * scaleb[bid * n + j];
          Bf32[(i + 3) * n + j] = (int(b24.d - 2)) * scaleb[bid * n + j];
        }
      }
    }
    gemmref_fp32fp32fp32(1, n, k, Af32.data(), Bf32.data(), Cref.data(), k, n, n);
    kernel::avx2::gemv_2bit_s8s8_fp32<float, 24>(
        {(uint8_t*)A.data(), scalea.data(), nullptr},
        {nullptr, (uint8_t*)b2.data(), nullptr, scaleb.data(), iasym ? bzp.data() : nullptr, 2}, Cf32.data(), k, n,
        kblock, cache, CacheSize);
    buffer_error(Cref.data(), Cf32.data(), Cref.size(), FP32_ERR);
  }
};
UT_avx2_gemv sUT_avx2_gemv;

}  // namespace ut
}  // namespace bestla
