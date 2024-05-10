#include "bestla_ut.h"
#include "sycl_ut.h"
#include "../sycl/sycl_wrapper.h"
#include "bestla_prologue_b.h"

namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
using namespace sycl_gemm;
namespace sycl_ut {
class UT_SyclSGemm {
 public:
  UT_SyclSGemm() {
    UT_START();
    ut(1, 1024, 1024);
    ut(300, 1024, 1024);
    ut(1024, 1024, 1024);
  }
  using SGemmT = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemmT>;

  void ut(int m, int n, int k) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    fill_buffer_randn(matB.data(), matB.size(), -0.5f, 0.5f);
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    sycl_vector<float> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 4).wait();
    q->memcpy(dB.data(), matB.data(), matB.size() * 4).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = KernelLauncher::compute(q, m, n, k, {{A_d, k}, {B_d, n}, {C_d, n}});
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclSGemm sUT_SyclSGemm;
#endif

class UT_SyclHGemm {
 public:
  UT_SyclHGemm() {
    UT_START();
    ut(1, 1024, 1024);
    ut(300, 1024, 1024);
    ut(1024, 1024, 1024);
    ut(1033, 1024, 1024);
  }
  using SGemmT = xve::DefaultHGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, sycl::half>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, sycl::half>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, sycl::half>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemmT>;

  void ut(int m, int n, int k) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    fill_buffer_randn(matB.data(), matB.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);

    sycl_vector<sycl::half> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dB.data(), matB.data(), matB.size() * 2).wait();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = KernelLauncher::compute(q, m, n, k, {{A_d, k}, {B_d, n}, {C_d, n}});
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclHGemm sUT_SyclHGemm;
#endif

class UT_SyclS4SGemm {
 public:
  UT_SyclS4SGemm() {
    UT_START();
    ut(300, 1024, 1024, 32);
    ut(1024, 1024, 1024, 32);
    utT(1024, 1024, 1024, 32);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightS4<GCT, float>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBT, EpiT, SGemm_t>;
  using KernelTLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBTransT, EpiT, SGemm_t>;

  void ut(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    int blks = k / blocksize;
    avector<float> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), 0.001f, 0.005f);
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j += 2) {
        auto tmp = srcptr[i * n / 2 + j / 2];
        auto noffset = i / blocksize * n + j;
        matB[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * B_scale[noffset + 0];
        matB[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * B_scale[noffset + 1];
      }
    }
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    sycl_vector<float> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 4).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 4).wait();
    sycl::range<2> group{SGemm_t::WgM, SGemm_t::WgN};
    sycl::range<2> problem{m / SGemm_t::TileM, n / SGemm_t::TileN};
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    utils::GemmProblem gp(1, m, n, k);
    auto e_esimd = KernelLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, n}, {C_d, n}});
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }

  void utT(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<float> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), -0.5f, 0.5f);
    int blks = k / blocksize;
    avector<float> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), 0.001f, 0.005f);
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        matB[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * B_scale[noffset];
        matB[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * B_scale[noffset];
      }
    }
    avector<float> matBNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(matB.data(), matBNT.data(), n, k, k, n);
    gemmref_fp32fp32fp32(m, n, k, matA.data(), matBNT.data(), ref.data(), k, n, n);
    sycl_vector<float> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 4).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 4).wait();
    sycl::range<2> group{SGemm_t::WgM, SGemm_t::WgN};
    sycl::range<2> problem{m / SGemm_t::TileM, n / SGemm_t::TileN};
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    utils::GemmProblem gp(1, m, n, k);
    auto e_esimd = KernelTLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, blks}, {C_d, n}});
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS4SGemm sUT_SyclS4SGemm;
#endif

class UT_SyclS4HGemm {
 public:
  UT_SyclS4HGemm() {
    UT_START();
    ut(300, 1024, 1024, 32);
    ut(1024, 1024, 1024, 32);
    utT(1024, 1024, 1024, 32);
  }
  using GemmT = xve::DefaultHGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, sycl::half>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightS4<GCT, sycl::half>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, sycl::half>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, sycl::half>;
  using KernelLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBT, EpiT, GemmT>;
  using KernelTLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBTransT, EpiT, GemmT>;

  void ut(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    int blks = k / blocksize;
    avector<utils::fp16> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), utils::fp16(0.001f), utils::fp16(0.005f));
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j += 2) {
        auto tmp = srcptr[i * n / 2 + j / 2];
        auto noffset = i / blocksize * n + j;
        matB[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * float(B_scale[noffset + 0]);
        matB[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * float(B_scale[noffset + 1]);
      }
    }
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matB.data(), ref.data(), k, n, n);
    sycl_vector<sycl::half> dA(matA.size(), q), dB(matB.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 2).wait();
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    auto e_esimd = KernelLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, n}, {C_d, n}});
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }

  void utT(int m, int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", m, n, k, dev->getName().c_str());
    avector<utils::fp16> matA(m * k), matB(k * n), matC(m * n), ref(m * n);
    fill_buffer_randn(matA.data(), matA.size(), utils::fp16(-0.5f), utils::fp16(0.5f));
    int blks = k / blocksize;
    avector<utils::fp16> B_scale(size_t(blks) * n);
    avector<uint8_t> B_s8(k * n / 2);
    fill_buffer_randn(B_s8.data(), B_s8.size(), uint8_t(0), uint8_t(255));
    fill_buffer_randn(B_scale.data(), B_scale.size(), utils::fp16(0.001f), utils::fp16(0.005f));
    auto srcptr = (utils::int4x2*)B_s8.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        matB[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * float(B_scale[noffset]);
        matB[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * float(B_scale[noffset]);
      }
    }
    avector<utils::fp16> matBNT(k * n);
    kernel::wrapper::Transpose2D<utils::fp16>::forward<BTLA_ISA::NoSIMD>(matB.data(), matBNT.data(), n, k, k, n);
    gemmref_fp16fp16fp16(m, n, k, matA.data(), matBNT.data(), ref.data(), k, n, n);
    sycl_vector<sycl::half> dA(matA.size(), q), dC(matC.size(), q), dB_scale(B_scale.size(), q);
    sycl_vector<uint8_t> dBs8(B_s8.size(), q);
    q->memcpy(dA.data(), matA.data(), matA.size() * 2).wait();
    q->memcpy(dBs8.data(), B_s8.data(), B_s8.size() * 1).wait();
    q->memcpy(dB_scale.data(), B_scale.data(), B_scale.size() * 2).wait();
    auto A_d = dA.data();
    auto Bs8_d = dBs8.data();
    auto B_scale_d = dB_scale.data();
    auto C_d = dC.data();
    auto e_esimd = KernelTLauncher::compute(q, m, n, k, blocksize, {{A_d, k}, {Bs8_d, B_scale_d, blks}, {C_d, n}});
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 2).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), utils::fp16(0.2f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS4HGemm sUT_SyclS4HGemm;
#endif

class UT_SyclInt4Dequant {
 public:
  UT_SyclInt4Dequant() {
    UT_START();
    ut_fp32(1024, 1024, 32);
    ut_fp32_T(1024, 1024, 32);
  }

  void ut_fp32(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j += 2) {
        auto tmp = srcptr[i * n / 2 + j / 2];
        auto noffset = i / blocksize * n + j;
        ref[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset + 0];
        ref[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset + 1];
      }
    }
    using ProB = sycl_prologue_b::WeightS4<sycl_gemm::xve::DefaultSGemmCore, float>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto e_esimd = ProB::dequant_s4<sycl_prologue_b::KernelConfigBase>(n, k, blocksize, {B_d, S_d, n}, DB_d, q);
    e_esimd.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);
  }

  void ut_fp32_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case: %d %d %d Device:%s\n", n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), dequant(n * k), ref(n * k);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        ref[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        ref[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    using ProB = sycl_prologue_b::WeightS4Trans<sycl_gemm::xve::DefaultSGemmCore, float>;
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto e_esimd = ProB::dequant_s4<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    e_esimd.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);

    avector<float> refNT(k * n);
    kernel::wrapper::Transpose2D<float>::forward<BTLA_ISA::NoSIMD>(ref.data(), refNT.data(), n, k, k, n);
    e_esimd = ProB::dequant_s4_trans<sycl_prologue_b::KernelConfigTrans>(n, k, blocksize, {B_d, S_d, blks}, DB_d, q);
    e_esimd.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(refNT.data(), dequant.data(), dequant.size(), 0.001f);
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclInt4Dequant sUT_SyclInt4Dequant;
#endif

class UT_SyclS4Gemv {
 public:
  UT_SyclS4Gemv() {
    UT_START();
    ut_T(1024, 1024, 32);
    ut_half(1024, 1024, 32);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemm_t>;

  void ut_T(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(A.data(), A.size(), -0.1f, 0.3f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * scale[noffset];
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 2;
    int constexpr GroupK = SgSize * TileK;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{n * SgSize};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = ProBTransT<SGemm_t>::gemv(A_d, {B_d, S_d, blks}, C_d, n, k, blocksize, q);
    e_esimd.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

  void ut_half(int n, int k, int blocksize) {
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<utils::fp16> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), utils::fp16(0.01f), utils::fp16(0.03f));
    fill_buffer_randn(A.data(), A.size(), utils::fp16(-0.1f), utils::fp16(0.3f));
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < k; j += 2) {
        auto tmp = srcptr[i * k / 2 + j / 2];
        auto noffset = i * blks + j / blocksize;
        float fscale = float(scale[noffset]);
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) - 8) * fscale;
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) - 8) * fscale;
      }
    }
    gemmref_fp16fp16fp16(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<sycl::half> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 2).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 2).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr GroupK = SgSize * TileK;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{n * SgSize};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = sycl_prologue_b::WeightS4Trans<xve::DefaultHGemmCore, sycl::half>::gemv(A_d, {B_d, S_d, blks}, C_d,
                                                                                           n, k, blocksize, q);
    e_esimd.wait();
    q->memcpy(C.data(), C_d, C.size() * 2).wait();
    buffer_error(refC.data(), C.data(), C.size(), utils::fp16(0.1f));
  }
};
#ifdef BTLA_UT_SYCL
static UT_SyclS4Gemv sUT_SyclS4Gemv;
#endif
}  // namespace sycl_ut
}  // namespace bestla
