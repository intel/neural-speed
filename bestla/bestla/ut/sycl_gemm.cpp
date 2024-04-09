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
    ut(1024, 1024, 1024);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemm_t>;

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
    sycl::range<2> group{SGemm_t::WgM, SGemm_t::WgN};
    sycl::range<2> problem{m / SGemm_t::TileM, n / SGemm_t::TileN};
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    utils::GemmProblem gp(1, m, n, k);
    auto e_esimd = KernelLauncher::compute({m, n, k, {A_d, k}, {B_d, n}, {C_d, n}}, q);
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
// static UT_SyclSGemm sUT_SyclSGemm;

class UT_SyclS4SGemm {
 public:
  UT_SyclS4SGemm() {
    UT_START();
    ut(1024, 1024, 1024, 32);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightS4<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::LauncherWOQ<ProAT, ProBT, EpiT, SGemm_t>;

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
        matB[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * B_scale[noffset + 0];
        matB[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * B_scale[noffset + 1];
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
    auto e_esimd = KernelLauncher::compute({m, n, k, blocksize, {A_d, k}, {Bs8_d, B_scale_d, n}, {C_d, n}}, q);
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
static UT_SyclS4SGemm sUT_SyclS4SGemm;

class UT_SyclInt4Dequant {
 public:
  UT_SyclInt4Dequant() {
    UT_START();
    ut(1024, 1024, 32);
  }
  using SGemm_t = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemm_t>;

  void ut(int n, int k, int blocksize) {
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
        ref[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * scale[noffset + 0];
        ref[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * scale[noffset + 1];
      }
    }
    sycl_vector<float> dS(scale.size(), q), dequantB(n * k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    int constexpr SgSize = 16;
    int constexpr TileN = 2;
    int constexpr GroupN = SgSize * TileN;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{n / TileN * blks};
    auto S_d = dS.data();
    auto B_d = dB.data();
    auto DB_d = dequantB.data();
    auto n_blks = updiv(n / TileN, SgSize);
    auto e_esimd = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_id = sg.get_local_id()[0];
                         int g_idx_n = g_idx % n_blks;
                         int g_idx_k = g_idx / n_blks;
                         int g_n = g_idx_n * GroupN;
                         int g_k = g_idx_k * blocksize;
                         auto sptr = S_d + g_idx_k * n + g_n;
                         auto bptr = B_d + (g_k * n + g_n) / 2;
                         auto dbptr = DB_d + g_k * n + g_n;
                         float scale[TileN];
                         for (int in = 0; in < TileN; in++) {
                           scale[in] = *(sptr + sg_id * TileN + in);
                         }
                         for (int ik = 0; ik < blocksize; ik++) {
                           uint8_t tmp = *(bptr + ik * n / 2 + sg_id);
                           float tmpf[TileN];
                           tmpf[0] = static_cast<int8_t>((tmp & 0x0f) << 4) * scale[0];
                           tmpf[1] = static_cast<int8_t>((tmp & 0xf0)) * scale[1];
                           for (int in = 0; in < TileN; in++) {
                             dbptr[in + sg_id * TileN + ik * n] = tmpf[in];
                           }
                         }
                       });
    });
    e_esimd.wait();
    q->memcpy(dequant.data(), DB_d, dequant.size() * 4).wait();
    buffer_error(ref.data(), dequant.data(), dequant.size(), 0.001f);
  }
};
static UT_SyclInt4Dequant sUT_SyclInt4Dequant;
}  // namespace sycl_ut
}  // namespace bestla
