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
  using SGemmT = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using ProBTransT = sycl_prologue_b::WeightS4Trans<GCT, float>;
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
    auto e_esimd = KernelLauncher::compute({m, n, k, {A_d, k}, {B_d, n}, {C_d, n}}, q);
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
//static UT_SyclSGemm sUT_SyclSGemm;

class UT_SyclS4SGemm {
 public:
  UT_SyclS4SGemm() {
    UT_START();
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
        matB[i * k + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * B_scale[noffset];
        matB[i * k + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * B_scale[noffset];
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
    auto e_esimd = KernelTLauncher::compute({m, n, k, blocksize, {A_d, k}, {Bs8_d, B_scale_d, blks}, {C_d, n}}, q);
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
// static UT_SyclInt4Dequant sUT_SyclInt4Dequant;

class UT_SyclS4Gemv {
 public:
  UT_SyclS4Gemv() {
    UT_START();
    ut(1024, 1024, 32);
    ut_T(1024, 1024, 32);
    ut_T2(1024, 1024, 32);
    ut_T3(1024, 1024, 32);
    ut_T4(1024, 1024, 32);
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
    printf("Test Case %s: %d %d %d Device:%s\n", __FUNCTION__, n, k, blocksize, dev->getName().c_str());
    avector<uint8_t> rawB(k * n / 2);
    int blks = updiv(k, blocksize);
    avector<float> scale(blks * n), C(n), dqB(n * k), A(k), refC(n);
    fill_buffer_randn(scale.data(), scale.size(), 0.01f, 0.03f);
    fill_buffer_randn(A.data(), A.size(), -0.1f, 0.3f);
    fill_buffer_randn(rawB.data(), rawB.size(), uint8_t(0), uint8_t(255));
    auto srcptr = (utils::int4x2*)rawB.data();
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j += 2) {
        auto tmp = srcptr[i * n / 2 + j / 2];
        auto noffset = i / blocksize * n + j;
        dqB[i * n + j + 0] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * scale[noffset + 0];
        dqB[i * n + j + 1] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * scale[noffset + 1];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    int constexpr SgSize = 16;
    int constexpr KSlicing = 16;
    int constexpr TileN = 2;
    int constexpr GroupN = SgSize * TileN;
    sycl::range<1> group{KSlicing * SgSize};
    sycl::range<1> problem{KSlicing * n / TileN};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto n_blks = updiv(n / TileN, SgSize);
    int sg_ksize = k / KSlicing;
    auto e_esimd = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm(sycl::range(GroupN * (KSlicing - 1)), cgh);
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_group_id = sg.get_group_id()[0];
                         int sg_id = sg.get_local_id()[0];
                         int g_n = g_idx * GroupN;
                         int sg_k = sg_group_id * sg_ksize;
                         auto sptr = S_d + sg_k / blocksize * n + g_n;
                         auto bptr = B_d + (sg_k * n + g_n) / 2;
                         auto aptr = A_d + sg_k;
                         auto cptr = C_d + g_n;
                         float tmpAcc[TileN];
#pragma unroll
                         for (int i = 0; i < TileN; i++) {
                           tmpAcc[i] = 0.f;
                         }
                         for (int i = 0; i < sg_ksize; i += blocksize) {
                           float localAcc[TileN];
                           for (int i = 0; i < TileN; i++) {
                             localAcc[i] = 0.f;
                           }
#pragma unroll
                           for (int ik = 0; ik < blocksize; ik++) {
                             uint8_t tmp = *(bptr + ik * n / 2 + sg_id);
                             float tmpf[TileN];
                             tmpf[0] = static_cast<int8_t>((tmp & 0x0f) << 4);
                             tmpf[1] = static_cast<int8_t>((tmp & 0xf0));
                             auto tmpA = aptr[ik];
#pragma unroll
                             for (int in = 0; in < TileN; in++) {
                               localAcc[in] += tmpf[in] * tmpA;
                             }
                           }
                           for (int in = 0; in < TileN; in++) {
                             tmpAcc[in] += localAcc[in] * *(sptr + sg_id * TileN + in);
                           }
                           sptr += n;
                           aptr += blocksize;
                           bptr += blocksize * n / 2;
                         }
                         int slm_idx = sg_group_id - 1;
                         if (slm_idx >= 0) {
#pragma unroll
                           for (int in = 0; in < TileN; in++) {
                             slm[slm_idx * GroupN + sg_id * TileN + in] = tmpAcc[in];
                           }
                         }
                         it.barrier(sycl::access::fence_space::local_space);
                         if (sg_group_id == 0) {
#pragma unroll
                           for (int is = 0; is < KSlicing - 1; is++) {
#pragma unroll
                             for (int in = 0; in < TileN; in++) {
                               tmpAcc[in] += slm[is * GroupN + sg_id * TileN + in];
                             }
                           }
#pragma unroll
                           for (int in = 0; in < TileN; in++) {
                             cptr[sg_id * TileN + in] = tmpAcc[in];
                           }
                         }
                       });
    });
    e_esimd.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

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
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * scale[noffset];
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * scale[noffset];
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
    auto e_esimd = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::nd_range<1>(problem, group),
                       [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
                         int g_idx = it.get_group(0);
                         auto sg = it.get_sub_group();
                         int sg_id = sg.get_local_id()[0];
                         int g_n = g_idx;
                         auto sptr = S_d + g_n * blks;
                         auto bptr = B_d + g_n * k / 2;
                         auto aptr = A_d;
                         auto cptr = C_d + g_n;
                         float tmpAcc[TileK];
#pragma unroll
                         for (int i = 0; i < TileK; i++) {
                           tmpAcc[i] = 0.f;
                         }
                         for (int i = 0; i < k; i += blocksize) {
                           float localAcc[TileK];
                           for (int i = 0; i < TileK; i++) {
                             localAcc[i] = 0.f;
                           }
                           auto scale = *sptr;
                           for (int ik = 0; ik < blocksize; ik += GroupK) {
                             uint8_t tmp = *(bptr + ik / 2 + sg_id);
                             float tmpf[TileK];
                             tmpf[0] = static_cast<int8_t>((tmp & 0x0f) << 4);
                             tmpf[1] = static_cast<int8_t>((tmp & 0xf0));
#pragma unroll
                             for (int in = 0; in < TileK; in++) {
                               localAcc[in] += tmpf[in] * aptr[sg_id * TileK + in];
                             }
                           }
                           for (int in = 0; in < TileK; in++) {
                             tmpAcc[in] += localAcc[in] * scale;
                           }
                           sptr += 1;
                           aptr += blocksize;
                           bptr += blocksize / 2;
                         }
                         tmpAcc[0] += tmpAcc[1];
                         auto sum = 0.f;
                         for (int i = 0; i < SgSize; i++) {
                           sum += sg.shuffle(tmpAcc[0], i);
                         }
                         if (sg_id == 0) {
                           *cptr = sum;
                         }
                       });
    });
    e_esimd.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

  void ut_T2(int n, int k, int blocksize) {
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
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * scale[noffset];
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr GroupK = SgSize * TileK;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{n * SgSize};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = q->submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int g_n = g_idx;
            auto sptr = S_d + g_n * blks;
            auto bptr = B_d + g_n * k / 2;
            auto aptr = A_d;
            auto cptr = C_d + g_n;
            float tmpAcc[TileK];
#pragma unroll
            for (int i = 0; i < TileK; i++) {
              tmpAcc[i] = 0.f;
            }
            for (int i = 0; i < k; i += GroupK) {
              float tmpf[TileK];
              uint8_t tmps8[TileK / 2];
              *(sycl::vec<uint8_t, TileK / 2>*)tmps8 = *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
              auto scale = *(sptr + sg_id * TileK / blocksize);
#pragma unroll
              for (int ikk = 0; ikk < TileK; ikk += 2) {
                tmpf[ikk] = static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) << 4) * scale;
                tmpf[ikk + 1] = static_cast<int8_t>((tmps8[ikk / 2] & 0xf0)) * scale;
              }
              for (int ikk = 0; ikk < TileK; ikk += 1) {
                tmpAcc[ikk] += aptr[sg_id * TileK + ikk] * tmpf[ikk];
              }
              sptr += GroupK / blocksize;
              aptr += GroupK;
              bptr += GroupK / 2;
            }
            if constexpr (TileK >= 2) {
              for (int i = 0; i < TileK / 2; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 2];
              }
            }
            if constexpr (TileK >= 4) {
              for (int i = 0; i < TileK / 4; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 4];
              }
            }
            if constexpr (TileK >= 8) {
              for (int i = 0; i < TileK / 8; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 8];
              }
            }
            if constexpr (TileK >= 16) {
              for (int i = 0; i < TileK / 16; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 16];
              }
            }
            if constexpr (TileK >= 32) {
              for (int i = 0; i < TileK / 32; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 32];
              }
            }
            auto sum = 0.f;
            for (int i = 0; i < SgSize; i++) {
              sum += sg.shuffle(tmpAcc[0], i);
            }
            if (sg_id == 0) {
              *cptr = sum;
            }
          });
    });
    e_esimd.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

  void ut_T3(int n, int k, int blocksize) {
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
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * scale[noffset];
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr KSlicing = 8;
    int constexpr GroupK = SgSize * TileK;
    int sg_ksize = k / KSlicing;
    sycl::range<1> group{KSlicing * SgSize};
    sycl::range<1> problem{n * KSlicing * SgSize};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm(KSlicing - 1, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int sg_group_id = sg.get_group_id()[0];
            int g_n = g_idx;
            int g_k = sg_group_id * GroupK;
            auto sptr = S_d + g_n * blks + g_k / blocksize;
            auto bptr = B_d + g_n * k / 2 + g_k / 2;
            auto aptr = A_d + g_k;
            auto cptr = C_d + g_n;
            float tmpAcc[TileK];
#pragma unroll
            for (int i = 0; i < TileK; i++) {
              tmpAcc[i] = 0.f;
            }
            for (int i = 0; i < sg_ksize; i += GroupK) {
              float tmpf[TileK];
              uint8_t tmps8[TileK / 2];
              *(sycl::vec<uint8_t, TileK / 2>*)tmps8 = *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
              auto scale = *(sptr + sg_id * TileK / blocksize);
              for (int ikk = 0; ikk < TileK; ikk += 2) {
                tmpf[ikk] = static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) << 4) * scale;
                tmpf[ikk + 1] = static_cast<int8_t>((tmps8[ikk / 2] & 0xf0)) * scale;
              }
              for (int ikk = 0; ikk < TileK; ikk += 1) {
                tmpAcc[ikk] += aptr[sg_id * TileK + ikk] * tmpf[ikk];
              }
              sptr += KSlicing * GroupK / blocksize;
              aptr += KSlicing * GroupK;
              bptr += KSlicing * GroupK / 2;
            }
            if constexpr (TileK >= 2) {
              for (int i = 0; i < TileK / 2; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 2];
              }
            }
            if constexpr (TileK >= 4) {
              for (int i = 0; i < TileK / 4; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 4];
              }
            }
            if constexpr (TileK >= 8) {
              for (int i = 0; i < TileK / 8; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 8];
              }
            }
            if constexpr (TileK >= 16) {
              for (int i = 0; i < TileK / 16; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 16];
              }
            }
            if constexpr (TileK >= 32) {
              for (int i = 0; i < TileK / 32; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 32];
              }
            }
            auto sum = 0.f;
            for (int i = 0; i < SgSize; i++) {
              sum += sg.shuffle(tmpAcc[0], i);
            }
            if (sg_group_id != 0) {
              slm[sg_group_id - 1] = sum;
            }
            it.barrier(sycl::access::fence_space::local_space);
            if (sg_group_id == 0) {
              for (int i = 0; i < KSlicing - 1; i++) {
                sum += slm[i];
              }
              if (sg_id == 0) {
                *cptr = sum;
              }
            }
          });
    });
    e_esimd.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }

  void ut_T4(int n, int k, int blocksize) {
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
        dqB[i + (j + 0) * n] = static_cast<float>(static_cast<int8_t>(tmp.x) << 4) * scale[noffset];
        dqB[i + (j + 1) * n] = static_cast<float>(static_cast<int8_t>(tmp.y) << 4) * scale[noffset];
      }
    }
    gemmref_fp32fp32fp32(1, n, k, A.data(), dqB.data(), refC.data(), k, n, n);
    sycl_vector<float> dS(scale.size(), q), dC(n, q), dA(k, q);
    sycl_vector<uint8_t> dB(rawB.size(), q);
    q->memcpy(dB.data(), rawB.data(), rawB.size() * 1).wait();
    q->memcpy(dS.data(), scale.data(), scale.size() * 4).wait();
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr GroupK = SgSize * TileK;
    sycl::range<1> group{SgSize};
    sycl::range<1> problem{n * SgSize};
    auto S_d = dS.data();
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm(GroupK, cgh);
      cgh.parallel_for(
          sycl::nd_range<1>(problem, group), [=](sycl::nd_item<1> it) [[intel::reqd_sub_group_size(SgSize)]] {
            int g_idx = it.get_group(0);
            auto sg = it.get_sub_group();
            int sg_id = sg.get_local_id()[0];
            int g_n = g_idx;
            auto sptr = S_d + g_n * blks;
            auto bptr = B_d + g_n * k / 2;
            auto aptr = A_d;
            auto cptr = C_d + g_n;
            float tmpAcc[TileK];
#pragma unroll
            for (int i = 0; i < TileK; i++) {
              tmpAcc[i] = 0.f;
            }
            for (int i = 0; i < k; i += GroupK) {
              float tmpf[TileK];
              uint8_t tmps8[TileK / 2];
              *(sycl::vec<uint8_t, TileK / 2>*)tmps8 = *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
              auto scale = *(sptr + sg_id * TileK / blocksize);
              for (int ikk = 0; ikk < TileK; ikk += 2) {
                tmpf[ikk] = static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) << 4) * scale;
                tmpf[ikk + 1] = static_cast<int8_t>((tmps8[ikk / 2] & 0xf0)) * scale;
              }
              for (int ikk = 0; ikk < TileK; ikk += 4) {
                *(sycl::float4*)&slm[sg_id * TileK + ikk] = *(sycl::float4*)&tmpf[ikk];
              }
              float tmpA[4];
              float tmpB[4];
              for (int ikk = 0; ikk < TileK; ikk += 4) {
                *(sycl::float4*)&tmpB[0] = *(sycl::float4*)&slm[sg_id * 4 + ikk * SgSize];
                *(sycl::float4*)&tmpA[0] = *(sycl::float4*)&aptr[sg_id * 4 + ikk * SgSize];
                for (int ikkk = 0; ikkk < 4; ikkk++) {
                  tmpAcc[ikk] += tmpA[ikkk] * tmpB[ikkk];
                }
              }
              sptr += GroupK / blocksize;
              aptr += GroupK;
              bptr += GroupK / 2;
            }
            if constexpr (TileK >= 2) {
              for (int i = 0; i < TileK / 2; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 2];
              }
            }
            if constexpr (TileK >= 4) {
              for (int i = 0; i < TileK / 4; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 4];
              }
            }
            if constexpr (TileK >= 8) {
              for (int i = 0; i < TileK / 8; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 8];
              }
            }
            if constexpr (TileK >= 16) {
              for (int i = 0; i < TileK / 16; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 16];
              }
            }
            if constexpr (TileK >= 32) {
              for (int i = 0; i < TileK / 32; i++) {
                tmpAcc[i] += tmpAcc[i + TileK / 32];
              }
            }
            auto sum = 0.f;
            for (int i = 0; i < SgSize; i++) {
              sum += sg.shuffle(tmpAcc[0], i);
            }
            if (sg_id == 0) {
              *cptr = sum;
            }
          });
    });
    e_esimd.wait();
    q->memcpy(C.data(), C_d, C.size() * 4).wait();
    buffer_error(refC.data(), C.data(), C.size(), 0.001f);
  }
};
//static UT_SyclS4Gemv sUT_SyclS4Gemv;
}  // namespace sycl_ut
}  // namespace bestla
