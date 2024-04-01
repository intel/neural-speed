#include <stdio.h>
#include "bestla_wrapper.h"
#include "bestla_ut.h"
#include "sycl_ut.h"
#include "sycl/sycl_utils.h"
#include "sycl/sycl_gemm.h"

namespace bestla {
using namespace ut;
using namespace utils;
using namespace sycl_utils;
using namespace sycl_gemm;
namespace sycl_ut {
int constexpr TestMs = 1000;
class Benchmark_Fp32Fp32 {
 public:
  Benchmark_Fp32Fp32() {
    UT_START();
    benchmark_all(1024, 4096, 4096);
    benchmark_all(4096, 4096, 4096);
  }

  using AType = float;
  using BType = float;
  using CType = float;
  template <typename LOG_T>
  void benchmark(int m, int n, int k, int batch, AType* A, BType* B, CType* C, float timems) {
    LOG_T log;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    utils::timer<std::chrono::milliseconds> tm;
    auto A_d = A;
    auto B_d = B;
    auto C_d = C;
    auto psize = (size_t)m * n * k * 2;
    using SGemm_t = xve::DefaultSGemmCore;
    sycl::range<2> group{SGemm_t::WgM, SGemm_t::WgN};
    sycl::range<2> problem{m / SGemm_t::TileM, n / SGemm_t::TileN};
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
        auto e_esimd = q->submit([&](sycl::handler& cgh) {
          sycl::local_accessor<float, 1> slm_b(sycl::range(SGemm_t::SLM_B_Size), cgh);
          sycl::local_accessor<float, 1> slm_a(sycl::range(SGemm_t::SLM_A_Size), cgh);
          cgh.parallel_for(sycl::nd_range<2>(problem, group),
                           [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SGemm_t::SgSize)]] {
                             int g_idxm = it.get_group(0);
                             int g_idxn = it.get_group(1);
                             auto sg = it.get_sub_group();
                             int sgGroupId = sg.get_group_id()[0];
                             int sgId = sg.get_local_id()[0];
                             float tmp[SGemm_t::TileM * SGemm_t::TileN];
                             for (size_t im = 0; im < SGemm_t::TileM; im++)
                               for (size_t in = 0; in < SGemm_t::TileN; in++) tmp[im * SGemm_t::TileN + in] = 0.f;
                             int sg_idxn = sgGroupId % SGemm_t::SgNStride;
                             int sg_idxm = sgGroupId / SGemm_t::SgNStride;
                             int gm = g_idxm * SGemm_t::WgM;
                             int gn = g_idxn * SGemm_t::WgN;
                             int sgm = gm + sg_idxm;
                             int sgn = gn + sg_idxn * SGemm_t::SgSize;
                             int tm = sgm * SGemm_t::TileM;
                             int tn = (sgn + sgId) * SGemm_t::TileN;
                             for (int i = 0; i < k; i += SGemm_t::TileK) {
                               int constexpr Iter_PerWorker = (SGemm_t::TileK + SGemm_t::WgM - 1) / SGemm_t::WgM;
#pragma unroll
                               for (int icp = 0; icp < Iter_PerWorker; icp++) {
                                 {
                                   for (size_t in = 0; in < SGemm_t::TileN; in++) {
                                     slm_b[(sg_idxm + icp * SGemm_t::WgM) * SGemm_t::WgNEle +
                                           (sg_idxn * SGemm_t::SgSize + sgId) * SGemm_t::TileN + in] =
                                         B_d[tn + in + (i + sg_idxm + icp * SGemm_t::WgM) * n];
                                   }
                                 }
                               }

                               it.barrier(sycl::access::fence_space::local_space);
                               SGemm_t::compute(&A_d[tm * k + i], k, slm_b, sg_idxn * SGemm_t::SgNEle, tmp, sg, sgId);
                               // #pragma unroll(1)
                               //               for (int ik = 0; ik < SGemm_t::TileK; ik += SGemm_t::UnrollK) {
                               //                 float regA[SGemm_t::UnrollK];
                               //                 if constexpr (SGemm_t::UnrollK == 8) {
                               //                   *(sycl::vec<float, 4>*)regA = *(sycl::vec<float, 4>*)&A_d[(tm +
                               //                   sgId) * k + (i + ik)];
                               //                   *(sycl::vec<float, 4>*)&regA[4] = *(sycl::vec<float, 4>*)&A_d[(tm +
                               //                   sgId) * k + (i + ik + 4)];
                               //                 } else {
                               //                   *(sycl::vec<float, SGemm_t::UnrollK>*)regA =
                               //                       *(sycl::vec<float, SGemm_t::UnrollK>*)&A_d[(tm + sgId) * k + (i
                               //                       + ik)];
                               //                 }
                               // #pragma unroll
                               //                 for (int ikk = 0; ikk < SGemm_t::UnrollK; ikk++) {
                               //                   float tmpB[SGemm_t::TileN];
                               // #pragma unroll
                               //                   for (int in = 0; in < SGemm_t::TileN; in++) {
                               //                     tmpB[in] =
                               //                         slm_b[sg_idxn * SGemm_t::SgNEle + sgId * SGemm_t::TileN + in +
                               //                         (ik + ikk) * SGemm_t::WgNEle];
                               //                   }
                               //
                               // #pragma unroll
                               //                   for (int im = 0; im < SGemm_t::TileM; im++) {
                               //                     auto tmpA = sg.shuffle(regA[ikk], im);
                               // #pragma unroll
                               //                     for (int in = 0; in < SGemm_t::TileN; in++) {
                               //                       tmp[im * SGemm_t::TileN + in] += tmpA * tmpB[in];
                               //                     }
                               //                   }
                               //                 }
                               //               }
                               it.barrier(sycl::access::fence_space::local_space);
                             }
#pragma unroll
                             for (int im = 0; im < SGemm_t::TileM; im++) {
#pragma unroll
                               for (int in = 0; in < SGemm_t::TileN; in++) {
                                 C_d[(tm + im) * n + tn + in] = tmp[im * SGemm_t::TileN + in];
                               }
                             }
                           });
        });
        e_esimd.wait();
        log.stop();
        if (tm.stop() >= timems) {
          break;
        }
      }
    }
    log.record();
    double flops = double(psize) / log.min_val / 1e6;
    printf(" %s Flops:%.3f\n", log.get_log_str(), flops);
  }

  void benchmark_all(int m, int n, int k) {
    auto memsize = gemm_memsize(m, n, k, BTLA_DTYPE::F32, BTLA_DTYPE::F32, BTLA_DTYPE::F32);
    auto batch = auto_batch(memsize);
    printf("%d %d %d %d %s %s %s\n", m, n, k, batch, bestla_dtype_str(BTLA_DTYPE::F32),
           bestla_dtype_str(BTLA_DTYPE::F32), bestla_dtype_str(BTLA_DTYPE::F32));
    avector<AType> A(size_t(m) * k * batch);
    avector<BType> B(size_t(k) * n * batch);
    avector<CType> C(size_t(m) * n * batch, 0);
    fill_buffer_randn(A.data(), m * k, -0.5f, 0.5f);
    fill_buffer_randn(B.data(), n * k, -0.5f, 0.5f);
    for (size_t i = 0; i < batch - 1; i++) {
      memcpy(A.data() + i * m * k, A.data(), m * k * sizeof(AType));
      memcpy(B.data() + i * n * k, B.data(), n * k * sizeof(BType));
    }
    using LOG = timer_statistics_logger<TestMs * 2>;
    float testtime = float(TestMs);
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_vector<float> dA(A.size(), q), dB(B.size(), q), dC(C.size(), q);
    q->memcpy(dA.data(), A.data(), A.size() * 4).wait();
    q->memcpy(dB.data(), B.data(), B.size() * 4).wait();

    benchmark<LOG>(m, n, k, batch, dA.data(), dB.data(), dC.data(), testtime);
  }
};
static Benchmark_Fp32Fp32 sBenchmark_Fp32Fp32;
}  // namespace sycl_ut
}  // namespace bestla
