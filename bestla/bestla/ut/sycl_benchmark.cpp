#include <stdio.h>
#include "bestla_wrapper.h"
#include "bestla_ut.h"
#include "sycl_ut.h"
#include "sycl/sycl_wrapper.h"

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
  using SGemmT = xve::DefaultSGemmCore;
  template <class GCT>
  using ProAT = sycl_prologue_a::ActivationBase<GCT, float>;
  template <class GCT>
  using ProBT = sycl_prologue_b::WeightBase<GCT, float>;
  template <class GCT>
  using EpiT = sycl_epilogue::OutputBase<GCT, float>;
  using KernelLauncher = sycl_wrapper::Launcher<ProAT, ProBT, EpiT, SGemmT>;

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
    sycl::range<2> group{SGemmT::WgM, SGemmT::WgN};
    sycl::range<2> problem{m / SGemmT::TileM, n / SGemmT::TileN};
    utils::GemmProblem gp(1, m, n, k);
    tm.start();
    while (tm.stop() < timems) {
      for (size_t i = 0; i < batch; i++) {
        log.start();
#if 0
        auto e_esimd = q->submit([&](sycl::handler& cgh) {
          sycl::local_accessor<float, 1> slm_b(sycl::range(SGemmT::SLM_B_Size), cgh);
          sycl::local_accessor<float, 1> slm_a(sycl::range(SGemmT::SLM_A_Size), cgh);
          cgh.parallel_for(
              sycl::nd_range<2>(problem, group),
              [=](sycl::nd_item<2> it) [[cl::reqd_work_group_size(
                  1, SGemmT::WgM,
                  SGemmT::WgN)]] [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(SGemmT::SgSize)]] {
                nd_item_helper<SGemmT> helper(it);
                float tmp[SGemmT::TileM * SGemmT::TileN];
                for (size_t im = 0; im < SGemmT::TileM; im++)
                  for (size_t in = 0; in < SGemmT::TileN; in++) tmp[im * SGemmT::TileN + in] = 0.f;

                for (int i = 0; i < k; i += SGemmT::TileK) {
                  sycl_prologue_b::WeightBase<SGemmT, float>::getWeight({B_d, n}, slm_b, i, helper);
                  it.barrier(sycl::access::fence_space::local_space);
                  SGemmT::compute(&A_d[helper.item_g_m() * k + i], k, slm_b, tmp, helper);
                  it.barrier(sycl::access::fence_space::local_space);
                }
                sycl_epilogue::OutputBase<SGemmT, float>::store({C_d, n}, tmp, helper);
              });
        });
#else
        //sycl::range<2> group{SGemmT::WgM, SGemmT::WgN};
        //using PrologueB = sycl_prologue_b::WeightBase<SGemmT, float>;
        //using Epilogue = sycl_epilogue::OutputBase<SGemmT, float>;
        //auto A = A_d;
        //auto B = B_d;
        //auto C = C_d;
        //int lda = k;
        //int ldb = n;
        //const int ldc = n;
        //sycl::range<2> problem{m / SGemmT::TileM, n / SGemmT::TileN};
        //auto e_esimd = q->submit([&](sycl::handler& cgh) {
        //  sycl::local_accessor<float, 1> slm_b(sycl::range(SGemmT::SLM_B_Size), cgh);
        //  sycl::local_accessor<float, 1> slm_a(sycl::range(SGemmT::SLM_A_Size), cgh);
        //  cgh.parallel_for(
        //      sycl::nd_range<2>(problem, group),
        //      [=](sycl::nd_item<2> it) [[cl::reqd_work_group_size(
        //          1, SGemmT::WgM,
        //          SGemmT::WgN)]] [[intel::kernel_args_restrict]] [[intel::reqd_sub_group_size(SGemmT::SgSize)]] {
        //        nd_item_helper<SGemmT> helper(it);
        //        float tmp[SGemmT::TileM * SGemmT::TileN];
        //        for (size_t im = 0; im < SGemmT::TileM; im++)
        //          for (size_t in = 0; in < SGemmT::TileN; in++) tmp[im * SGemmT::TileN + in] = 0.f;

        //        for (int i = 0; i < k; i += SGemmT::TileK) {
        //          PrologueB::getWeight({B_d, ldb}, slm_b, i, helper);
        //          sycl_prologue_b::WeightBase<SGemmT, float>::getWeight({B_d, ldc}, slm_b, i, helper);
        //          it.barrier(sycl::access::fence_space::local_space);
        //          SGemmT::compute(&A[helper.item_g_m() * lda + i], lda, slm_b, tmp, helper);
        //          it.barrier(sycl::access::fence_space::local_space);
        //        }
        //        //sycl_epilogue::OutputBase<SGemmT, float>::store({C_d, n}, tmp, helper);
        //        Epilogue::store({C, ldc}, tmp, helper);
        //      });
        //});
         auto e_esimd = KernelLauncher::compute({m, n, k, {A, k}, {B, n}, {C, n}}, q);
#endif
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
