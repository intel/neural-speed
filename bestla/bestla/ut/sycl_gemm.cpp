#include "bestla_ut.h"
#include "sycl_ut.h"
#include "../sycl/sycl_wrapper.h"

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
    /*auto e_esimd = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm_b(sycl::range(SGemm_t::SLM_B_Size), cgh);
      sycl::local_accessor<float, 1> slm_a(sycl::range(SGemm_t::SLM_A_Size), cgh);
      cgh.parallel_for(sycl::nd_range<2>(problem, group),
                       [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SGemm_t::SgSize)]] {
                         nd_item_helper<SGemm_t> helper(it);
                         float tmp[SGemm_t::TileM * SGemm_t::TileN];
                         for (size_t im = 0; im < SGemm_t::TileM; im++)
                           for (size_t in = 0; in < SGemm_t::TileN; in++) tmp[im * SGemm_t::TileN + in] = 0.f;

                         for (int i = 0; i < k; i += SGemm_t::TileK) {
                           sycl_prologue_b::WeightBase<SGemm_t, float>::getWeight({B_d, n}, slm_b, i, helper);
                           it.barrier(sycl::access::fence_space::local_space);
                           SGemm_t::compute(&A_d[helper.item_g_m() * k + i], k, slm_b, tmp, helper);
                           it.barrier(sycl::access::fence_space::local_space);
                         }
                         sycl_epilogue::OutputBase<SGemm_t, float>::store({C_d, n}, tmp, helper);
                       });
    });*/
    utils::GemmProblem gp(1, m, n, k);
    auto e_esimd = KernelLauncher::compute({m, n, k, {A_d, k}, {B_d, n}, {C_d, n}}, q);

    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
static UT_SyclSGemm sUT_SyclSGemm;

}  // namespace sycl_ut
}  // namespace bestla
