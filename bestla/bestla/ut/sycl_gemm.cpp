#include "bestla_ut.h"
#include "sycl_ut.h"
#include "../sycl/sycl_utils.h"
#include "../sycl/sycl_device.h"
#include "../sycl/sycl_gemm.h"
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
    using SGemm_t = xve::DefaultSGemmCore;
    sycl::range<2> group{SGemm_t::WgM, SGemm_t::WgN};
    sycl::range<2> problem{m / SGemm_t::TileM, n / SGemm_t::TileN};
    auto A_d = dA.data();
    auto B_d = dB.data();
    auto C_d = dC.data();
    auto e_esimd = q->submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> slm_b(sycl::range(SGemm_t::SLM_B_Size), cgh);
      sycl::local_accessor<float, 1> slm_a(sycl::range(SGemm_t::SLM_A_Size), cgh);
      cgh.parallel_for(
          sycl::nd_range<2>(problem, group), [=](sycl::nd_item<2> it) [[intel::reqd_sub_group_size(SGemm_t::SgSize)]] {
            nd_item_helper<SGemm_t> helper(it);
            float tmp[SGemm_t::TileM * SGemm_t::TileN];
            for (size_t im = 0; im < SGemm_t::TileM; im++)
              for (size_t in = 0; in < SGemm_t::TileN; in++) tmp[im * SGemm_t::TileN + in] = 0.f;

            for (int i = 0; i < k; i += SGemm_t::TileK) {
              int constexpr Iter_PerWorker = (SGemm_t::TileK + SGemm_t::WgM - 1) / SGemm_t::WgM;
#pragma unroll
              for (int icp = 0; icp < Iter_PerWorker; icp++) {
                // if (sg_idxm + icp * GroupM < TileK)
                {
                  for (size_t in = 0; in < SGemm_t::TileN; in++) {
                    slm_b[helper.sg_idx_m() * helper.wg_size_n() + icp * SGemm_t::WgM * SGemm_t::WgNEle +
                          (helper.sg_idx_n() * SGemm_t::SgSize + helper.sg_id()) * SGemm_t::TileN + in] =
                        B_d[helper.item_g_n() + in + (i + helper.sg_idx_m() + icp * SGemm_t::WgM) * n];
                  }
                }
              }

              it.barrier(sycl::access::fence_space::local_space);
              SGemm_t::compute(&A_d[helper.item_g_m() * k + i], k, slm_b, helper.sg_idx_n() * SGemm_t::SgNEle, tmp,
                               helper.sg,
                               helper.sg_id());
              it.barrier(sycl::access::fence_space::local_space);
            }
#pragma unroll
            for (int im = 0; im < SGemm_t::TileM; im++) {
#pragma unroll
              for (int in = 0; in < SGemm_t::TileN; in++) {
                C_d[(helper.item_g_m() + im) * n + helper.item_g_n() + in] = tmp[im * SGemm_t::TileN + in];
              }
            }
          });
    });
    e_esimd.wait();
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
static UT_SyclSGemm sUT_SyclSGemm;

}  // namespace sycl_ut
}  // namespace bestla
