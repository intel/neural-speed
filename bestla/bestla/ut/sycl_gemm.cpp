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
                // if (sg_idxm + icp * GroupM < TileK)
                {
                  for (size_t in = 0; in < SGemm_t::TileN; in++) {
                    slm_b[(sg_idxm + icp * SGemm_t::WgM) * SGemm_t::WgNEle +
                          (sg_idxn * SGemm_t::SgSize + sgId) * SGemm_t::TileN + in] =
                        B_d[tn + in + (i + sg_idxm + icp * SGemm_t::WgM) * n];
                  }
                  //for (size_t in = 0; in < SGemm_t::TileN; in++) {
                  //  slm_b[(sg_idxm + icp * SGemm_t::WgM) * SGemm_t::WgNEle + (sg_idxn * SGemm_t::SgSize + sgId) +
                  //        in * SGemm_t::WgN] =
                  //      B_d[tn + in + (i + sg_idxm + icp * SGemm_t::WgM) * n];
                  //}
                }
              }

              it.barrier(sycl::access::fence_space::local_space);
#pragma unroll(1)
              for (int ik = 0; ik < SGemm_t::TileK; ik += SGemm_t::UnrollK) {
                float regA[SGemm_t::UnrollK];
                if constexpr (SGemm_t::UnrollK == 8) {
                  *(sycl::vec<float, 4>*)regA = *(sycl::vec<float, 4>*)&A_d[(tm + sgId) * k + (i + ik)];
                  *(sycl::vec<float, 4>*)&regA[4] = *(sycl::vec<float, 4>*)&A_d[(tm + sgId) * k + (i + ik + 4)];
                } else {
                  *(sycl::vec<float, SGemm_t::UnrollK>*)regA =
                      *(sycl::vec<float, SGemm_t::UnrollK>*)&A_d[(tm + sgId) * k + (i + ik)];
                }
#pragma unroll
                for (int ikk = 0; ikk < SGemm_t::UnrollK; ikk++) {
                  float tmpB[SGemm_t::TileN];
#pragma unroll
                  for (int in = 0; in < SGemm_t::TileN; in++) {
                    tmpB[in] =
                        slm_b[sg_idxn * SGemm_t::SgNEle + sgId * SGemm_t::TileN + in + (ik + ikk) * SGemm_t::WgNEle];
                  }

#pragma unroll
                  for (int im = 0; im < SGemm_t::TileM; im++) {
                    auto tmpA = sg.shuffle(regA[ikk], im);
#pragma unroll
                    for (int in = 0; in < SGemm_t::TileN; in++) {
                      tmp[im * SGemm_t::TileN + in] += tmpA * tmpB[in];
                    }
                  }
                }
              }
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
    q->memcpy(matC.data(), C_d, matC.size() * 4).wait();
    buffer_error(ref.data(), matC.data(), ref.size(), 0.001f);
  }
};
static UT_SyclSGemm sUT_SyclSGemm;

}  // namespace sycl_ut
}  // namespace bestla
