#include "bestla_ut.h"
#include "bestla_prologue_b.h"
#include "sycl_ut.h"
#include "sycl/sycl_device.h"
#include "sycl/sycl_utils.h"
#include "sycl/sycl_storage.h"
#include "sycl/sycl_gemm.h"
#include "sycl/sycl_prologue_b.h"
namespace bestla {
using namespace ut;
using namespace utils;
namespace sycl_ut {
class UT_SyclDevice {
 public:
  UT_SyclDevice() {
    UT_START();
    auto dev = UT_Device::get();
    dev->print();
  }
};
// static UT_SyclDevice sUT_SyclDevice;

class UT_SyclVector {
 public:
  UT_SyclVector() {
    UT_START();
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    auto svec = sycl_utils::sycl_vector<float>(1000, q);
    utils::avector<float> hsrc(1000);
    ut::fill_buffer_randn(hsrc.data(), hsrc.size(), -0.5f, 0.5f);
    q->memcpy(svec.data(), hsrc.data(), hsrc.size() * 4).wait();
    auto hdst = sycl_utils::sycl2host(svec.data(), svec.size(), q);
    ut::buffer_error(hsrc.data(), hdst.data(), hsrc.size(), 0.f);
  }
};
// static UT_SyclVector sUT_SyclVector;

class UT_StorageMemCheck {
 public:
  UT_StorageMemCheck() {
    UT_START();
    ut_s4(4096, 4096, 128, BTLA_DTYPE::S4_CLIP);
  }

  void ut_s4(int n, int k, int blocksize, BTLA_DTYPE qtype, bool asym = false) {
    printf("Test C type Case: %d %d %d %s\n", n, k, blocksize, asym ? "asym" : "sym");
    int ldb = n;
    int kblk_num = utils::updiv(k, blocksize);
    using GemmCore = sycl_gemm::xve::DefaultSGemmCore;
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore, BTLA_ISA::AVX2>;
    PrologueB ProWei;

    auto packedW = ProWei.createStorage(n, k, blocksize, qtype, bestla_dtype<float>, bestla_dtype<utils::bf16>, asym);
    avector<int8_t> buf0(packedW.mSize), buf1(packedW.mSize);
    packedW.assign(buf0.data());
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    sycl_storage::StorageWeightKBlockNInteger sycl_stor(packedW, q);
    storage::gemm::StorageWeightKBlockNInteger tmp = packedW;
    tmp.assign(buf1.data());
    sycl_stor.toHost(tmp, q);
    buffer_error(buf0.data(), buf1.data(), buf0.size());
  }
};
static UT_StorageMemCheck sUT_StorageMemCheck;

class UT_BlockQunatize_S3S4 {
 public:
  UT_BlockQunatize_S3S4() {
    UT_START();
    using GemmCore = sycl_gemm::xve::DefaultSGemmCore;

    ut<GemmCore>(4096, 4096, 32, BTLA_DTYPE::S4_CLIP);
  }
  template <class GemmCore>
  void ut(int n, int k, int blocksize, BTLA_DTYPE QUANT_T, bool isAsym = false) {
    auto constexpr RuntimeISA = BTLA_ISA::AVX2;
    auto dev = UT_Device::get();
    auto q = dev->getQueue();
    printf("%s DType %s %d: %d %d %d Asym:%d\n", __FUNCTION__, utils::bestla_dtype_str(QUANT_T), int(RuntimeISA), n, k,
           blocksize, isAsym);
    utils::aligned_vector<float> raw(n * k);
    ut::fill_buffer_randn(raw.data(), raw.size(), -0.5f, 0.5f);
    using PrologueB = prologue_b::gemm::WeightKBlockNInteger<GemmCore, RuntimeISA>;
    PrologueB kernel;
    auto ptr = kernel.createStorage(n, k, blocksize, QUANT_T, BTLA_DTYPE::F32, BTLA_DTYPE::F32, isAsym);
    avector<int8_t> buffer(ptr.mSize);
    ptr.assign(buffer.data());
    kernel.packTransposeWeight(n, k, raw.data(), k, &ptr, UT_Threading::get());
    sycl_storage::StorageWeightKBlockNInteger sycl_stor(ptr, q);
    avector<float> dequant(n * k, 0);
    using ProB = sycl_prologue_b::WeightS4Trans<GemmCore, float>;
    sycl_utils::sycl_vector<float> dequantB(n * k, q);
    int blks = updiv(k, blocksize);
    auto evt = ProB::dequant_s4<sycl_prologue_b::KernelConfigTrans>(
        n, k, blocksize, {(uint8_t*)sycl_stor.mQBuf.data(), (float*)sycl_stor.mScaleBuf.data(), blks}, dequantB.data(),
        q);
    evt.wait();
    q->memcpy(dequant.data(), dequantB.data(), dequantB.size() * 4).wait();
    ut::buffer_error(raw.data(), dequant.data(), dequant.size(), 0.01f);
  }
};
#ifdef BTLA_UT_PROLOGUE_B
// no proper threshold for this UT
//
#endif
static UT_BlockQunatize_S3S4 sUT_BlockQunatize_S3S4;
}  // namespace sycl_ut
}  // namespace bestla
