
#include <utils/utils.hpp>
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

struct test_params_t {
  // Q: [FxBxNxH] or [BxFxMxH] ; similar for K/V/O
  // BIAS: [1/B, 1/N, 1/F, T]
  bool kUseBias;
  bool kSeqLast;
  uint32_t bs;
  uint32_t hn;
  uint32_t hs;
  uint32_t qlen;
  uint32_t klen;

  static std::vector<test_params_t> cases() {
    std::vector<test_params_t> ret;
    std::vector<std::array<uint32_t, 5>> shapes{
        {1, 32, 64, 1, 33},
        {1, 32, 64, 34, 34},
        {1, 32, 64, 1023, 1023},

        {1, 32, 128, 1, 33},
        {1, 32, 128, 1, 1023},
        {1, 32, 128, 1, 16384},
        {1, 32, 128, 34, 34},
        {1, 32, 128, 34, 1023},
        {1, 32, 128, 1023, 1023},
    };
    for (auto [bs, hn, hs, qlen, klen] : shapes)
      for (auto kUseBias : {false, true})
        for (auto kSeqLast : {false, true})
          ret.emplace_back(kUseBias, kSeqLast, bs, hn, hs, qlen, klen);
    return ret;
  }

  std::string to_string() const {
    std::vector<std::string> params;
    params.push_back(std::string("kUseBias") + (kUseBias ? "ON" : "OFF"));
    params.push_back(std::string("kSeqLast") + (kSeqLast ? "ON" : "OFF"));
    params.push_back("bs" + std::to_string(bs));
    params.push_back("hn" + std::to_string(hn));
    params.push_back("hs" + std::to_string(hs));
    params.push_back("qlen" + std::to_string(qlen));
    params.push_back("klen" + std::to_string(klen));
    return std::accumulate(
        std::next(params.begin()),
        params.end(),
        params[0],
        [](std::string a, std::string b) { return a + '_' + b; });
  }
};

class IFMHATest : public TestWithParam<test_params_t> {
 protected:
  IFMHATest() {}
  ~IFMHATest() {}
  void SetUp() override {}
  void TearDown() override {}
};
TEST_P(IFMHATest, ) {
  // TODO
}
INSTANTIATE_TEST_SUITE_P(
    XeTLA,
    IFMHATest,
    ValuesIn(test_params_t::cases()),
    [](TestParamInfo<test_params_t> info) { return info.param.to_string(); });
