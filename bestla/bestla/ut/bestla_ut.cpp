#include <stdio.h>
#include <bestla_parallel.h>

namespace bestla {
namespace ut {
#ifdef _OPENMP
parallel::OMPThreading DefaultThreading(4);
#else
parallel::StdThreading DefaultThreading(4);
#endif  // _OPNEMP
}  // namespace ut
}  // namespace bestla
int main() {
  printf("BesTLA UT done\n");
  return 0;
}
