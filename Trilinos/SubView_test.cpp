#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

void test() {

  /**
   * vec = [vec1;vec2]. vec, vec1, vec2 are all partitioned across ranks
   * vec1 = [0,1,1,2,2,2,3,3,3,3,...]
   * vec2 = [0,0,1,1,1,1,2,2,2,2,2,2,...]
   * vec  = [0,1,1,...,0,0,1,1,1,1,...]
   */

  auto commRcp = Tpetra::getDefaultComm();
  const int rank = commRcp->getRank();
  const int nprocs = commRcp->getSize();
  const int localSize1 = 1 * (rank + 1);
  const int localSize2 = 2 * (rank + 1);

  auto map1 = getTMAPFromLocalSize(localSize1, commRcp);
  auto map2 = getTMAPFromLocalSize(localSize2, commRcp);

  TEUCHOS_TEST_FOR_EXCEPTION(!map1->isContiguous(), std::invalid_argument,
                             "map1 must be contiguous");
  TEUCHOS_TEST_FOR_EXCEPTION(!map2->isContiguous(), std::invalid_argument,
                             "map2 must be contiguous")

  // create map and vec
  auto map = getTMAPFromTwoBlockTMAP(map1, map2);
  auto vec = Teuchos::rcp(new TV(map, true));

  auto vecSubView1 = vec->offsetViewNonConst(map1, 0);
  auto vecSubView2 = vec->offsetViewNonConst(map2, localSize1);

  auto vecSubView1Ptr = vecSubView1->getLocalView<Kokkos::HostSpace>();
  auto vecSubView2Ptr = vecSubView2->getLocalView<Kokkos::HostSpace>();
  vecSubView1->modify<Kokkos::HostSpace>();
  vecSubView2->modify<Kokkos::HostSpace>();
  assert(vecSubView1Ptr.extent(0) == localSize1);
  assert(vecSubView2Ptr.extent(0) == localSize2);
  for (long i = 0; i < localSize1; i++) {
    vecSubView1Ptr(i, 0) = rank;
  }
  for (long i = 0; i < localSize2; i++) {
    vecSubView2Ptr(i, 0) = rank + nprocs;
  }

  commRcp->barrier();

  // check content of vec
  auto vecView = vec->getLocalView<Kokkos::HostSpace>();
  const auto localSize = vecView.extent(0);
  assert(localSize == localSize1 + localSize2);
  for (long i = 0; i < localSize1; i++) {
    if (vecView(i, 0) != rank)
      printf("Error %d %g\n", i, vecView(i, 0));
  }
  for (long i = 0; i < localSize2; i++) {
    if (vecView(i + localSize1, 0) != rank + nprocs)
      printf("Error %d %g\n", i, vecView(i + localSize1, 0));
  }

  dumpTV(vecSubView1, "vecSubView1");
  dumpTV(vecSubView2, "vecSubView2");
  dumpTV(vec, "vec");
  dumpTMAP(map, "map");
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();

  test();

  MPI_Finalize();
  return 0;
}