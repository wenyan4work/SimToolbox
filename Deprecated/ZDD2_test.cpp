#include "ZDD2.hpp"

#include <random>

#include <mpi.h>

struct Data {
  long gid;
  double pos[3] = {0, 0, 0};

  Data &operator+=(const Data &other) { return *this; }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  {
    const auto &commRcp = getMPIWORLDTCOMM();
    const int rank = commRcp->getRank();
    const int nProcs = commRcp->getSize();

    Logger::setup_mpi_spdlog();

    ZDD2<Data> doubleDataDirectory;

    // // prepare data on each node
    // const long nLocal = 100;

    // doubleDataDirectory.gidOnLocal.resize(nLocal);
    // doubleDataDirectory.dataOnLocal.resize(nLocal);
    // for (long i = 0; i < nLocal; i++) {
    //   auto gid = rank * nLocal + i;
    //   doubleDataDirectory.gidOnLocal[i] = gid;
    //   doubleDataDirectory.dataOnLocal[i].gid = gid;
    //   doubleDataDirectory.dataOnLocal[i].pos[1] = 100 * (rank * nLocal + i);
    // }

    // // build index
    // doubleDataDirectory.build();

    // // each rank find some data
    // // some gids are invalid(negative), the code should still run, just
    // return
    // // default data
    // const size_t nFind = 20;
    // doubleDataDirectory.gidToFind.resize(nFind);

    // // invalid gid for the first half
    // for (long i = 0; i < nFind / 2; i++) {
    //   doubleDataDirectory.gidToFind[i] = -i;
    // }
    // // valid random gid for the second half
    // // on all ranks:
    // int gidMin = 0;
    // int gidMax = nProcs * nLocal - 1;
    // std::mt19937 gen(rank); // Standard mersenne_twister_engine seeded with
    // rank std::uniform_int_distribution<> dis(gidMin, gidMax);

    // for (long i = nFind / 2; i < nFind; i++) {
    //   doubleDataDirectory.gidToFind[i] = dis(gen);
    // }
    // doubleDataDirectory.find();
    // for (long i = 0; i < nFind; i++) {
    //   auto want = doubleDataDirectory.gidToFind[i];
    //   auto get = doubleDataDirectory.dataToFind[i].gid;
    //   spdlog::debug("gidToFind {}, gidReceived {}", want, get);
    //   if (want != get && want < gidMax && want >= gidMin) {
    //     spdlog::debug("Error: gidToFind {}, gidReceived {}", want, get);
    //   }
    // }

    commRcp->barrier();
  }

  MPI_Finalize();
  return 0;
}