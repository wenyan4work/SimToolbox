// #define ZDDDEBUG
#include "ZDD.hpp"

#include <random>

#include <mpi.h>

struct data {
    int gid;
    double pos[3] = {0, 0, 0};
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int nprocs = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Logger::setup_mpi_spdlog();

    {
        ZDD<data> doubleDataDirectory(100);

        // prepare data on each node
        const size_t nLocal = 100;

        doubleDataDirectory.gidOnLocal.resize(nLocal);
        doubleDataDirectory.dataOnLocal.resize(nLocal);
        for (int i = 0; i < nLocal; i++) {
            auto gid = rank * nLocal + i;
            doubleDataDirectory.gidOnLocal[i] = gid;
            doubleDataDirectory.dataOnLocal[i].gid = gid;
            doubleDataDirectory.dataOnLocal[i].pos[1] = 100 * (rank * nLocal + i);
        }

        // build index
        doubleDataDirectory.buildIndex();

        // each rank find some data
        // some gids are invalid(negative), the code should still run, just return default data
        const size_t nFind = 20;
        doubleDataDirectory.gidToFind.resize(nFind);

        // invalid gid for the first half
        for (int i = 0; i < nFind / 2; i++) {
            doubleDataDirectory.gidToFind[i] = -i;
        }
        // valid random gid for the second half
        // on all ranks:
        int gidMin = 0;
        int gidMax = nprocs * nLocal - 1;
        std::mt19937 gen(rank); // Standard mersenne_twister_engine seeded with rank
        std::uniform_int_distribution<> dis(gidMin, gidMax);

        for (int i = nFind / 2; i < nFind; i++) {
            doubleDataDirectory.gidToFind[i] = dis(gen);
        }
        doubleDataDirectory.find();
        for (int i = 0; i < nFind; i++) {
            auto want = doubleDataDirectory.gidToFind[i];
            auto get = doubleDataDirectory.dataToFind[i].gid;
#ifdef ZDDDEBUG
            printf("gidToFind %d, gidReceived %d\n", want, get);
#endif
            if (want != get && want < gidMax && want >= gidMin) {
                printf("Error: gidToFind %d, gidReceived %d\n", want, get);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}