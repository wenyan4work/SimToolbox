#include "ZDD.hpp"

#include <cstdio>

#include <mpi.h>

struct data {
    double pos[3] = {0, 0, 0};
};

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    {
        ZDD<data> doubleDataDirectory(100);
        const int nLocal = 100;
        const int nFind = 20;
        doubleDataDirectory.localID.resize(nLocal);
        doubleDataDirectory.localData.resize(nLocal);

        for (int i = 0; i < nLocal; i++) {
            doubleDataDirectory.localID[i] = rank * nLocal + i;
            doubleDataDirectory.localData[i].pos[1] = 100 * (rank * nLocal + i);
        }

        doubleDataDirectory.findID.resize(nFind);
        for (int i = 0; i < nFind; i++) {
            doubleDataDirectory.findID[i] = i - nFind / 2;
        }
        doubleDataDirectory.buildIndex();
        doubleDataDirectory.find();
        for (int i = 0; i < nFind; i++) {
            printf("findID %d, findData %g\n", doubleDataDirectory.findID[i], doubleDataDirectory.findData[i].pos[1]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}