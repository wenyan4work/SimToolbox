#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

#include <vector>

#include <mpi.h>

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    {
        Logger::setup_mpi_spdlog();
        auto commRcp = getMPIWORLDTCOMM();

        std::vector<GO> gColIndexOnLocal = {1 + commRcp->getRank(), 3 + commRcp->getRank()};
        std::vector<GO> gRowIndexOnLocal = {commRcp->getRank() + 1};

        auto rowMapRcp = getTMAPFromLocalSize(1, commRcp);
        auto colOpMapRcp = getTMAPFromLocalSize(gColIndexOnLocal.size(), commRcp);
        auto colMapRcp = Teuchos::rcp<TMAP>(new TMAP(5LL, gColIndexOnLocal.data(), 2, 0, commRcp));

        // entries
        Kokkos::View<size_t *> rowPointers("rowPointers", gRowIndexOnLocal.size() + 1);
        rowPointers[0] = 0;
        rowPointers[1] = rowPointers[0] + gColIndexOnLocal.size();
        Kokkos::View<int *> columnIndices("columnIndices", rowPointers[1]);
        Kokkos::View<double *> values("values", rowPointers[1]);
        columnIndices[0] = gColIndexOnLocal[0];
        columnIndices[1] = gColIndexOnLocal[1];
        values[0] = gColIndexOnLocal[0];
        values[1] = gColIndexOnLocal[1];

        auto &colmap = *colMapRcp;
        const int colIndexCount = gColIndexOnLocal.size();
#pragma omp parallel for
        for (int i = 0; i < colIndexCount; i++) {
            columnIndices[i] = colmap.getLocalElement(columnIndices[i]);
        }

        Teuchos::RCP<TCMAT> matTestRcp =
            Teuchos::rcp(new TCMAT(rowMapRcp, colMapRcp, rowPointers, columnIndices, values));
        matTestRcp->fillComplete(colOpMapRcp, rowMapRcp); // domainMap, rangeMap

        dumpTCMAT(matTestRcp, "matTestRcp");
    }
    MPI_Finalize();
    return 0;
}