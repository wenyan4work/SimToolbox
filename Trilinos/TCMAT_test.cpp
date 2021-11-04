#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

#include <vector>

#include <mpi.h>

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  {
    auto commRcp = Tpetra::getDefaultComm();
    const int rank = commRcp->getRank();
    const int nProcs = commRcp->getSize();

    // create a sparse matrix with colMap and rowMap
    // CrsMat : nProcs x 2 * nProcs
    // row map: 1 row per rank
    // col map: 2 per rank
    std::vector<TGO> gRowIndexOnLocal = {commRcp->getRank()};
    std::vector<TGO> gColIndexOnLocal = {1 + commRcp->getRank(),
                                         3 + commRcp->getRank()};

    auto rowMapRcp = getTMAPFromLocalSize(gRowIndexOnLocal.size(), commRcp);
    auto colMapRcp = getTMAPFromGlobalIndexOnLocal(gColIndexOnLocal, //
                                                   2 * nProcs, commRcp);

    // entries
    // const Kokkos::View< row_offset_type * > & 	rowPointers,
    // const Kokkos::View< LocalOrdinal * > & 	columnIndices,
    Kokkos::View<TLRO *> rowPointers("rowPointers",
                                     gRowIndexOnLocal.size() + 1);
    rowPointers[0] = 0;
    rowPointers[1] = rowPointers[0] + gColIndexOnLocal.size();
    Kokkos::View<TLO *> columnIndices("columnIndices", rowPointers[1]);
    Kokkos::View<double *> values("values", rowPointers[1]);
    for (long i = 0; i < gColIndexOnLocal.size(); i++) {
      columnIndices[i] = i; // local index
      values[i] = rank;
    }

    Teuchos::RCP<TCMAT> matTestRcp = Teuchos::rcp(
        new TCMAT(rowMapRcp, colMapRcp, rowPointers, columnIndices, values));

    auto domainMapRcp = getTMAPFromLocalSize(2, commRcp);
    auto rangeMapRcp = getTMAPFromLocalSize(1, commRcp);
    matTestRcp->fillComplete(domainMapRcp, rangeMapRcp); // domainMap, rangeMap

    dumpTCMAT(matTestRcp, "matTestRcp");
  }
  MPI_Finalize();
  return 0;
}