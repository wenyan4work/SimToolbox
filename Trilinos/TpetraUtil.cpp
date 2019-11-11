#include "TpetraUtil.hpp"

#include <limits>

void dumpTCMAT(const Teuchos::RCP<const TCMAT> &A, std::string filename) {
    filename = filename + std::string("_TCMAT.mtx");
    if (A->getComm()->getRank() == 0) {
        std::cout << "dumping " << filename << std::endl;
    }

    Tpetra::MatrixMarket::Writer<TCMAT> matDumper;
    matDumper.writeSparseFile(filename, A, filename, filename, true);
}

void dumpTV(const Teuchos::RCP<const TV> &A, std::string filename) {
    filename = filename + std::string("_TV.mtx");
    if (A->getMap()->getComm()->getRank() == 0) {
        std::cout << "dumping " << filename << std::endl;
    }

    const auto &fromMap = A->getMap();
    const auto &toMap =
        Teuchos::rcp(new TMAP(fromMap->getGlobalNumElements(), 0, fromMap->getComm(), Tpetra::GloballyDistributed));
    Tpetra::Import<TV::local_ordinal_type, TV::global_ordinal_type, TV::node_type> importer(fromMap, toMap);
    Teuchos::RCP<TV> B = Teuchos::rcp(new TV(toMap, true));
    B->doImport(*A, importer, Tpetra::CombineMode::REPLACE);

    Tpetra::MatrixMarket::Writer<TV> matDumper;
    matDumper.writeDenseFile(filename, B, filename, filename);
}

Teuchos::RCP<const TCOMM> getMPIWORLDTCOMM() { return Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD)); }

// return a contiguous TMAP from local Size
Teuchos::RCP<TMAP> getTMAPFromLocalSize(const int &localSize, Teuchos::RCP<const TCOMM> &commRcp) {
    int globalSize = localSize;
    MPI_Allreduce(MPI_IN_PLACE, &globalSize, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    return Teuchos::rcp(new TMAP(globalSize, localSize, 0, commRcp));
}

Teuchos::RCP<TV> getTVFromVector(const std::vector<double> &in, Teuchos::RCP<const TCOMM> &commRcp) {

    const int localSize = in.size();

    Teuchos::RCP<TMAP> contigMapRcp = getTMAPFromLocalSize(localSize, commRcp);

    Teuchos::RCP<TV> out = Teuchos::rcp(new TV(contigMapRcp, false));

    auto out_2d = out->getLocalView<Kokkos::HostSpace>();
    assert(out_2d.dimension_0() == localSize);

    out->modify<Kokkos::HostSpace>();
    for (int c = 0; c < out_2d.dimension_1(); c++) {
#pragma omp parallel for schedule(dynamic, 1024)
        for (int i = 0; i < out_2d.dimension_0(); i++) {
            out_2d(i, c) = in[i];
        }
    }

    return out;
}