#include "Trilinos/TpetraUtil.hpp"
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_oblackholestream.hpp>

// Tpetra container
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrixTransposer_decl.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

// int main(int argc, char **argv) {
//     MPI_Init(&argc, &argv);
//     int rank, nprocs;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

//     MPI_Finalize();
// }

// Teuchos utility

void test() {

    using TMAP = Tpetra::Map<int, int>;          ///< default Teuchos::Map type
    using TV = Tpetra::Vector<double, int, int>; ///< default to Tpetra::Vector type

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int localSize1 = 1;
    const int localSize2 = 2;
    const int globalSize1 = localSize1 * nprocs;
    const int globalSize2 = localSize2 * nprocs;

    std::vector<int> vecGlobalIndexOnLocal(localSize1 + localSize2);
    for (int i = 0; i < localSize1; i++) {
        vecGlobalIndexOnLocal[i] = rank * localSize1 + i;
    }
    for (int i = 0; i < localSize2; i++) {
        vecGlobalIndexOnLocal[i + localSize1] = rank * localSize2 + i + globalSize1;
    }
    auto commRcp = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
    auto map = Teuchos::rcp(
        new TMAP(globalSize1 + globalSize2, vecGlobalIndexOnLocal.data(), localSize1 + localSize2, 0, commRcp));

    auto vec = Teuchos::rcp(new TV(map, true));
    for (int i = 0; i < localSize1 + localSize2; i++) {
        vec->replaceLocalValue(i, rank);
    }

    Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();

    Tpetra::MatrixMarket::Writer<TV> writer;
    writer.writeMapFile("map.mtx", *map);
    dumpTV(vec, "vec");
}

void test2() {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int localSize1 = 5 * rank;       // [0,5,10,15,...]
    const int localSize2 = 3 * (rank + 1); // [3,6,9,12,15,...]

    auto commRcp = getMPIWORLDTCOMM();
    auto map1 = getTMAPFromLocalSize(localSize1, commRcp);
    auto map2 = getTMAPFromLocalSize(localSize2, commRcp);

    TEUCHOS_TEST_FOR_EXCEPTION(!map1->isContiguous(), std::invalid_argument, "map1 must be contiguous");
    TEUCHOS_TEST_FOR_EXCEPTION(!map2->isContiguous(), std::invalid_argument, "map2 must be contiguous")

    std::vector<double> vecOnLocal(localSize1 + localSize2);
    int count = 1;
    for (auto &v : vecOnLocal) {
        v = (count *= (rank + 1));
    }

    auto vec = getTVFromVector(vecOnLocal, commRcp);

    auto vecSubView1 = vec->offsetView(map1, 0);
    auto vecSubView2 = vec->offsetViewNonConst(map2, localSize1);

    auto vecSubView2Ptr = vecSubView2->getLocalView<Kokkos::HostSpace>();
    vecSubView2->modify<Kokkos::HostSpace>();
    for (int i = 0; i < vecSubView2Ptr.dimension_0(); i++) {
        vecSubView2Ptr(i, 0) = i;
    }

    dumpTV(vec, "vec");
    dumpTV(vecSubView1, "vecSubView1");
    dumpTV(vecSubView2, "vecSubView2");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    // test();
    test2();

    MPI_Finalize();
    return 0;
}