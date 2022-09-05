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

void test() {

    /**
     * vec = [vec1;vec2]. vec, vec1, vec2 are all partitioned across ranks
     * vec1 = [0,1,1,2,2,2,3,3,3,3,...]
     * vec2 = [0,0,1,1,1,1,2,2,2,2,2,2,...]
     * vec = [0,1,1,...,0,0,1,1,1,1,...]
     */

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int localSize1 = 1 * (rank + 1);
    const int localSize2 = 2 * (rank + 1);

    auto commRcp = getMPIWORLDTCOMM();
    auto map1 = getTMAPFromLocalSize(localSize1, commRcp);
    auto map2 = getTMAPFromLocalSize(localSize2, commRcp);

    TEUCHOS_TEST_FOR_EXCEPTION(!map1->isContiguous(), std::invalid_argument, "map1 must be contiguous");
    TEUCHOS_TEST_FOR_EXCEPTION(!map2->isContiguous(), std::invalid_argument, "map2 must be contiguous")

    // create map and vec
    auto map = getTMAPFromTwoBlockTMAP(map1, map2);
    auto vec = Teuchos::rcp(new TV(map, true));

    auto vecSubView1 = vec->offsetViewNonConst(map1, 0);
    auto vecSubView2 = vec->offsetViewNonConst(map2, localSize1);

    auto vecSubView1Ptr = vecSubView1->getLocalView<Kokkos::HostSpace>();
    auto vecSubView2Ptr = vecSubView2->getLocalView<Kokkos::HostSpace>();
    vecSubView1->modify<Kokkos::HostSpace>();
    vecSubView2->modify<Kokkos::HostSpace>();
    for (size_t i = 0; i < vecSubView1Ptr.extent(0); i++) {
        vecSubView1Ptr(i, 0) = rank;
    }
    for (size_t i = 0; i < vecSubView2Ptr.extent(0); i++) {
        vecSubView2Ptr(i, 0) = rank;
    }

    commRcp->barrier();
    dumpTV(vecSubView1, "vecSubView1");
    dumpTV(vecSubView2, "vecSubView2");
    dumpTV(vec, "vec");
    dumpTMAP(map, "map");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    test();

    MPI_Finalize();
    return 0;
}