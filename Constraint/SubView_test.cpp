#include "Trilinos/TpetraUtil.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    {
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

    MPI_Finalize();
}