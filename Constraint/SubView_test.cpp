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
        auto totalMap = getTMAPFromLocalSize(localSize1 + localSize2, commRcp);

        std::vector<double> vecOnLocal(localSize1 + localSize2);
        int count = 1;
        for (auto &v : vecOnLocal) {
            v = (count *= (rank + 1));
        }

        auto vec = getTVFromVector(vecOnLocal, commRcp);

        auto vecSubView1 = vec->offsetView(map1, 0);
        auto vecSubView2 = vec->offsetView(map2, localSize1);

        std::cout << vec->description() << std::endl;
        std::cout << vecSubView1->description() << std::endl;
        std::cout << vecSubView2->description() << std::endl;
        dumpTV(vec, "vec");
        dumpTV(vecSubView1, "vecSubView1");
        dumpTV(vecSubView2, "vecSubView2");
    }

    MPI_Finalize();
}