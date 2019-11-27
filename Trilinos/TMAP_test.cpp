#include "TpetraUtil.hpp"

#include <random>

void test() {
    const int localSize = 2;
    auto commRcp = getMPIWORLDTCOMM();
    // uniform, contiguous map
    auto map1 = Teuchos::rcp(new TMAP(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), localSize, 0, commRcp));
    dumpTMAP(map1, "map1");
    // non uniform, contiguous map
    auto map2 = getTMAPFromLocalSize(localSize + commRcp->getRank(), commRcp);
    dumpTMAP(map2, "map2");
    // non uniform, non-contiguous, uniquely owned map
    auto map3 = getTMAPFromTwoBlockTMAP(map1, map2);
    dumpTMAP(map3, "map3");
    // not uniquely owned map
    const int globalSize = localSize * commRcp->getSize();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, globalSize - 1);
    std::vector<int> gidOnLocal(localSize);
    for (auto &v : gidOnLocal) {
        v = dis(gen);
    }
    std::sort(gidOnLocal.begin(), gidOnLocal.end());
    auto map4 = getTMAPFromGlobalIndexOnLocal(gidOnLocal, globalSize, commRcp);
    dumpTMAP(map4, "map4");
    auto vec4 = Teuchos::rcp(new TV(map4, true));
    dumpTV(vec4, "vec4");
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    test();
    MPI_Finalize();
    return 0;
}