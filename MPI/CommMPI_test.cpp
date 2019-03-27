#include "CommMPI.hpp"
#include "Util/TRngPool.hpp"

#include <cstdio>

#include <mpi.h>
#include <omp.h>

struct Data {
    int fromRank;
    int destRank;
};

void testAllToAllV() {
    // prepare data
    CommMPI commMpi;
    const int rank = commMpi.getRank();
    const int nProcs = commMpi.getSize();

    const int length = 100000;
    std::vector<int> sendRank(length);
    std::vector<Data> sendData(length);
    std::vector<int> recvRank;
    std::vector<Data> recvData;

    TRngPool rngPool(length);

#pragma omp parallel for
    for (size_t i = 0; i < length; i++) {
        sendRank[i] = static_cast<int>(rngPool.getU01() * 10 * nProcs) % nProcs; // rng in [0,nProcs)
        sendData[i].fromRank = rank;
        sendData[i].destRank = sendRank[i];
    }
    commMpi.exchangeAllToAllV(sendRank, sendData, recvRank, recvData);
    const int recvSize = recvRank.size();
    assert(recvRank.size() == recvData.size());

    for (int i = 0; i < recvSize; i++) {
        if (recvData[i].destRank != rank)
            printf("recv rank error \n");
        if (recvData[i].fromRank != recvRank[i])
            printf("send rank error \n");
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    testAllToAllV();
    MPI_Finalize();
    return 0;
}