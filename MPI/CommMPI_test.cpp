#include "CommMPI.hpp"
#include "Util/Logger.hpp"

#include <random>

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

  const auto length = 100000LL;
  std::vector<int> sendRank(length);
  std::vector<Data> sendData(length);
  std::vector<int> recvRank;
  std::vector<Data> recvData;

  auto gen = std::mt19937(rank);
  auto dis = std::uniform_int_distribution<>(0, nProcs - 1);

#pragma omp parallel for
  for (long i = 0; i < length; i++) {
    sendRank[i] = dis(gen);
    sendData[i].fromRank = rank;
    sendData[i].destRank = sendRank[i];
  }
  commMpi.exchangeAllToAllV(sendRank, sendData, recvRank, recvData);
  const int recvSize = recvRank.size();
  assert(recvRank.size() == recvData.size());

  for (long i = 0; i < recvSize; i++) {
    spdlog::info("dest {}, from {}", recvData[i].destRank,
                 recvData[i].fromRank);
    if (recvData[i].destRank != rank)
      spdlog::error("recv rank error");
    if (recvData[i].fromRank != recvRank[i])
      spdlog::error("send rank error");
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  testAllToAllV();
  MPI_Finalize();
  return 0;
}