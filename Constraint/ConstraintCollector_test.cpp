#include "ConstraintCollector.hpp"
#include "Trilinos/TpetraUtil.hpp"

#include <memory>
#include <random>

#include <mpi.h>
#include <omp.h>

constexpr long lclMaxParNum = 200;
constexpr long lclMaxConNum = 50;

ConstraintBlockPool genConBlkPool(const TMAP &mobMap) {
  const int nThreads = omp_get_max_threads();
  ConstraintBlockPool conBlkPool(nThreads);
  const auto &commRcp = mobMap.getComm();
  const auto glbParNum = mobMap.getGlobalNumElements() / 6;

  std::uniform_int_distribution<long> udis(0, glbParNum - 1);
  std::uniform_real_distribution<double> ddis(-1, 1);
#pragma omp parallel num_threads(nThreads)
  {
    std::mt19937_64 gen(commRcp->getRank() * nThreads + omp_get_thread_num());
#pragma omp for
    for (auto &que : conBlkPool) {
      const int queSize = gen() % lclMaxConNum;
      for (int j = 0; j < queSize; j++) {
        double normI[3]{ddis(gen), ddis(gen), ddis(gen)};
        double normJ[3]{ddis(gen), ddis(gen), ddis(gen)};
        double posI[3]{ddis(gen), ddis(gen), ddis(gen)};
        double posJ[3]{ddis(gen), ddis(gen), ddis(gen)};
        int gidI = udis(gen);
        int gidJ = udis(gen);
        int globalIndexI = udis(gen);
        int globalIndexJ = udis(gen);
        que.emplace_back(
            ddis(gen), ddis(gen),                               // delta0, gamma
            gidI, gidJ == gidI ? (gidI + 1) % glbParNum : gidJ, // gidI, gidJ
            globalIndexI,
            globalIndexJ == globalIndexI
                ? (globalIndexI + 1) % glbParNum
                : globalIndexJ,           // globalIndexI, globalIndexJ
            normI, normJ,                 // normI, normJ
            posI, posJ,                   // posI, posJ
            posI, posJ,                   // labI, labJ
            ddis(gen) > 0 ? true : false, // oneside
            ddis(gen) > 0 ? true : false, // bilateral
            pow(10, ddis(gen)),           // kappa
            0.);
      }
    }
  }

  //   for (auto &q : conBlkPool) {
  //     spdlog::debug("q.size() {}", q.size());
  //   }
  commRcp->barrier();

  return conBlkPool;
}

void testBuildMatrix() {
  auto commRcp = Tpetra::getDefaultComm();

  std::mt19937_64 gen(commRcp->getRank());
  const auto lclParNum = gen() % lclMaxParNum;

  auto mobMapRcp = getTMAPFromLocalSize(6 * lclParNum, commRcp);

  describe(*mobMapRcp);

  ConstraintCollector conCollector;
  conCollector.constraintPoolPtr =
      std::make_shared<ConstraintBlockPool>(genConBlkPool(*mobMapRcp));

  conCollector.writeConstraintBlockPool(std::string("conBlockPool_r") +
                                            std::to_string(commRcp->getRank()) +
                                            std::string(".msgpack"),
                                        true);

  Teuchos::RCP<TCMAT> DMatTransRcp; //
  Teuchos::RCP<TV> delta0Rcp;       //
  Teuchos::RCP<TV> invKappaRcp;     //
  Teuchos::RCP<TV> biFlagRcp;       //
  Teuchos::RCP<TV> gammaGuessRcp;

  conCollector.buildConstraintMatrixVector(mobMapRcp, DMatTransRcp, delta0Rcp,
                                           invKappaRcp, biFlagRcp,
                                           gammaGuessRcp);

  dumpTCMAT(DMatTransRcp, "DMatTrans");
  dumpTV(delta0Rcp, "delta0");
  dumpTV(invKappaRcp, "invKappa");
  dumpTV(biFlagRcp, "biFlag");
  dumpTV(gammaGuessRcp, "gammaGuess");

  return;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  Logger::set_level(3);

  testBuildMatrix();

  MPI_Finalize();
  return 0;
}