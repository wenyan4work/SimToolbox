#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_InputTraits.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningSolution.hpp>
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>

#include <random>
#include <vector>

void testPartition() {
  auto commRcp = Tpetra::getDefaultComm();
  const int rank = commRcp->getRank();
  const int nprocs = commRcp->getSize();

  using MyTypes = Zoltan2::BasicUserTypes<double, TLO, TGO>;
  using InputAdapter = Zoltan2::BasicVectorAdapter<MyTypes>;
  using Part = InputAdapter::part_t;
  using EvalPartition = Zoltan2::EvaluatePartition<InputAdapter>;

  const int localCount = 40;

  std::mt19937_64 gen(rank);
  std::uniform_int_distribution<int> uint_dis(0, localCount * nprocs - 1);
  std::uniform_real_distribution<double> real_dis(0, 10);

  // Create input data.
  int dim = 3;
  std::vector<double> coords(localCount * 3);

  for (auto &c : coords) {
    c = real_dis(gen);
  }

  // Create global ids for the coordinates.
  std::vector<TGO> globalIds(localCount);
  TGO offset = rank * localCount;
  for (auto &gid : globalIds) {
    gid = offset++;
  }

  // Create parameters for an RCB problem
  double tolerance = 1.05;
  if (rank == 0)
    std::cout << "Imbalance tolerance is " << tolerance << std::endl;
  Teuchos::ParameterList params("test params");
  params.set("debug_level", "basic_status");
  params.set("debug_procs", "0");
  params.set("error_check_level", "debug_mode_assertions");
  params.set("algorithm", "multijagged");
  params.set("imbalance_tolerance", tolerance);
  params.set("num_global_parts", nprocs);

  {
    InputAdapter ia1(localCount, globalIds.data(), coords.data(),
                     coords.data() + 1, coords.data() + 2, 3, 3, 3);
    spdlog::info("adater created");

    Zoltan2::PartitioningProblem<InputAdapter> problem(&ia1, &params);
    problem.solve();
    spdlog::info("partition solved");

    EvalPartition eval(&ia1, &params, &(problem.getSolution()));
    spdlog::info("partition solved");

    if (rank == 0) {
      eval.printMetrics(std::cout);
    }
    if (rank == 0) {
      double imb = eval.getObjectCountImbalance();
      if (imb <= tolerance)
        std::cout << "pass: " << imb << std::endl;
      else
        std::cout << "fail: " << imb << std::endl;
      std::cout << std::endl;
    }
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  Kokkos::initialize(argc, argv);
  testPartition();
  MPI_Finalize();
}
