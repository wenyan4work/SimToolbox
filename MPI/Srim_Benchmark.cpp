#include "Srim.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Tpetra_Core.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>
#include <omp.h>

struct ParA {
  int gid = 0;
  double boxlow[3] = {0, 0, 0};
  double boxhigh[3] = {1, 1, 1};
  double mass = 0, charge = 0;
  double vel[3] = {0, 0, 0};

  std::pair<const double *, const double *> getBox() const {
    return std::make_pair(boxlow, boxhigh);
  }
};

struct ParB {
  int gid = 0;
  double boxlow[3] = {0.5, 0.5, 0.5};
  double boxhigh[3] = {0.6, 0.6, 0.6};
  double mass = 0, charge = 0;
  double vel[3] = {0, 0, 0};
  double acc[3] = {0, 0, 0};
  double force[3] = {0, 0, 0};

  std::pair<const double *, const double *> getBox() const {
    return std::make_pair(boxlow, boxhigh);
  }
};

// generates random pars
template <class Par, class Dist>
void par_rand(std::vector<Par> &pars, double r, Dist &dist, std::mt19937 &gen,
              int start_id) {
  std::uniform_real_distribution<double> distr(0, r);
  for (int i = 0; i < pars.size(); ++i) {
    pars[i].boxlow[0] = dist(gen);
    pars[i].boxlow[1] = dist(gen);
    pars[i].boxlow[2] = dist(gen);
    pars[i].boxhigh[0] = distr(gen) + pars[i].boxlow[0];
    pars[i].boxhigh[1] = distr(gen) + pars[i].boxlow[1];
    pars[i].boxhigh[2] = distr(gen) + pars[i].boxlow[2];
    pars[i].gid = start_id + i;
  }
}

auto partitioning(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                  std::array<double, 3> pbcBox, int comm_rank) {
  srim::Partition part(true);
  part.setPBCBox(pbcBox);
  part.setParam("imbalance_tolerance", 1.03);
  part.setParam("rectilinear", true);

  std::mt19937 gen(comm_rank);
  std::uniform_real_distribution<double> distW(0.1, 10.0);

  std::vector<double> weight(sysA.size());
  for (auto &w : weight) {
    w = distW(gen);
  }

  using Teuchos::TimeMonitor;

  MPI_Barrier(MPI_COMM_WORLD);
  auto solution = part.MJ(sysA, weight);

  std::vector<ParA> new_sysA;
  std::vector<ParB> new_sysB;

  MPI_Barrier(MPI_COMM_WORLD);
  part.applyPartition<std::vector<ParB>, ParB>(sysB, new_sysB, solution);
  part.applyPartition<std::vector<ParA>, ParA>(sysA, new_sysA, solution);

  return std::make_pair(new_sysA, new_sysB);
}

auto query_tree(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                double r, std::array<double, 3> pbcBox, int comm_rank,
                int comm_size, bool print_size) {
  srim::Srim im(true);
  im.setPBCBox(pbcBox);
  im.setPBCMax(r);

  using Teuchos::TimeMonitor;

  MPI_Barrier(MPI_COMM_WORLD);
  const auto &pbc_bvh = im.buildBVH(sysA, sysA.size());
  MPI_Barrier(MPI_COMM_WORLD);

  const auto &bvh = pbc_bvh.first;
  const auto &shift_map = pbc_bvh.second;

  MPI_Barrier(MPI_COMM_WORLD);
  const auto &query = im.query(bvh, sysB, sysB.size());

  MPI_Barrier(MPI_COMM_WORLD);
  const auto &dt = im.buildDataTransporter(query, sysA.size(), shift_map);

  const auto &nb_indices = dt.getNBI();
  const auto &offset = query.first;
  const auto &indices = query.second;

  MPI_Barrier(MPI_COMM_WORLD);
  std::vector<ParA> nb_container;
  dt.updateNBL<std::vector<ParA>, ParA>(sysA, nb_container);

  if (print_size) {
    std::vector<std::array<int, 5>> pairs(nb_indices.size());
    for (int i = 0; i < sysB.size(); ++i) {
      int lb = offset(i);
      int ub = offset(i + 1);
      for (int j = lb; j < ub; ++j) {
        pairs[j][0] = sysB[i].gid;
        pairs[j][1] = nb_container[nb_indices[j][3]].gid;
        pairs[j][2] = nb_indices[j][0];
        pairs[j][3] = nb_indices[j][1];
        pairs[j][4] = nb_indices[j][2];
      }
    }
  }
}

void test_check(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                double r, std::array<double, 3> pbcBox, int comm_rank,
                int comm_size, bool print_size) {
  // run partitioning algorithm and get partitioned sysA and sysB
  if (comm_rank == 0) {
    std::cout << "nProc: " << comm_size
              << ", nThread: " << omp_get_max_threads() << std::endl;
  }
  const auto &partitioned_pars = partitioning(sysA, sysB, pbcBox, comm_rank);
  const auto &new_sysA = partitioned_pars.first;
  const auto &new_sysB = partitioned_pars.second;

  query_tree(new_sysA, new_sysB, r, pbcBox, comm_rank, comm_size, print_size);

  MPI_Barrier(MPI_COMM_WORLD);
}

void testUniform(int parA_count, int parB_count, double r, double low,
                 double high, std::array<double, 3> &pbcBox, int comm_rank,
                 int comm_size, bool print_size) {
  std::mt19937 gen(comm_rank);

  std::uniform_real_distribution<double> distA(low, high);
  std::lognormal_distribution<double> distB(5, 2);

  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);

  // Generate sysA and sysB
  par_rand(sysA, r, distA, gen, parA_count * comm_rank);
  par_rand(sysB, r, distB, gen, parB_count * comm_rank);

  // test
  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size, print_size);
}

/**
 * @brief run this like Benchmark.X N r F
 *
 * example: Benchmark.X 10000 10.0 T
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {

    auto commRcp = Tpetra::getDefaultComm();

    // default parameters
    int test_count = 1000000;
    double r = 0.1;
    bool print_size = false;

    if (argc > 1) {
      test_count = std::atoi(argv[1]); // global total number of objects
    }
    if (argc > 2) {
      r = std::strtod(argv[2], NULL); // search radius
    }
    if (argc > 3) {
      print_size = true;
    }

    if (commRcp->getRank() == 0)
      std::cout << "running with parameters: " << test_count << " " << r << " "
                << print_size << std::endl;

    const int proc_count = test_count / commRcp->getSize();
    std::array<double, 3> pbcBox{10, 10, 10};
    testUniform(proc_count, proc_count, r, -100, 100, pbcBox,
                commRcp->getRank(), commRcp->getSize(), print_size);

    Teuchos::TimeMonitor::summarize();
    MPI_Barrier(MPI_COMM_WORLD);
  }

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
