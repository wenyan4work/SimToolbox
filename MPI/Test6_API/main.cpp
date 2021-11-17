#include "MPI/Srim.hpp"

#include <iostream>
#include <random>
#include <vector>

#include <ArborX.hpp>
#include <ArborX_BruteForce.hpp>
#include <Kokkos_Core.hpp>

#include "mpi.h"
#include "omp.h"

constexpr double eps = 1e-5;
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

// transfer data to process 0
template <class Data>
void gather(const std::vector<Data> &in_data, std::vector<Data> &out_data,
            int comm_rank, int comm_size) {
  static MPI_Datatype type = MPI_DATATYPE_NULL;
  if (type == MPI_DATATYPE_NULL) {
    MPI_Type_contiguous(sizeof(Data), MPI_BYTE, &type);
    MPI_Type_commit(&type);
  }
  std::vector<int> data_count(comm_size);
  int n = in_data.size();
  MPI_Gather(&n, 1, MPI_INT, data_count.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> rdsp(data_count.size() + 1);
  rdsp[0] = 0;
  std::partial_sum(data_count.begin(), data_count.end(), rdsp.begin() + 1);
  out_data.resize(rdsp.back());
  MPI_Gatherv(in_data.data(), n, type, out_data.data(), data_count.data(),
              rdsp.data(), type, 0, MPI_COMM_WORLD);
}

void test_check(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                double r, int comm_rank, int comm_size) {
  // query pars with distributed tree
  srim::Srim im;
  std::array<double, 3> pbcBox{-1, -1, -1};
  im.setPBCBox(pbcBox);
  const auto &pbc_bvh = im.buildBVH(sysA, sysA.size());
  const auto &bvh = pbc_bvh.first;
  const auto &shift_map = pbc_bvh.second;
  const auto &query = im.query(bvh, sysB, sysB.size());
  const auto &dt = im.buildDataTransporter(query, sysA.size(), shift_map);
  const auto &nb_indices = dt.getNBI();
  std::vector<ParA> nb_container;
  dt.updateNBL<std::vector<ParA>, ParA>(sysA, nb_container);

  std::cout << "Distributed tree query complete" << std::endl;

  // expand result to full list with gid
  const auto &offset = query.first;
  const auto &indices = query.second;
  std::vector<std::array<int, 2>> pairs(nb_indices.size());
  for (int i = 0; i < sysB.size(); ++i) {
    int lb = offset(i);
    int ub = offset(i + 1);
    for (int j = lb; j < ub; ++j) {
      pairs[j][0] = sysB[i].gid;
      pairs[j][1] = nb_container[nb_indices[j][3]].gid;
    }
  }
  // send all pars and query result to process 0 and compare to brute force.
  std::vector<ParA> sysA_all;
  std::vector<ParB> sysB_all;
  std::vector<std::array<int, 2>> pairs_all;

  gather(sysA, sysA_all, comm_rank, comm_size);
  gather(sysB, sysB_all, comm_rank, comm_size);
  gather(pairs, pairs_all, comm_rank, comm_size);
  std::cout << "All data sent to proccess 0" << std::endl;
  // check results bruteforcely, and see if the pair list correct.
  if (comm_rank == 0) {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExecutionSpace execution_space;

    // query with bruteforce method
    srim::Boxes<DeviceType, std::vector<ParA>> pts_boxes(
        execution_space, sysA_all, sysA_all.size());
    srim::Boxes<DeviceType, std::vector<ParB>> query_boxes(
        execution_space, sysB_all, sysB_all.size());
    ArborX::BruteForce<MemorySpace> bf(execution_space, pts_boxes);
    Kokkos::View<int *, MemorySpace> bf_indices("indices", 0);
    Kokkos::View<int *, MemorySpace> bf_offsets("offsets", 0);
    bf.query(execution_space, query_boxes, bf_indices, bf_offsets);
    std::vector<std::array<int, 2>> pairs_bf(bf_indices.size());
    
    // expand result to full list with gid
    for (int i = 0; i < sysB_all.size(); ++i) {
      int lb = bf_offsets(i);
      int ub = bf_offsets(i + 1);
      for (int j = lb; j < ub; ++j) {
        pairs_bf[j][0] = sysB_all[i].gid;
        pairs_bf[j][1] = sysA_all[bf_indices(j)].gid;
      }
    }

    // sort and compare, this makes sure the result is correct.
    sort(pairs_all.begin(), pairs_all.end());
    sort(pairs_bf.begin(), pairs_bf.end());
    if (!(pairs_all.size() == pairs_bf.size() &&
          std::equal(pairs_all.begin(), pairs_all.end(), pairs_bf.begin()))) {
      exit(1);
    }
    std::cout << "Success" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void testUniform(int parA_count, int parB_count, double r, double low,
                 double high, int comm_rank, int comm_size) {
  // Generate sysA and sysB
  std::uniform_real_distribution<double> dist(low, high);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);
  par_rand(sysA, r, dist, gen, parA_count * comm_rank);
  par_rand(sysB, r, dist, gen, parB_count * comm_rank);
  std::cout << "Random Generates complete!" << std::endl;
  // test
  test_check(sysA, sysB, r, comm_rank, comm_size);
}

void testNonUniform(int parA_count, int parB_count, double r, double m,
                    double s, int comm_rank, int comm_size) {
  // Generate sysA and sysB
  std::lognormal_distribution<double> distA(m, s);
  std::normal_distribution<double> distB(m, s);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);
  par_rand(sysA, r, distA, gen, parA_count * comm_rank);
  par_rand(sysB, r, distB, gen, parB_count * comm_rank);
  std::cout << "Random Generates complete!" << std::endl;
  // test
  test_check(sysA, sysB, r, comm_rank, comm_size);
}

void testNonUniformMPI(int parA_count, int parB_count, double r, double m,
                       double s, int comm_rank, int comm_size) {
  // Generate sysA and sysB
  if (comm_rank == 0) {
    parA_count = 0;
  } else if (comm_rank == 1) {
    parB_count = 0;
  } else if (comm_rank == 2) {
    parA_count = 0;
    parB_count = 0;
  } else {
    parA_count = rand() % (2 * parA_count);
    parB_count = rand() % (2 * parB_count);
  }
  std::lognormal_distribution<double> distA(m, s);
  std::normal_distribution<double> distB(m, s);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);
  par_rand(sysA, r, distA, gen, parA_count * comm_rank * 2);
  par_rand(sysB, r, distB, gen, parB_count * comm_rank * 2);
  std::cout << "Random Generates complete!" << std::endl;
  // test
  test_check(sysA, sysB, r, comm_rank, comm_size);
}

void testSmall(double r, double m, double s, int comm_rank, int comm_size) {
  // Generate sysA and sysB
  int parA_count = comm_rank % 2;
  int parB_count = 1 - parA_count;

  std::lognormal_distribution<double> distA(m, s);
  std::normal_distribution<double> distB(m, s);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);
  par_rand(sysA, r, distA, gen, comm_rank);
  par_rand(sysB, r, distB, gen, comm_rank);
  std::cout << "Random Generates complete!" << std::endl;
  // test
  test_check(sysA, sysB, r, comm_rank, comm_size);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  srand(comm_rank);

  // uniformly distributed
  testUniform(10000, 10000, 1, 0, 10, comm_rank, comm_size);

  // non-uniformly distributed
  testNonUniform(10000, 10000, 1, 5, 2, comm_rank, comm_size);

  // MPI non-uniformly distributed
  testNonUniformMPI(10000, 10000, 1, 5, 2, comm_rank, comm_size);

  // less than MPI and threads
  testSmall(1, 5, 2, comm_rank, comm_size);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}