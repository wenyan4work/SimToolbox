#include "MPI/Srim.hpp"

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

void testNonUniformMPI(int parA_count, double r, double m, double s,
                       std::array<double, 3> &pbcBox, int comm_rank,
                       int comm_size, std::string file_name) {
  // Generate sysA
  if (comm_rank == 0) {
    parA_count = 1;
  } else {
    parA_count = rand() % (comm_rank * parA_count);
  }
  std::lognormal_distribution<double> distA(m, s);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  par_rand(sysA, r, distA, gen, parA_count * comm_rank * 2);
  std::cout << "Random Generates complete!" << std::endl;

  // test
  // run partitioning algorithm and get partitioned sysA
  srim::Partition part;
  part.setPBCBox(pbcBox);
  part.setParam("imbalance_tolerance", 1.03);
  part.setParam("rectilinear", true);

  // generate weight
  std::vector<double> weight(sysA.size());
  for (int i = 0; i < weight.size(); ++i) {
    weight[i] = (double)rand() / RAND_MAX;
  }

  // if weight not used, then just part.MJ(sysA)
  auto solution = part.MJ(sysA, weight);
  if (!solution.oneToOnePartDistribution()) {
    exit(100);
  }
  // get partition
  std::vector<ParA> new_sysA;

  // if there is a weight, you can send the weight when applyPartition.
  const std::vector<double> &new_weight =
      part.applyPartition<std::vector<ParA>, ParA>(sysA, new_sysA, solution,
                                                   weight);

  // create solA to save on file
  // x, y, z, weight,rank
  int attribute_count = 5;
  std::vector<double> solA(new_sysA.size() * attribute_count);
  for (int i = 0; i < new_sysA.size(); ++i) {
    const auto &box = new_sysA[i].getBox();
    int k = i * attribute_count;
    // save the center of each box with pbc applied
    for (int j = 0; j < 3; ++j) {
      solA[k + j] = part.imposePBC(box.first[j], box.second[j], pbcBox[j]);
    }
    // save weight
    solA[k + 3] = new_weight[i];
    // save rank
    solA[k + 4] = comm_rank;
  }

  // gather the solutions to rank 0
  std::vector<double> solA_all;
  std::vector<double> solB_all;
  gather(solA, solA_all, comm_rank, comm_size);
  if (comm_rank == 0) {
    std::ofstream solA_file(file_name + "_A.csv");
    for (int i = 0; i < solA_all.size(); i += 5) {
      for (int j = i; j < i + 4; ++j) {
        solA_file << solA_all[j] << " ";
      }
      solA_file << solA_all[i + 4] << std::endl;
    }
    solA_file.close();
  }
  MPI_Barrier(MPI_COMM_WORLD);
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

  std::array<double, 3> pbcBox{7, 8, 9};

  // MPI non-uniformly distributed
  testNonUniformMPI(10000, 1, 0, 0.1, pbcBox, comm_rank, comm_size,
                    "NonUniformMPI");
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}