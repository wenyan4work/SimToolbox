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

auto partitioning(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                  std::array<double, 3> pbcBox, int comm_rank) {
  srim::Partition part;
  part.setPBCBox(pbcBox);
  part.setParam("imbalance_tolerance", 1.03);
  part.setParam("rectilinear", true);
  std::vector<double> weight(sysA.size());
  for (int i = 0; i < weight.size(); ++i) {
    weight[i] = (double)rand() / RAND_MAX;
  }
  auto solution = part.MJ(sysA, weight);
  if (!solution.oneToOnePartDistribution()) {
    exit(100);
  }
  std::vector<ParA> new_sysA;
  std::vector<ParB> new_sysB;
  part.applyPartition<std::vector<ParB>, ParB>(sysB, new_sysB, solution);
  part.applyPartition<std::vector<ParA>, ParA>(sysA, new_sysA, solution);

  // check the bounding box of each thread, make sure they are not overlapping
  std::vector<ParA> bounding_box(1);
  bounding_box[0].boxlow[0] = FLT_MAX;
  bounding_box[0].boxlow[1] = FLT_MAX;
  bounding_box[0].boxlow[2] = FLT_MAX;
  bounding_box[0].boxhigh[0] = FLT_MIN;
  bounding_box[0].boxhigh[1] = FLT_MIN;
  bounding_box[0].boxhigh[2] = FLT_MIN;
  for (int i = 0; i < new_sysA.size(); ++i) {
    const auto &box = new_sysA[i].getBox();
    for (int j = 0; j < 3; ++j) {
      bounding_box[0].boxlow[j] = std::min(
          bounding_box[0].boxlow[j],
          part.imposePBC(box.first[j], box.second[j], pbcBox[j]) + srim::eps);
      bounding_box[0].boxhigh[j] = std::max(
          bounding_box[0].boxhigh[j],
          part.imposePBC(box.first[j], box.second[j], pbcBox[j]) - srim::eps);
    }
  }

  srim::Srim im;
  if (bounding_box[0].boxlow[0] == FLT_MAX) {
    bounding_box = std::vector<ParA>{};
  }
  const auto &pbc_bvh = im.buildBVH(bounding_box, bounding_box.size());
  const auto &bvh = pbc_bvh.first;
  const auto &shift_map = pbc_bvh.second;
  const auto &query = im.query(bvh, bounding_box, bounding_box.size());
  const auto &dt =
      im.buildDataTransporter(query, bounding_box.size(), shift_map);
  const auto &nb_indices = dt.getNBI();
  const auto &offset = query.first;
  const auto &indices = query.second;
  std::vector<ParA> nb_container;
  dt.updateNBL<std::vector<ParA>, ParA>(bounding_box, nb_container);
  if (nb_container.size() > 1) {
    exit(13);
  }
  return std::make_pair(new_sysA, new_sysB);
}

auto query_tree(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                double r, std::array<double, 3> pbcBox, int comm_rank,
                int comm_size) {
  srim::Srim im;
  im.setPBCBox(pbcBox);
  im.setPBCMax(r);

  const auto &pbc_bvh = im.buildBVH(sysA, sysA.size());
  const auto &bvh = pbc_bvh.first;
  const auto &shift_map = pbc_bvh.second;
  const auto &query = im.query(bvh, sysB, sysB.size());
  const auto &dt = im.buildDataTransporter(query, sysA.size(), shift_map);
  const auto &nb_indices = dt.getNBI();
  std::cout << "Distributed tree query complete" << std::endl;
  const auto &offset = query.first;
  const auto &indices = query.second;
  std::vector<ParA> nb_container;
  dt.updateNBL<std::vector<ParA>, ParA>(sysA, nb_container);
  // expand result to full list with gid
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
  // send all pars and query result to process 0
  std::vector<ParA> sysA_all;
  std::vector<ParB> sysB_all;
  std::vector<std::array<int, 5>> pairs_all;

  gather(pairs, pairs_all, comm_rank, comm_size);
  std::cout << "All data sent to proccess 0" << std::endl;
  sort(pairs_all.begin(), pairs_all.end());
  return pairs_all;
}

void test_check(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                double r, std::array<double, 3> pbcBox, int comm_rank,
                int comm_size) {
  // run partitioning algorithm and get partitioned sysA and sysB
  const auto &partitioned_pars = partitioning(sysA, sysB, pbcBox, comm_rank);
  const auto &new_sysA = partitioned_pars.first;
  const auto &new_sysB = partitioned_pars.second;

  // query the original particles.
  const auto &pairs_all =
      query_tree(sysA, sysB, r, pbcBox, comm_rank, comm_size);

  // query the partitioned particles.
  const auto &new_pairs_all =
      query_tree(new_sysA, new_sysB, r, pbcBox, comm_rank, comm_size);
  // compare result with and without partitioning, and see if the pair list
  // correct.
  if (comm_rank == 0) {
    std::cout << " CountPar " << new_pairs_all.size() << std::endl;
    std::cout << " Count " << pairs_all.size() << std::endl;
    if (pairs_all.size() != new_pairs_all.size()) {
      exit(1);
    }
    for (int i = 0; i < pairs_all.size(); ++i) {
      if (!std::equal(pairs_all[i].begin(), pairs_all[i].end(),
                      new_pairs_all[i].begin())) {
        exit(2);
      }
    }
    std::cout << "Success" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

void testUniform(int parA_count, int parB_count, double r, double low,
                 double high, std::array<double, 3> &pbcBox, int comm_rank,
                 int comm_size) {
  // Generate sysA and sysB
  std::uniform_real_distribution<double> dist(low, high);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);
  par_rand(sysA, r, dist, gen, parA_count * comm_rank);
  par_rand(sysB, r, dist, gen, parB_count * comm_rank);
  std::cout << "Random Generates complete!" << std::endl;
  // test
  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size);
}

void testNonUniform(int parA_count, int parB_count, double r, double m,
                    double s, std::array<double, 3> &pbcBox, int comm_rank,
                    int comm_size) {
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
  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size);
}

void testNonUniformMPI(int parA_count, int parB_count, double r, double m,
                       double s, std::array<double, 3> &pbcBox, int comm_rank,
                       int comm_size) {
  // Generate sysA and sysB
  if (comm_rank == 0) {
    parA_count = 1;
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
  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size);
}

void testSmall(double r, double m, double s, std::array<double, 3> &pbcBox,
               int comm_rank, int comm_size) {
  // Generate sysA and sysB
  int parA_count = (comm_rank + 1) % 2;
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
  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size);
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

  std::array<double, 3> pbcBox{5, 4, 3};
  // uniformly distributed
  testUniform(2000, 3000, 1, -100, 100, pbcBox, comm_rank, comm_size);

  pbcBox[1] = -1;
  // non-uniformly distributed
  testNonUniform(3000, 2000, 1, 5, 2, pbcBox, comm_rank, comm_size);

  pbcBox[2] = -1;
  // MPI non-uniformly distributed
  testNonUniformMPI(2000, 2000, 1, 5, 2, pbcBox, comm_rank, comm_size);

  // less than MPI and threads
  testSmall(1, 5, 2, pbcBox, comm_rank, comm_size);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}