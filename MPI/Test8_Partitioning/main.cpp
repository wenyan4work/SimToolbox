#include "MPI/Srim.hpp"

#include <ArborX_BruteForce.hpp>

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

template <class Boxes, class Par>
std::vector<std::array<int, 4>>
verifyImposePBC(Boxes &input_boxes, srim::Srim &im,
                std::array<double, 3> &pbcBox, Par &sysA) {
  auto &boxes = input_boxes._boxes;
  std::vector<std::array<int, 4>> shift_map(boxes.size());
#pragma omp parallel for
  for (int i = 0; i < boxes.size(); ++i) {
    shift_map[i] = im.imposePBC(boxes(i), i);
    // verify the result
    for (int j = 0; j < 3; ++j) {
      double center = (boxes[i]._min_corner[j] + boxes[i]._max_corner[j]) / 2.;
      // make sure the center is in range
      if (pbcBox[j] > 0 && (center < 0 || center >= pbcBox[j])) {
        exit(10);
      }

      // make sure the shift_map is correct to get the coordinate from the
      // original coordinate

      int shift = shift_map[i][j];
      float box_low = sysA[i].boxlow[j] - srim::eps;
      float box_high = sysA[i].boxhigh[j] + srim::eps;
      while (shift < 0) {
        box_low -= pbcBox[j];
        box_high -= pbcBox[j];
        ++shift;
      }
      while (shift > 0) {
        box_low += pbcBox[j];
        box_high += pbcBox[j];
        --shift;
      }
      if (box_low != boxes[i]._min_corner[j] ||
          box_high != boxes[i]._max_corner[j]) {
        exit(9);
      }
    }
  }
  return shift_map;
}

void test_check(const std::vector<ParA> &sysA, const std::vector<ParB> &sysB,
                double r, std::array<double, 3> pbcBox, int comm_rank,
                int comm_size) {
  // query pars with distributed tree

  srim::Srim im;
  im.setPBCBox(pbcBox);
  im.setPBCMax(r);
  srim::Partition part;
  part.setPBCBox(pbcBox);
  part.setParam(part.imbalance_tolerance, 1.03);
  // part.setParam(part.rectilinear, true);
  std::vector<double> weight(sysB.size());
  for(int i=0; i<weight.size(); ++i){
    weight[i] = (double)rand()/RAND_MAX;
  }
  auto solution = part.MJ(sysA, weight);
  if(!solution.oneToOnePartDistribution()){
    exit(100);
  }
  std::vector<ParA> new_sysA;
  std::vector<ParB> new_sysB;
  part.applyPartition(sysB, new_sysB, solution);
  std::cout << "Rank" << comm_rank << " Count" << new_sysB.size() << std::endl;
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
  // send all pars and query result to process 0 and compare to brute force.
  std::vector<ParA> sysA_all;
  std::vector<ParB> sysB_all;
  std::vector<std::array<int, 5>> pairs_all;

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
    srim::Boxes<DeviceType, std::vector<ParA>> pts_boxes(
        execution_space, sysA_all, sysA_all.size());
    srim::Boxes<DeviceType, std::vector<ParB>> query_boxes(
        execution_space, sysB_all, sysB_all.size());

    // verify ImposePBC function
    auto pts_shift_map = verifyImposePBC(pts_boxes, im, pbcBox, sysA_all);
    verifyImposePBC(query_boxes, im, pbcBox, sysB_all);

    // verify the query result is the same compare to make all 26 copies
    std::vector<int> dirs = {-1, 0, 1};
    auto &boxes = pts_boxes._boxes;
    int pts_size = boxes.size();
    Kokkos::resize(boxes, pts_size * (pbcBox[0] > 0 ? 3 : 1) *
                              (pbcBox[1] > 0 ? 3 : 1) *
                              (pbcBox[2] > 0 ? 3 : 1));
    pts_shift_map.resize(boxes.size());
    for (int idx = 0, jdx = pts_size; idx < pts_size; ++idx) {
      for (int i = 0; i < dirs.size(); ++i) {
        for (int j = 0; j < dirs.size(); ++j) {
          for (int k = 0; k < dirs.size(); ++k) {
            // copy 26 times, ignore the pbcBox < 0 cases
            if ((i == 1 && j == 1 && k == 1) || (pbcBox[0] <= 0 && i != 1) ||
                (pbcBox[1] <= 0 && j != 1) || (pbcBox[2] <= 0 && k != 1)) {
              continue;
            }

            pts_shift_map[jdx] = pts_shift_map[idx];
            boxes[jdx] = boxes[idx];

            pts_shift_map[jdx][0] += dirs[i];
            pts_shift_map[jdx][1] += dirs[j];
            pts_shift_map[jdx][2] += dirs[k];

            boxes[jdx]._min_corner[0] += pbcBox[0] * dirs[i];
            boxes[jdx]._min_corner[1] += pbcBox[1] * dirs[j];
            boxes[jdx]._min_corner[2] += pbcBox[2] * dirs[k];
            boxes[jdx]._max_corner[0] += pbcBox[0] * dirs[i];
            boxes[jdx]._max_corner[1] += pbcBox[1] * dirs[j];
            boxes[jdx]._max_corner[2] += pbcBox[2] * dirs[k];
            ++jdx;
          }
        }
      }
    }
    ArborX::BruteForce<MemorySpace> bf(execution_space, pts_boxes);
    Kokkos::View<int *, MemorySpace> bf_indices("indices", 0);
    Kokkos::View<int *, MemorySpace> bf_offsets("offsets", 0);
    bf.query(execution_space, query_boxes, bf_indices, bf_offsets);
    std::vector<std::array<int, 5>> pairs_bf(bf_indices.size());
    for (int i = 0; i < sysB_all.size(); ++i) {
      int lb = bf_offsets(i);
      int ub = bf_offsets(i + 1);
      for (int j = lb; j < ub; ++j) {
        pairs_bf[j][0] = sysB_all[i].gid;
        pairs_bf[j][1] = sysA_all[pts_shift_map[bf_indices(j)][3]].gid;
        pairs_bf[j][2] = pts_shift_map[bf_indices(j)][0];
        pairs_bf[j][3] = pts_shift_map[bf_indices(j)][1];
        pairs_bf[j][4] = pts_shift_map[bf_indices(j)][2];
      }
    }
    sort(pairs_all.begin(), pairs_all.end());
    sort(pairs_bf.begin(), pairs_bf.end());
    if (pairs_all.size() != pairs_bf.size()) {
      exit(1);
    }
    for (int i = 0; i < pairs_bf.size(); ++i) {
      if (!std::equal(pairs_all[i].begin(), pairs_all[i].end(),
                      pairs_bf[i].begin())) {
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
  std::uniform_real_distribution<double> dist(low, high);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);
  par_rand(sysA, r, dist, gen, parA_count * comm_rank);
  par_rand(sysB, r, dist, gen, parB_count * comm_rank);
  std::cout << "Random Generates complete!" << std::endl;

  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size);
}

void testNonUniform(int parA_count, int parB_count, double r, double m,
                    double s, std::array<double, 3> &pbcBox, int comm_rank,
                    int comm_size) {
  std::lognormal_distribution<double> distA(m, s);
  std::normal_distribution<double> distB(m, s);
  std::mt19937 gen(comm_rank);
  std::vector<ParA> sysA(parA_count);
  std::vector<ParB> sysB(parB_count);
  par_rand(sysA, r, distA, gen, parA_count * comm_rank);
  par_rand(sysB, r, distB, gen, parB_count * comm_rank);
  std::cout << "Random Generates complete!" << std::endl;

  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size);
}

void testNonUniformMPI(int parA_count, int parB_count, double r, double m,
                       double s, std::array<double, 3> &pbcBox, int comm_rank,
                       int comm_size) {
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

  test_check(sysA, sysB, r, pbcBox, comm_rank, comm_size);
}

void testSmall(double r, double m, double s, std::array<double, 3> &pbcBox,
               int comm_rank, int comm_size) {
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
  pbcBox[1] = -1;
  // uniformly distributed
  testUniform(1000, 1000, 1, -100, 100, pbcBox, comm_rank, comm_size);

  pbcBox[1] = -1;
  // // non-uniformly distributed
  testNonUniform(1000, 1000, 1, 5, 2, pbcBox, comm_rank, comm_size);

  // pbcBox[2] = -1;
  // // // MPI non-uniformly distributed
  // testNonUniformMPI(20, 20, 1, 5, 2, pbcBox, comm_rank, comm_size);

  // // // less than MPI and threads
  // testSmall(1, 5, 2, pbcBox, comm_rank, comm_size);

  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}