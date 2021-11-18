/**
 * @file srim.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief short-range interaction manager
 * @version 0.1
 * @date 2021-08-27
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SRIM_HPP_
#define SRIM_HPP_

#include "Trilinos/TpetraUtil.hpp"

#include <ArborX.hpp>
#include <Zoltan2_BasicVectorAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
// #include <Zoltan2_XpetraMultiVectorAdapter.hpp>

#include <array>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <omp.h>

/**
 * @brief Contains methods to access box object
 *
 */
namespace ArborX {

/**
 * @brief Method to access box object
 *
 * For creating the bounding volume hierarchy given a Boxes object, we
 * need to define the memory space, how to get the total number of objects,
 * and how to access a specific box. Since there are corresponding functions in
 * the Boxes class, we just resort to them.
 *
 * @tparam Boxes
 * @tparam DeviceType
 * @tparam ObjIter
 */
template <template <typename, typename> class Boxes, typename DeviceType,
          typename ObjIter>
struct AccessTraits<Boxes<DeviceType, ObjIter>, PrimitivesTag> {
  using memory_space = typename DeviceType::memory_space;
  /**
   * @brief Returns the size of the boxes.
   *
   * @param boxes
   * @return KOKKOS_FUNCTION
   */
  static KOKKOS_FUNCTION int size(Boxes<DeviceType, ObjIter> const &boxes) {
    return boxes.size();
  }

  /**
   * @brief Returns a single box object with index i in boxes.
   *
   * @param boxes
   * @param i
   * @return KOKKOS_FUNCTION
   */
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType, ObjIter> const &boxes,
                                  int i) {
    return boxes.get_box(i);
  }
};

/**
 * @brief Method to access box object for query
 *
 * For performing the queries given a Boxes object, we need to define memory
 * space, how to get the total number of queries, and what the query with index
 * i should look like. Since we are using self-intersection (which boxes
 * intersect with the given one), the functions here very much look like the
 * ones in ArborX::AccessTraits<Boxes<DeviceType, ObjIter>, PrimitivesTag>.
 *
 * @tparam Boxes
 * @tparam DeviceType
 * @tparam ObjIter
 */
template <template <typename, typename> class Boxes, typename DeviceType,
          typename ObjIter>
struct AccessTraits<Boxes<DeviceType, ObjIter>, PredicatesTag> {
  using memory_space = typename DeviceType::memory_space;
  /**
   * @brief Returns the size of the boxes.
   *
   * @param boxes
   * @return KOKKOS_FUNCTION
   */
  static KOKKOS_FUNCTION int size(Boxes<DeviceType, ObjIter> const &boxes) {
    return boxes.size();
  }

  /**
   * @brief Returns the spatial predicate of a query box in boxes with index i.
   *
   * @param boxes
   * @param i
   * @return KOKKOS_FUNCTION
   */
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType, ObjIter> const &boxes,
                                  int i) {
    return intersects(boxes.get_box(i));
  }
};

} // namespace ArborX

/**
 * @brief Classes and functions related to geometric search and partition.
 *
 */
namespace srim {

constexpr double eps = 1e-5;

using Box = ArborX::Box;
using Point = ArborX::Point;
using Sphere = ArborX::Sphere;
using IdArray = std::array<int, 2>; ///< The ID of each object is consist of
                                    ///< local_id and rank_id.

/**
 * @brief create a new mpi type for data exchange
 *
 * @tparam T
 * @return MPI_Datatype
 */
template <class T>
inline MPI_Datatype createMPIStructType() {
  static_assert(std::is_trivially_copyable<T>::value, "");
  static_assert(std::is_default_constructible<T>::value, "");
  static MPI_Datatype type = MPI_DATATYPE_NULL;
  if (type == MPI_DATATYPE_NULL) {
    MPI_Type_contiguous(sizeof(T), MPI_BYTE, &type);
    MPI_Type_commit(&type);
  }
  return type;
};

/**
 * @brief interface between obj Iterator and ArborX
 *
 * @tparam DeviceType
 * @tparam ObjIter
 */
template <typename DeviceType, typename ObjIter>
class Boxes {
public:
  /**
   * @brief Construct a new Boxes object
   *
   * @param execution_space
   * @param objIter
   * The basic assumption in the constructor is that objects in objIter
   * have a getBox method, which returns the lower bound and upper bound
   * of the box of each object as pair of two array.
   * And it will be used to create a box object.
   * @param N The total number of objects in objIter
   */
  Boxes(typename DeviceType::execution_space const &execution_space,
        const ObjIter &objIter, const int N) {
    _boxes = Kokkos::View<ArborX::Box *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "boxes"), N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      const auto &box = objIter[i].getBox();
      // set lower bound
      Point p_lower{
          {box.first[0] - eps, box.first[1] - eps, box.first[2] - eps}};
      // set upper bound
      Point p_upper{
          {box.second[0] + eps, box.second[1] + eps, box.second[2] + eps}};
      _boxes[i] = {p_lower, p_upper};
    }
  }

  /**
   * @brief Return the number of boxes.
   *
   * @return KOKKOS_FUNCTION
   */
  KOKKOS_FUNCTION int size() const { return _boxes.size(); }

  /**
   * @brief return the box with index i.
   *
   * @param i
   * @return KOKKOS_FUNCTION const&
   */
  KOKKOS_FUNCTION const ArborX::Box &get_box(int i) const { return _boxes(i); }

  Kokkos::View<ArborX::Box *, typename DeviceType::memory_space>
      _boxes; ///< Container that hold all boxes
};

/**
 * @brief Each process directly send the objects to all process that need the
 * object.
 * @details In this function, we assume that each object in in_vec will be send
 * to zero, one, or multiple processes. The destinations of each object are
 * determined using offset and destination vectors.
 * @tparam ObjContainer Container such as std::vector.
 * @tparam ObjType The type of objects that need to be scattered.
 * @param in_vec The input vector contains all local objects.
 * @param out_vec The output vector contains all received objects.
 * @param offset The offset corresponding to the in_vec objects and the
 * destination vector.
 * @param destination The process ranks that each in_vec object needs to be sent
 * to.
 */
template <class ObjContainer, class ObjType>
void forwardScatter(const ObjContainer &in_vec, ObjContainer &out_vec,
                    const std::vector<int> &offset,
                    const std::vector<int> &destination) {
  static_assert(std::is_trivially_copyable<ObjType>::value, "");
  static_assert(std::is_default_constructible<ObjType>::value, "");
  assert(in_vec.size() == offset.size() - 1);

  // get MPI rank and total count.
  MPI_Comm comm_ = MPI_COMM_WORLD;
  auto mpiDataType = createMPIStructType<ObjType>();
  int comm_rank;
  MPI_Comm_rank(comm_, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm_, &comm_size);

  // calculate the length that need to be sent to each rank
  std::vector<int> send_size(comm_size);
  for (int j = 0; j < destination.size(); ++j) {
    ++send_size[destination[j]];
  }

  // re-order the offset and destination array
  // so that it ordered by rank
  std::vector<ObjType> send_buff(destination.size());
  std::vector<int> rank_idx(send_size.size());
  // calculate where each rank should start
  for (int i = 1; i < rank_idx.size(); ++i) {
    rank_idx[i] = rank_idx[i - 1] + send_size[i - 1];
  }

  // then save id of in_vec to obj_to_rank
  for (int i = 0; i < in_vec.size(); ++i) {
#pragma omp parallel for
    for (int j = offset[i]; j < offset[i + 1]; ++j) {
      int k = rank_idx[destination[j]]++;
      send_buff[k] = in_vec[i];
    }
  }

  // gather the needed size to save all data
  std::vector<int> recv_size(comm_size);

  MPI_Alltoall(send_size.data(), 1, MPI_INT, recv_size.data(), 1, MPI_INT,
               comm_);

  std::vector<int> rdsp(comm_size + 1);
  std::vector<int> sdsp(comm_size + 1);
#pragma omp parallel sections
  {
#pragma omp section
    {
      sdsp[0] = 0;
      std::partial_sum(send_size.begin(), send_size.end(), sdsp.begin() + 1);
    }
#pragma omp section
    {
      rdsp[0] = 0;
      std::partial_sum(recv_size.begin(), recv_size.end(), rdsp.begin() + 1);
    }
  }

  // for out_vec
  out_vec.resize(rdsp.back());
  MPI_Alltoallv(send_buff.data(), send_size.data(), sdsp.data(), mpiDataType,
                out_vec.data(), recv_size.data(), rdsp.data(), mpiDataType,
                comm_);
}
/**
 * @brief Each process require a list of elements from other processes.
 * @details In this function, we assume that each object in in_vec will be
 * requested by zero, one, or multiple processes. And the offset and source
 * vector contains the index of objects needed from all processes.
 * @tparam ObjContainer Container such as std::vector.
 * @tparam ObjType The type of objects that need to be scattered.
 * @param in_vec The input vector contains all local objects.
 * @param out_vec The output vector contains all received objects.
 * @param offset The offset corresponding to each process and the source vector.
 * @param source The index of objects needed from each process.
 */
template <class ObjContainer, class ObjType>
void reverseScatter(const ObjContainer &in_vec, ObjContainer &out_vec,
                    const std::vector<int> &offset,
                    const std::vector<int> &source) {
  static_assert(std::is_trivially_copyable<ObjType>::value, "");
  static_assert(std::is_default_constructible<ObjType>::value, "");

  // get MPI rank and total count.
  MPI_Comm comm_ = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm_, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm_, &comm_size);

  assert(offset.size() == comm_size + 1);

  auto mpiDataType = createMPIStructType<ObjType>();

  // caculate the size send to each process
  std::vector<int> rev_send_size(comm_size);
  for (int i = 0; i < rev_send_size.size(); ++i) {
    rev_send_size[i] = offset[i + 1] - offset[i];
  }

  // get the size to receive
  std::vector<int> rev_recv_size(comm_size);
  MPI_Alltoall(rev_send_size.data(), 1, MPI_INT, rev_recv_size.data(), 1,
               MPI_INT, comm_);

  std::vector<int> rev_rdsp(comm_size + 1);
  rev_rdsp[0] = 0;
  std::partial_sum(rev_recv_size.begin(), rev_recv_size.end(),
                   rev_rdsp.begin() + 1);

  // exchange the ids that each node need.
  std::vector<int> rev_recv_buff(rev_rdsp.back());
  MPI_Alltoallv(source.data(), rev_send_size.data(), offset.data(), MPI_INT,
                rev_recv_buff.data(), rev_recv_size.data(), rev_rdsp.data(),
                MPI_INT, comm_);
  std::vector<ObjType> send_buff(rev_rdsp.back());
#pragma omp parallel for
  for (int i = 0; i < rev_recv_buff.size(); ++i) {
    send_buff[i] = in_vec[rev_recv_buff[i]];
  }

  // the rev_send_size is the receive size
  // and the rev_recv_size is the send size
  std::vector<int> sdsp(comm_size + 1);
  std::vector<int> rdsp(comm_size + 1);
#pragma omp parallel sections
  {
#pragma omp section
    {
      sdsp[0] = 0;
      std::partial_sum(rev_recv_size.begin(), rev_recv_size.end(),
                       sdsp.begin() + 1);
    }
#pragma omp section
    {
      rdsp[0] = 0;
      std::partial_sum(rev_send_size.begin(), rev_send_size.end(),
                       rdsp.begin() + 1);
    }
  }
  // for out_vec
  out_vec.resize(rdsp.back());
  MPI_Alltoallv(send_buff.data(), rev_recv_size.data(), sdsp.data(),
                mpiDataType, out_vec.data(), rev_send_size.data(), rdsp.data(),
                mpiDataType, comm_);
}

/**
 * @brief DataTransporter
 *
 * This class handles data update (repeatedly) with index results by srim
 */
class DataTransporter {
private:
  // necessary private data to maintain status
  // in_offsets and in_sources are the offsets and indices for reverseScatter.
  // nb_indices contains the amount of shift in 3D and the local index.
  std::vector<int> in_offsets;
  std::vector<int> in_sources;
  std::vector<std::array<int, 4>> nb_indices;

public:
  // default ctor and copy/move
  DataTransporter() = default;
  DataTransporter(const DataTransporter &) = default;
  DataTransporter(DataTransporter &&) = default;
  DataTransporter &operator=(const DataTransporter &) = default;
  DataTransporter &operator=(DataTransporter &&) = default;

  /**
   * @brief Construct a new Data Transporter object
   * @tparam Query Should be generated by Srim.
   * @tparam ShiftMap Should also be generated by Srim
   * @param query This consists of a pair of the offsets and indices of the BVH
   * tree query result.
   * @param N The total number of local objects.
   * @param shiftMap The local index of the objects and the amount of shift
   * compare to the original coordinates.
   */
  template <class Query, class ShiftMap>
  DataTransporter(const Query &query, int N,
                  const std::vector<ShiftMap> shiftMap) {
    // get the offset and indices from query
    const auto &offset = query.first;
    const auto &indices = query.second;

    // get MPI rank and size
    MPI_Comm comm_ = MPI_COMM_WORLD;
    int comm_rank;
    MPI_Comm_rank(comm_, &comm_rank);
    int comm_size;
    MPI_Comm_size(comm_, &comm_size);

    // create offset of shiftmap
    int shift_map_size = shiftMap.size();
    std::vector<int> shift_map_count(comm_size);
    MPI_Allgather(&shift_map_size, 1, MPI_INT, shift_map_count.data(), 1,
                  MPI_INT, comm_);
    std::vector<int> shift_map_offset(comm_size + 1);
    shift_map_offset[0] = 0;
    std::partial_sum(shift_map_count.begin(), shift_map_count.end(),
                     shift_map_offset.begin() + 1);

    // gather shiftmap from all nodes
    std::vector<ShiftMap> shift_map_all(shift_map_offset.back());
    auto mpiDataType = createMPIStructType<ShiftMap>();
    MPI_Allgatherv(shiftMap.data(), shift_map_size, mpiDataType,
                   shift_map_all.data(), shift_map_count.data(),
                   shift_map_offset.data(), mpiDataType, comm_);

    // make sure we don't require a point multiple times from one process
    int in_vec_size = offset.size() - 1;
    std::vector<IdArray> in_id(N);
    std::vector<IdArray> out_id;

    // fill the id
    for (int i = 0; i < N; ++i) {
      in_id[i][0] = i;
      in_id[i][1] = comm_rank;
    }

    // get the total number of objects in all processes
    std::vector<int> node_count(comm_size);
    MPI_Allgather(&N, 1, MPI_INT, node_count.data(), 1, MPI_INT, comm_);

    std::vector<int> comm_offset(comm_size + 1);
    comm_offset[0] = 0;
    std::partial_sum(node_count.begin(), node_count.end(),
                     comm_offset.begin() + 1);

    // remove duplicate object ids
    std::vector<int> comm_nodes(comm_offset.back());
    for (int i = 0; i < in_vec_size; ++i) {
      int lb = offset(i);
      int ub = offset(i + 1);
#pragma omp parallel for
      for (int j = lb; j < ub; ++j) {
        comm_nodes[comm_offset[indices(j)[1]] +
                   shift_map_all[shift_map_offset[indices(j)[1]] +
                                 indices(j)[0]][3]] = 1;
      }
    }

    // calculate the size
    std::vector<int> recv_size(comm_size);
#pragma omp parallel for
    for (int i = 0; i < comm_size; ++i) {
      recv_size[i] =
          std::accumulate(comm_nodes.begin() + comm_offset[i],
                          comm_nodes.begin() + comm_offset[i + 1], 0);
    }

    // create offset and source. all value will be overwrite, so the initial
    // value doesn't matter
    in_offsets.resize(comm_size + 1);
    in_offsets[0] = 0;
    std::partial_sum(recv_size.begin(), recv_size.end(),
                     in_offsets.begin() + 1);

    // create sourse
    in_sources.resize(in_offsets.back());
#pragma omp parallel for
    for (int i = 0; i < comm_size; ++i) {
      int lb = comm_offset[i];
      int ub = comm_offset[i + 1];
      for (int j = lb, k = in_offsets[i]; j < ub; ++j) {
        if (comm_nodes[j] == 1) {
          in_sources[k] = j - lb;
          ++k;
        }
      }
    }

    // reverseScatter to get the id list
    reverseScatter<std::vector<IdArray>, IdArray>(in_id, out_id, in_offsets,
                                                  in_sources);

    // use lower_bound to find the index.
    // the out_id are guaranteed to be sorted.
    // this part is read-only for out_id, so no lock.
    // if no error while forward scatter, we will always find a index.
    nb_indices.resize(indices.size());

    auto id_comparator = [](const IdArray &x, const IdArray &y) {
      return x[1] == y[1] ? (x[0] < y[0]) : (x[1] < y[1]);
    };

#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
      int j = shift_map_offset[indices(i)[1]] + indices(i)[0];
      nb_indices[i][3] =
          std::lower_bound(out_id.begin(), out_id.end(),
                           IdArray{shift_map_all[j][3], indices(i)[1]},
                           id_comparator) -
          out_id.begin();

      // in case we have an scatter error, which should never happen.
      assert(nb_indices[i][3] < out_id.size());
      assert(shift_map_all[j][3] == out_id[nb_indices[i][3]][0]);
      assert(indices(i)[1] == out_id[nb_indices[i][3]][1]);

      // save the amount of shift
      nb_indices[i][0] = shift_map_all[j][0];
      nb_indices[i][1] = shift_map_all[j][1];
      nb_indices[i][2] = shift_map_all[j][2];
    }
  }

  /**
   * @brief Get neighbor indices
   *
   * @return const std::vector<int>&
   */
  const std::vector<std::array<int, 4>> &getNBI() const { return nb_indices; }

  /**
   * @brief update the neighbor object list
   *
   * @tparam ObjContainer
   * @tparam ObjType
   * @param in_vec The local objects
   * @param out_vec The neighbors for local objects
   */
  template <class ObjContainer, class ObjType>
  void updateNBL(const ObjContainer &in_vec, ObjContainer &out_vec) const {
    reverseScatter<ObjContainer, ObjType>(in_vec, out_vec, this->in_offsets,
                                          this->in_sources);
    return;
  }
};

/**
 * @brief short range interaction manager
 * @details Functions related query distributed tree using ArborX.
 */
class Srim {
private:
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  // necessary private data to maintain status
  double pbcMax = -1; // Max possible length for neighbor searching.
  std::array<double, 3> pbcBox{-1, -1, -1}; // The periodic boundaries
                                            // in 3 dimension, default value -1,
                                            // value <=0 means no boundary.
  std::vector<std::array<int, 3>> directions{
      {1, 0, 0},   {0, 1, 0},   {0, 0, 1},   {-1, 0, 0},  {0, -1, 0},
      {0, 0, -1},  {1, 1, 0},   {1, 0, 1},   {0, 1, 1},   {-1, -1, 0},
      {-1, 0, -1}, {0, -1, -1}, {1, -1, 0},  {-1, 1, 0},  {1, 0, -1},
      {-1, 0, 1},  {0, 1, -1},  {0, -1, 1},  {1, 1, 1},   {1, 1, -1},
      {1, -1, 1},  {-1, 1, 1},  {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1},
      {-1, -1, -1}}; // 26 copy directions
  ExecutionSpace execution_space;

public:
  // constructor
  Srim() {
    if (!Kokkos::is_initialized()) {
      Kokkos::initialize();
    }
  }

  // destructor
  ~Srim() = default;

  // no move, no copy
  Srim(const Srim &) = delete;
  Srim(Srim &&) = delete;
  Srim &operator=(const Srim &) = delete;
  Srim &operator=(Srim &&) = delete;

  /**
   * @brief MPI_Barrier
   *
   */
  void barrier() { MPI_Barrier(MPI_COMM_WORLD); }

  /**
   * @brief Setup pbcBox
   *
   * @param pbcBox_
   */
  void setPBCBox(const std::array<double, 3> &pbcBox_) { pbcBox = pbcBox_; }

  /**
   * @brief Setup pbcMax
   *
   * @param pbcMax_
   */
  void setPBCMax(const double &pbcMax_) { pbcMax = pbcMax_; }

  /**
   * @brief Calculate and update the coordinate of a box with or without
   * periodic boundaries.
   *
   * @param box AborX::Box saved in Boxes.
   * @param id The index of the box, used for shift map.
   * @return std::array<int, 4> Three shift count and one index.
   */
  std::array<int, 4> imposePBC(Box &box, int id) const {
    // apply pbc on box, fit in [0,pbcBox)
    std::array<int, 4> shift{0, 0, 0, id};
    for (int i = 0; i < pbcBox.size(); ++i) {
      if (pbcBox[i] > 0) {
        // calculate the center of the box as the coordinate of the box
        // each time move pbcBox[i] length
        // so each time the amount of shift change by 1
        double center = (box._min_corner[i] + box._max_corner[i]) / 2.;

        // shift to larger coordinate if center < 0
        while (center < 0) {
          box._min_corner[i] += pbcBox[i];
          box._max_corner[i] += pbcBox[i];
          center = (box._min_corner[i] + box._max_corner[i]) / 2.;
          ++shift[i];
        }

        // shift to smaller coordinate if center >= pbcBox[i];
        while (center >= pbcBox[i]) {
          box._min_corner[i] -= pbcBox[i];
          box._max_corner[i] -= pbcBox[i];
          center = (box._min_corner[i] + box._max_corner[i]) / 2.;
          --shift[i];
        }
      }
    }
    return shift;
  }

  /**
   * @brief Build the distributed BVH tree.
   * @details A distributed tree is built across the processes, with possible
   * copies to make sure it works with PBC.
   * @tparam ObjIter
   * @param objIter A container such as vector and contains objects.
   * @param N The length of the objIter.
   * @return auto Return a pair of the distributed tree and the shift_map of all
   * boxes in the tree.
   */
  template <class ObjIter>
  auto buildBVH(const ObjIter &objIter, const int N) const {
    // build boxes
    Boxes<DeviceType, ObjIter> pts_boxes(execution_space, objIter, N);
    auto &boxes = pts_boxes._boxes;
    std::vector<std::array<int, 4>> shift_map(boxes.size());
#pragma omp parallel for
    for (int i = 0; i < boxes.size(); ++i) {
      shift_map[i] = imposePBC(boxes(i), i);
    }

    // loop over all boxes already exist
    // check if any of the 26 directions need a copy
    // make a copy if needed.
    int k = N;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < directions.size(); ++j) {
        // check all three dimentions, determin if a copy is needed.
        bool need_copy = true;
        for (int d = 0; d < directions[j].size(); ++d) {
          if ((pbcBox[d] <= 0 && directions[j][d] != 0) ||
              (directions[j][d] == 1 &&
               pbcBox[d] - boxes(i)._max_corner[d] > pbcMax) ||
              (directions[j][d] == -1 && boxes(i)._min_corner[d] >= pbcMax)) {
            need_copy = false;
            break;
          }
        }
        // copy the box if need_copy is true
        if (need_copy) {
          if (k >= boxes.size()) { // double the size if space is not enough
            int new_size = boxes.size() * 2;
            Kokkos::resize(boxes, new_size);
            shift_map.resize(new_size);
          }
          // copy shift_map and box.
          shift_map[k] = shift_map[i];
          boxes[k] = boxes[i];

          // move the box to the new coordinates
          // and change the shift map accordingly
          for (int d = 0; d < directions[j].size(); ++d) {
            shift_map[k][d] -= directions[j][d];
            boxes[k]._max_corner[d] -= directions[j][d] * pbcBox[d];
            boxes[k]._min_corner[d] -= directions[j][d] * pbcBox[d];
          }
          ++k;
        }
      }
    }
    Kokkos::resize(boxes, k);
    shift_map.resize(k);
    // build tree
    ArborX::DistributedTree<MemorySpace> distributed_tree(
        MPI_COMM_WORLD, execution_space, pts_boxes);
    return std::make_pair(distributed_tree, shift_map);
  }

  /**
   * @brief Query the distributed tree
   *
   * @tparam bvhType
   * @tparam ObjIter
   * @param bvh The bvh tree, can be generated with buildBVH function.
   * @param objIter A container such as vector and contains objects.
   * @param N The length of the objIter.
   * @return auto The query result, a pair of offsets and indices.
   */
  template <class bvhType, class ObjIter>
  auto query(bvhType &bvh, const ObjIter &objIter, const int N) const {

    // build box
    Boxes<DeviceType, ObjIter> query_boxes(execution_space, objIter, N);
    auto &boxes = query_boxes._boxes;
#pragma omp parallel for
    for (int i = 0; i < boxes.size(); ++i) {
      imposePBC(boxes(i), i);
    }
    // query
    Kokkos::View<IdArray *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    bvh.query(execution_space, query_boxes, indices, offsets);
    return std::make_pair(offsets, indices);
  }

  /**
   * @brief Build a DataTransporter class
   *
   * @tparam Query
   * @tparam ShiftMap
   * @param query The query result, can be generated with query function.
   * @param N The count of local objects.
   * @param shiftMap The shift map of all boxes in the BVH tree, can be
   * generated with buildBVH function.
   * @return DataTransporter
   */
  template <class Query, class ShiftMap>
  DataTransporter
  buildDataTransporter(const Query &query, int N,
                       const std::vector<ShiftMap> &shiftMap) const {
    return DataTransporter(query, N, shiftMap);
  };
};

/**
 * @brief Partitioning algorithm based on Zoltan2
 *
 */
class Partition {
private:
  std::array<double, 3> pbcBox{-1, -1, -1}; // The periodic boundaries
                                            // in 3 dimension, default value -1,
                                            // value <=0 means no boundary.

  /**
   * @brief parameter list for partition method
   *
   */
  Teuchos::RCP<Teuchos::ParameterList> params;

public:
  /**
   * @brief Construct a new Partition object
   *
   */
  Partition() {
    params =
        Teuchos::RCP<Teuchos::ParameterList>(new Teuchos::ParameterList, true);
  }

  /**
   * @brief Setup pbcBox.
   *
   * @param pbcBox_
   */
  void setPBCBox(const std::array<double, 3> &pbcBox_) { pbcBox = pbcBox_; }

  /**
   * @brief Calculate the coordinate of the center of a pair of the lower bound
   * and upper bound with or without periodic boundaries.
   *
   * @param lb The lower bound
   * @param ub The upper bound
   * @param boundary Periodic boundary
   * @return double The center coordinate after applying PBC.
   */
  double imposePBC(double lb, double ub, double boundary) const {
    // apply pbc on center of box, fit in [0,boundary)
    double center = (lb + ub) / 2;
    // if periodic boundary exist, move the center to the range.
    if (boundary > 0) {
      while (center < 0) {
        center += boundary;
      }
      while (center >= boundary) {
        center -= boundary;
      }
    }
    return center;
  }

  /**
   * @brief Set the Param object
   * @details Use when a specific param needs to be changed for the multi-jagged
   * algorithm.
   * @tparam T
   * @param param_name
   * @param param_value
   */
  template <class T>
  void setParam(std::string param_name, T param_value) {
    params->set(param_name, param_value);
  }

  /**
   * @brief The multi-jagged partitioning function.
   *
   * @tparam ObjIter
   * @param objIter A container such as vector and contains objects.
   * @param weights 1D weight vector, it should have the same length as the
   * objIter. By default, there is no weight.
   * @return auto The solution of the partitioning problem.
   */
  template <class ObjIter>
  auto MJ(const ObjIter &objIter, const std::vector<double> &weights = {}) {
    TEUCHOS_ASSERT(weights.size() == objIter.size() || weights.size() == 0);

    // types
    using Teuchos::RCP;
    using Teuchos::rcp;

    // MPI
    auto comm = Tpetra::getDefaultComm();
    int comm_rank = comm->getRank();
    int comm_size = comm->getSize();

    // set up coords, weights and other params
    int coord_dim = 3;
    TLO numLocalPoints = objIter.size();
    auto pointMapRcp = getTMAPFromLocalSize(numLocalPoints, comm);

    using ZTypes = Zoltan2::BasicUserTypes<double, TLO, TGO>;
    using inputAdapter_t = Zoltan2::BasicVectorAdapter<ZTypes>;

    std::vector<double> coords(3 * numLocalPoints);
    std::vector<TGO> globalIds(numLocalPoints);
    auto minGlobalId = pointMapRcp->getMinGlobalIndex();

#pragma omp parallel for
    for (int i = 0; i < numLocalPoints; ++i) {
      const auto &box = objIter[i].getBox();
      for (int j = 0; j < coord_dim; ++j) {
        // calculate the center of box as coordinate
        coords[3 * i + j] = imposePBC(box.first[j], box.second[j], pbcBox[j]);
      }
      globalIds[i] = minGlobalId + i;
    }

    // setup the problem
    inputAdapter_t *ia = nullptr;
    if (weights.size() != 0) {
      TEUCHOS_ASSERT(weights.size() == numLocalPoints);
      ia = new inputAdapter_t(numLocalPoints, globalIds.data(), coords.data(),
                              coords.data() + 1, coords.data() + 2, 3, 3, 3,
                              true, weights.data(), 1);
    } else {
      ia = new inputAdapter_t(numLocalPoints, globalIds.data(), coords.data(),
                              coords.data() + 1, coords.data() + 2, 3, 3, 3);
    }

    params->set("algorithm", "multijagged");
    params->set("mj_keep_part_boxes", true);

    Zoltan2::PartitioningProblem<inputAdapter_t> *problem =
        new Zoltan2::PartitioningProblem<inputAdapter_t>(ia, params.getRawPtr(),
                                                         comm);

    // solve the problem
    problem->solve();
    auto solution = problem->getSolution();

    delete problem;
    delete ia;
    return solution;
  }

  /**
   * @brief Apply the partition solution to objects, and get a new list of
   * objects.
   *
   * @tparam ObjContainer Container such as vector.
   * @tparam ObjType The type of the object.
   * @tparam Solution
   * @param in_vec The original list of objects.
   * @param out_vec The new list of objects.
   * @param solution The solution of the partitioning problem, can be generated
   * with MJ function.
   */
  template <class ObjContainer, class ObjType, class Solution>
  void applyPartition(const ObjContainer &in_vec, ObjContainer &out_vec,
                      const Solution &solution) {
    int in_vec_size = in_vec.size();
    int dim = 3;

    // generate the destination and offset.
    std::vector<int> destination(in_vec_size);
    std::vector<int> offset(in_vec_size + 1);
    for (int i = 0; i < in_vec_size; ++i) {
      double coord[3];
      const auto &box = in_vec[i].getBox();
      for (int j = 0; j < dim; ++j) {
        coord[j] = imposePBC(box.first[j], box.second[j], pbcBox[j]);
      }
      int part = solution.pointAssign(dim, coord);
      destination[i] = part;
      offset[i] = i;
    }
    offset[in_vec_size] = in_vec_size;

    // forwardScatter to get a updated list of objects.
    forwardScatter<ObjContainer, ObjType>(in_vec, out_vec, offset, destination);
  }
};

} // namespace srim
#endif