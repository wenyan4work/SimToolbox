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

#include "ArborX.hpp"

#include <algorithm>
#include <array>
#include <mpi.h>
#include <numeric>
#include <omp.h>
#include <vector>

namespace ArborX {

/**
 * @brief
 *
 * @tparam Boxes
 * @tparam DeviceType
 * @tparam ObjIter
 */
template <template <typename, typename> class Boxes, typename DeviceType,
          typename ObjIter>
struct AccessTraits<Boxes<DeviceType, ObjIter>, PrimitivesTag> {
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Boxes<DeviceType, ObjIter> const &boxes) {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType, ObjIter> const &boxes,
                                  int i) {
    return boxes.get_box(i);
  }
};

/**
 * @brief
 *
 * @tparam Boxes
 * @tparam DeviceType
 * @tparam ObjIter
 */
template <template <typename, typename> class Boxes, typename DeviceType,
          typename ObjIter>
struct AccessTraits<Boxes<DeviceType, ObjIter>, PredicatesTag> {
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Boxes<DeviceType, ObjIter> const &boxes) {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType, ObjIter> const &boxes,
                                  int i) {
    return intersects(boxes.get_box(i));
  }
};

} // namespace ArborX

namespace srim {

constexpr double eps = 1e-5;

using Box = ArborX::Box;
using Point = ArborX::Point;
using Sphere = ArborX::Sphere;
using id_array = std::array<int, 2>;

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
  Boxes(typename DeviceType::execution_space const &execution_space,
        const ObjIter &objIter, const int N) {
    _boxes = Kokkos::View<ArborX::Box *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "boxes"), N);
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
      const auto &box = objIter[i].getBox();
      ArborX::Point p_lower{
          {box.first[0] - eps, box.first[1] - eps, box.first[2] - eps}};
      ArborX::Point p_upper{
          {box.second[0] + eps, box.second[1] + eps, box.second[2] + eps}};
      _boxes[i] = {p_lower, p_upper};
    }
  }

  // Return the number of boxes.
  KOKKOS_FUNCTION int size() const { return _boxes.size(); }

  // Return the box with index i.
  KOKKOS_FUNCTION const ArborX::Box &get_box(int i) const { return _boxes(i); }

  Kokkos::View<ArborX::Box *, typename DeviceType::memory_space> _boxes;
};

/**
 * @brief
 *
 * @tparam ObjType
 * @param in_vec
 * @param out_vec
 * @param offset
 * @param destination
 */
template <class ObjContainer, class ObjType>
void forwardScatter(const ObjContainer &in_vec, ObjContainer &out_vec,
                    const std::vector<int> &offset,
                    const std::vector<int> &destination) {
  static_assert(std::is_trivially_copyable<ObjType>::value, "");
  static_assert(std::is_default_constructible<ObjType>::value, "");
  assert(in_vec.size() == offset.size() - 1);

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
 * @brief
 *
 * @tparam ObjType
 * @param in_vec
 * @param out_vec
 * @param offset
 * @param source
 */
template <class ObjContainer, class ObjType>
void reverseScatter(const ObjContainer &in_vec, ObjContainer &out_vec,
                    const std::vector<int> &offset,
                    const std::vector<int> &source) {
  static_assert(std::is_trivially_copyable<ObjType>::value, "");
  static_assert(std::is_default_constructible<ObjType>::value, "");

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
   *
   * @tparam Query
   * @tparam ShiftMap
   * @param query
   * @param N
   * @param shiftMap
   */
  template <class Query, class ShiftMap>
  DataTransporter(const Query &query, int N,
                  const std::vector<ShiftMap> shiftMap) {
    const auto &offset = query.first;
    const auto &indices = query.second;

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
    std::vector<id_array> in_id(N);
    std::vector<id_array> out_id;
    // fill the id
    for (int i = 0; i < N; ++i) {
      in_id[i][0] = i;
      in_id[i][1] = comm_rank;
    }
    std::vector<int> node_count(comm_size);
    MPI_Allgather(&N, 1, MPI_INT, node_count.data(), 1, MPI_INT, comm_);

    std::vector<int> comm_offset(comm_size + 1);
    comm_offset[0] = 0;
    std::partial_sum(node_count.begin(), node_count.end(),
                     comm_offset.begin() + 1);

    // remove duplicates
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
    reverseScatter<std::vector<id_array>, id_array>(in_id, out_id, in_offsets,
                                                    in_sources);

    // use lower_bound to find the index.
    // the out_id are guaranteed to be sorted.
    // this part is read-only for out_id, so no lock.
    // if no error while forward scatter, we will always find a index.
    nb_indices.resize(indices.size());

    auto id_comparator = [](const id_array &x, const id_array &y) {
      return x[1] == y[1] ? (x[0] < y[0]) : (x[1] < y[1]);
    };

#pragma omp parallel for
    for (int i = 0; i < indices.size(); ++i) {
      int j = shift_map_offset[indices(i)[1]] + indices(i)[0];
      nb_indices[i][3] =
          std::lower_bound(
              out_id.begin(), out_id.end(),
              std::array<int, 2>{shift_map_all[j][3], indices(i)[1]},
              id_comparator) -
          out_id.begin();

      // in case we have an scatter error
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
   * @brief get neighbor indices
   *
   * @return const std::vector<int>&
   */
  const std::vector<std::array<int, 4>> &getNBI() const { return nb_indices; }

  /**
   * @brief update the neighbor object list
   *
   * @tparam ObjContainer
   * @tparam ObjType
   * @param in_vec
   * @param out_vec
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
 *
 */
class Srim {
private:
  // necessary private data to maintain status
  int rank = 0;
  int nProcs = 1;
  double pbcMax;
  std::array<double, 3> pbcBox;
  std::vector<std::array<int, 3>> directions{
      {1, 0, 0},   {0, 1, 0},   {0, 0, 1},   {-1, 0, 0},  {0, -1, 0},
      {0, 0, -1},  {1, 1, 0},   {1, 0, 1},   {0, 1, 1},   {-1, -1, 0},
      {-1, 0, -1}, {0, -1, -1}, {1, -1, 0},  {-1, 1, 0},  {1, 0, -1},
      {-1, 0, 1},  {0, 1, -1},  {0, -1, 1},  {1, 1, 1},   {1, 1, -1},
      {1, -1, 1},  {-1, 1, 1},  {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1},
      {-1, -1, -1}};

public:
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  ExecutionSpace execution_space;

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

  void barrier() { MPI_Barrier(MPI_COMM_WORLD); }

  void setPBCBox(const std::array<double, 3> &pbcBox_) { pbcBox = pbcBox_; }

  void setPBCMax(const double &pbcMax_) { pbcMax = pbcMax_; }

  /**
   * @brief
   *
   * @param box
   * @param id
   * @return std::array<int, 4>
   */
  std::array<int, 4> imposePBC(Box &box, int id) const {
    // apply pbc on box, fit in [0,pbcBox)
    std::array<int, 4> shift{0, 0, 0, id};
    for (int i = 0; i < pbcBox.size(); ++i) {
      if (pbcBox[i] > 0) {
        double center = (box._min_corner[i] + box._max_corner[i]) / 2.;
        while (center < 0) {
          box._min_corner[i] += pbcBox[i];
          box._max_corner[i] += pbcBox[i];
          center = (box._min_corner[i] + box._max_corner[i]) / 2.;
          ++shift[i];
        }
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
   * @brief
   *
   * @tparam ObjIter
   * @param objIter
   * @param N
   * @param pbcBox
   * @return auto
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
        // check all three directions, is
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
        if (need_copy) {
          if (k >= boxes.size()) { // double the size if space is not enough;
            int new_size = boxes.size() * 2;
            Kokkos::resize(boxes, new_size);
            shift_map.resize(new_size);
          }
          shift_map[k] = shift_map[i];
          boxes[k] = boxes[i];
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
   * @brief
   *
   * @tparam bvhType
   * @tparam ObjIter
   * @param bvh
   * @param objIter
   * @param N
   * @return auto
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
    Kokkos::View<std::array<int, 2> *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    bvh.query(execution_space, query_boxes, indices, offsets);
    return std::make_pair(offsets, indices);
  }

  template <class Query, class ShiftMap>
  DataTransporter
  buildDataTransporter(const Query &query, int N,
                       const std::vector<ShiftMap> &shiftMap) const {
    return DataTransporter(query, N, shiftMap);
  };
};

} // namespace srim
#endif