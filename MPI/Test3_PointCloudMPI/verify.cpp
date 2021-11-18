#include <ArborX.hpp>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

constexpr int x = 0;
constexpr int y = 1;
constexpr int z = 2;
constexpr int r = 3;
constexpr int data_size = 4;
constexpr double eps = 1e-4;

template <typename DeviceType>
class Boxes {
public:
  // Create non-intersecting boxes on a 3D cartesian grid
  // used both for queries and predicates.
  Boxes(typename DeviceType::execution_space const &execution_space,
        float *content, int content_size, bool is_tree) {
    int n = content_size;
    _boxes = Kokkos::View<ArborX::Box *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "boxes"), n);

#pragma omp parallel for
    for (int i = 0; i < content_size * data_size; i += data_size) {
      if (is_tree) { // for tree, no radius
        ArborX::Point p_lower{{content[i + x], content[i + y], content[i + z]}};
        ArborX::Point p_upper{{content[i + x], content[i + y], content[i + z]}};
        _boxes[i / data_size] = {p_lower, p_upper};
      } else { // for query, with redius
        ArborX::Point p_lower{{content[i + x] - content[i + r] - eps,
                               content[i + y] - content[i + r] - eps,
                               content[i + z] - content[i + r] - eps}};
        ArborX::Point p_upper{{content[i + x] + content[i + r] + eps,
                               content[i + y] + content[i + r] + eps,
                               content[i + z] + content[i + r] + eps}};
        _boxes[i / data_size] = {p_lower, p_upper};
      }
    }
  }

  // Return the number of boxes.
  KOKKOS_FUNCTION int size() const { return _boxes.size(); }

  // Return the box with index i.
  KOKKOS_FUNCTION const ArborX::Box &get_box(int i) const { return _boxes(i); }

private:
  Kokkos::View<ArborX::Box *, typename DeviceType::memory_space> _boxes;
};

// For creating the bounding volume hierarchy given a Boxes object, we
// need to define the memory space, how to get the total number of objects,
// and how to access a specific box. Since there are corresponding functions in
// the Boxes class, we just resort to them.
template <typename DeviceType>
struct ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PrimitivesTag> {
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Boxes<DeviceType> const &boxes) {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType> const &boxes, int i) {
    return boxes.get_box(i);
  }
};

// For performing the queries given a Boxes object, we need to define memory
// space, how to get the total number of queries, and what the query with index
// i should look like. Since we are using self-intersection (which boxes
// intersect with the given one), the functions here very much look like the
// ones in ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PrimitivesTag>.
template <typename DeviceType>
struct ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PredicatesTag> {
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Boxes<DeviceType> const &boxes) {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType> const &boxes, int i) {
    return intersects(boxes.get_box(i));
  }
};

// read query and pts files
std::vector<std::array<float, data_size>> file_reader(std::string path) {
  std::ifstream data_file(path);
  std::string data;
  getline(data_file, data);
  std::vector<std::array<float, data_size>> content;
  while (getline(data_file, data)) {
    std::istringstream iss(data);
    std::array<float, data_size> xyzr;
    iss >> xyzr[x] >> xyzr[y] >> xyzr[z] >> xyzr[r];
    content.push_back(xyzr);
  }
  std::cout << "size: " << content.size() << std::endl;
  data_file.close();
  return content;
}

// do query
std::vector<std::array<int, 2>>
get_neighbor(float *query_data, int query_data_count, float *pts_data,
             int pts_data_count, MPI_Comm comm, int comm_rank, bool reverse) {
  std::vector<std::array<int, 2>> nbs;
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExecutionSpace execution_space;

    std::cout << "Create grid with bounding boxes" << '\n';
    Boxes<DeviceType> pts_boxes(execution_space, pts_data, pts_data_count,
                                true);
    Boxes<DeviceType> query_boxes(execution_space, query_data, query_data_count,
                                  false);

    std::cout << "Creating BVH tree." << '\n';
    std::chrono::time_point<std::chrono::system_clock> before =
        std::chrono::system_clock::now();
    ArborX::DistributedTree<MemorySpace> distributed_tree(comm, execution_space,
                                                          pts_boxes);
    MPI_Barrier(MPI_COMM_WORLD);
    std::chrono::time_point<std::chrono::system_clock> after =
        std::chrono::system_clock::now();
    if (comm_rank == 0) {
      std::chrono::duration<float> difference = after - before;
      std::cout << "tree time " << difference.count() << std::endl;
    }

    // sync before moving on, not sure if this matter.
    // The query will resize indices and offsets accordingly

    std::cout << "Starting the queries." << '\n';
    before = std::chrono::system_clock::now();
    Kokkos::View<std::array<int, 2> *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    distributed_tree.query(execution_space, query_boxes, indices, offsets);
    MPI_Barrier(MPI_COMM_WORLD);
    after = std::chrono::system_clock::now();
    if (comm_rank == 0) {
      std::chrono::duration<float> difference = after - before;
      std::cout << "query time " << difference.count() << std::endl;
    }

    // TODO: parallelize this for-loop by openmp
    for (int i = 0; i < query_data_count; ++i) {
      int start = offsets(i);
      int end = offsets(i + 1);
      for (int j = start; j < end; ++j) {
        std::array<int, 2> indice = indices(j);
        int k = indice[0];
        // calculate the real index
        int real_i = i + query_data_count * comm_rank;
        int real_k = k + query_data_count * indice[1];
        nbs.push_back(reverse ? std::array<int, 2>{real_k, real_i}
                              : std::array<int, 2>{real_i, real_k});
      }
    }
  }
  return nbs;
}

// Now that we have encapsulated the objects and queries to be used within the
// Boxes class, we can continue with performing the actual search.
int main(int argc, char **argv) {
  // file path
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  Kokkos::initialize(argc, argv);

  std::cout << "Process " << comm_rank << " of " << comm_size << std::endl;
  int query_data_count, pts_data_count;
  float *query_data_all;
  float *pts_data_all;
  std::vector<std::array<float, data_size>> query_data;
  std::vector<std::array<float, data_size>> pts_data;
  if (comm_rank == 0) {
    std::string query_file = "Query.txt";
    std::string pts_file = "Pts.txt";
    query_data = file_reader(query_file);
    pts_data = file_reader(pts_file);
    int *data_count = (int *)malloc(2 * sizeof(int));
    data_count[0] = query_data.size() / comm_size;
    data_count[1] = pts_data.size() / comm_size;
    query_data_count = data_count[0];
    pts_data_count = data_count[1];
    // send the size to every process
    for (int i = 1; i < comm_size; ++i) {
      MPI_Send(data_count, 2, MPI_INT, i, 0, comm);
    }
    std::cout << "Array size sent to every process" << std::endl;
    // put 2d vector to 1d array.
    query_data_all =
        (float *)malloc(query_data.size() * data_size * sizeof(float));
    pts_data_all = (float *)malloc(pts_data.size() * data_size * sizeof(float));
    for (int i = 0; i < query_data.size(); ++i) {
      for (int j = 0; j < data_size; ++j) {
        query_data_all[i * data_size + j] = query_data[i][j];
      }
    }
    for (int i = 0; i < pts_data.size(); ++i) {
      for (int j = 0; j < data_size; ++j) {
        pts_data_all[i * data_size + j] = pts_data[i][j];
      }
    }
  } else { // get the proper size to save query and pts data
    int *data_count = (int *)malloc(2 * sizeof(int));
    MPI_Recv(data_count, 2, MPI_INT, 0, 0, comm, MPI_STATUS_IGNORE);
    query_data_count = data_count[0];
    pts_data_count = data_count[1];
  }
  std::cout << "Process " << comm_rank << " of " << comm_size << std::endl;
  // create space of size N*4
  float *query_data_part =
      (float *)malloc(query_data_count * data_size * sizeof(float));
  float *pts_data_part =
      (float *)malloc(pts_data_count * data_size * sizeof(float));
  // send data to each process, then delete the array.
  MPI_Scatter(query_data_all, query_data_count * data_size, MPI_FLOAT,
              query_data_part, query_data_count * data_size, MPI_FLOAT, 0,
              comm);
  MPI_Scatter(pts_data_all, pts_data_count * data_size, MPI_FLOAT,
              pts_data_part, pts_data_count * data_size, MPI_FLOAT, 0, comm);

  // each process get all query results

  std::vector<std::array<int, 2>> nbs =
      get_neighbor(query_data_part, query_data_count, pts_data_part,
                   pts_data_count, comm, comm_rank, false);
  std::vector<std::array<int, 2>> rnbs =
      get_neighbor(pts_data_part, pts_data_count, query_data_part,
                   query_data_count, comm, comm_rank, true);
  nbs.insert(nbs.end(), rnbs.begin(), rnbs.end());

  if (comm_rank == 0) { // gather data and process

    std::vector<std::array<int, 2>> query_result;
    query_result.insert(query_result.end(), nbs.begin(), nbs.end());
    for (int i = 1; i < comm_size; ++i) {
      int nbs_count = 0;
      MPI_Recv(&nbs_count, 1, MPI_INT, i, 1, comm, MPI_STATUS_IGNORE);
      int *query_result_part = (int *)malloc(nbs_count * 2 * sizeof(int));
      MPI_Recv(query_result_part, nbs_count * 2, MPI_INT, i, 2, comm,
               MPI_STATUS_IGNORE);
      for (int i = 0; i < nbs_count; ++i) {
        query_result.push_back(
            {query_result_part[2 * i], query_result_part[2 * i + 1]});
      }
    }
    std::sort(query_result.begin(), query_result.end());
    std::ofstream test_file("test.txt");
    for (int i = 0; i < query_result.size(); ++i) {
      int j = query_result[i][0];
      int k = query_result[i][1];
      // remove duplicates
      if (i > 0 && query_result[i][0] == query_result[i - 1][0] &&
          query_result[i][1] == query_result[i - 1][1])
        continue;
      // remove points outside of the sphere
      if (sqrt(pow(query_data[j][x] - pts_data[k][x], 2) +
               pow(query_data[j][y] - pts_data[k][y], 2) +
               pow(query_data[j][z] - pts_data[k][z], 2)) >
          std::max(query_data[j][r], pts_data[k][r]))
        continue;
      test_file << query_result[i][0] + 1 << " " << query_result[i][1] + 1
                << std::endl;
    }
    test_file.close();
  } else { // send data to the major process, since there is no fixed size, we
           // use send instead of gather.
    int nbs_count = nbs.size();
    int *query_result = (int *)malloc(nbs_count * 2 * sizeof(int));
    for (int i = 0; i < nbs_count; ++i) {
      query_result[2 * i] = nbs[i][0];
      query_result[2 * i + 1] = nbs[i][1];
    }
    MPI_Send(&nbs_count, 1, MPI_INT, 0, 1, comm);
    MPI_Send(query_result, 2 * nbs_count, MPI_INT, 0, 2, comm);
  }
  Kokkos::finalize();
  MPI_Finalize();
}