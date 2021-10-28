#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr int x = 0;
constexpr int y = 1;
constexpr int z = 2;
constexpr int r = 3;

constexpr double eps = 1e-4;

template <typename DeviceType>
class Boxes {
public:
  // Create non-intersecting boxes on a 3D cartesian grid
  // used both for queries and predicates.
  Boxes(typename DeviceType::execution_space const &execution_space,
        std::vector<std::vector<float>> content, bool is_tree) {
    int n = content.size();
    _boxes = Kokkos::View<ArborX::Box *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "boxes"), n);
    auto boxes_host = Kokkos::create_mirror_view(_boxes);

    for (int i = 0; i < content.size(); ++i) {
      if (is_tree) { // for tree, no radius
        ArborX::Point p_lower{{content[i][x], content[i][y], content[i][z]}};
        ArborX::Point p_upper{{content[i][x], content[i][y], content[i][z]}};
        boxes_host[i] = {p_lower, p_upper};
      } else { // for query, with redius
        ArborX::Point p_lower{{content[i][x] - content[i][r] - eps,
                               content[i][y] - content[i][r] - eps,
                               content[i][z] - content[i][r] - eps}};
        ArborX::Point p_upper{{content[i][x] + content[i][r] + eps,
                               content[i][y] + content[i][r] + eps,
                               content[i][z] + content[i][r] + eps}};
        boxes_host[i] = {p_lower, p_upper};
      }
    }
    Kokkos::deep_copy(execution_space, _boxes, boxes_host);
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
std::vector<std::vector<float>> file_reader(std::string path) {
  std::ifstream data_file(path);
  std::string data;
  getline(data_file, data);
  std::vector<std::vector<float>> content;
  while (getline(data_file, data)) {
    std::istringstream iss(data);
    std::vector<float> xyzr(4);
    iss >> xyzr[x] >> xyzr[y] >> xyzr[z] >> xyzr[r];
    content.push_back(xyzr);
  }
  std::cout << "size: " << content.size() << std::endl;
  data_file.close();
  return content;
}

// do query
std::vector<std::vector<int>>
get_neighbor(std::vector<std::vector<float>> query_data,
             std::vector<std::vector<float>> pts_data, bool reverse) {
  std::vector<std::vector<int>> nbs;
  Kokkos::initialize();
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExecutionSpace execution_space;

    std::cout << "Create grid with bounding boxes" << '\n';
    Boxes<DeviceType> query_boxes(execution_space, query_data, false);
    Boxes<DeviceType> pts_boxes(execution_space, pts_data, true);
    std::cout << "Bounding boxes set up." << '\n';

    std::cout << "Creating BVH tree." << '\n';
    ArborX::BVH<MemorySpace> const tree(execution_space, pts_boxes);
    std::cout << "BVH tree set up." << '\n';

    std::cout << "Starting the queries." << '\n';
    // The query will resize indices and offsets accordingly
    Kokkos::View<int *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);

    ArborX::query(tree, execution_space, query_boxes, indices, offsets);
    std::cout << "Queries done." << '\n';

    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    auto indices_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
    for (int i = 0; i < query_data.size(); ++i) {
      int start = offsets_host(i);
      int end = offsets_host(i + 1);
      for (int j = start; j < end; ++j) {
        int k = indices_host(j);
        if (sqrt(pow(query_data[i][x] - pts_data[k][x], 2) +
                 pow(query_data[i][y] - pts_data[k][y], 2) +
                 pow(query_data[i][z] - pts_data[k][z], 2)) <=
            query_data[i][r]) {
          // reverse means using pts to search query
          nbs.push_back(reverse ? std::vector<int>{k, i}
                                : std::vector<int>{i, k});
        }
      }
    }
  }
  Kokkos::finalize();
  return nbs;
}

// Now that we have encapsulated the objects and queries to be used within the
// Boxes class, we can continue with performing the actual search.
int main() {
  // file path
  std::string query_file = "Query.txt";
  std::string pts_file = "Pts.txt";
  std::vector<std::vector<float>> query_data = file_reader(query_file);
  std::vector<std::vector<float>> pts_data = file_reader(pts_file);
  std::vector<std::vector<int>> nbs = get_neighbor(query_data, pts_data, false);
  std::vector<std::vector<int>> rnbs = get_neighbor(pts_data, query_data, true);
  nbs.insert(nbs.end(), rnbs.begin(), rnbs.end());
  sort(nbs.begin(), nbs.end());
  std::ofstream test_file("test.txt");
  for (int i = 0; i < nbs.size(); ++i) {
    // remove duplicates
    if (i > 0 && nbs[i][0] == nbs[i - 1][0] && nbs[i][1] == nbs[i - 1][1])
      continue;
    test_file << nbs[i][0] + 1 << " " << nbs[i][1] + 1 << std::endl;
  }
  test_file.close();
}