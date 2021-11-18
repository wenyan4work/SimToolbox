#include "MPI/Srim.hpp"

#include <iostream>
#include <vector>

#include <mpi.h>

struct TestNode {
  float x;
  float y;
  float z;
  bool is_sphere;
  int id;
  double weight;
  int property[100];

  // overload == for comparing two nodes
  bool operator==(const TestNode &node) const {
    for (int i = 0; i < 100; ++i) {
      if (property[i] != node.property[i]) {
        return false;
      }
    }
    return (x == node.x) && (y == node.y) && (z == node.z) &&
           (is_sphere == node.is_sphere) && (id == node.id) &&
           (weight == node.weight);
  }
};

// generate random node
TestNode randNode() {
  TestNode node;
  node.x = ((float)rand()) / ((float)RAND_MAX);
  node.y = ((float)rand()) / ((float)RAND_MAX);
  node.z = ((float)rand()) / ((float)RAND_MAX);
  node.is_sphere = rand() & 1;
  node.weight = ((double)rand()) / ((double)RAND_MAX);
  node.id = rand();
  int array_size = sizeof(node.property) / sizeof(node.property[0]);
  for (int j = 0; j < array_size; ++j) {
    node.property[j] = rand() % 10000000;
  }
  return node;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // use a fixed node count for easier testing
  int node_count = 100000;

  std::vector<TestNode> in_vec;
  std::vector<TestNode> out_vec;
  std::vector<int> offset;
  std::vector<int> source;

  srand(comm_rank + node_count);

  for (int i = 0; i < node_count; ++i) {
    // build node
    in_vec.push_back(randNode());
  }

  // build offset and source
  offset.push_back(0);
  for (int i = 0; i < comm_size; ++i) {
    for (int j = 0; j < node_count; ++j) {
      if (rand() & 1) {
        source.push_back(j);
      }
    }
    offset.push_back(source.size());
  }

  // run reverseScatter
  srim::reverseScatter<std::vector<TestNode>, TestNode>(in_vec, out_vec, offset,
                                                        source);

  // now test the result by generating them again
  int idx = 0;
  for (int rank = 0; rank < comm_size; ++rank) {
    srand(rank + node_count);
    for (int i = 0; i < node_count; ++i) {
      // build node
      TestNode rank_node = randNode();

      // see if this is the node the comm_rank requested
      // if it is, then check if the value equal.
      // we compare them with idx one by one
      if (i == source[idx]) {
        if (!(rank_node == out_vec[idx])) {
          printf("Error\n");
          std::exit(1);
        }
        ++idx;
      }
    }
  }
  if (idx != out_vec.size()) {
    printf("Error\n");
    std::exit(1);
  }
  MPI_Finalize();
  return 0;
}
