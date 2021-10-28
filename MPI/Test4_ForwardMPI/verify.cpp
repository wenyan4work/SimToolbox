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
  int max_node_count = 100000;

  std::vector<TestNode> in_vec;
  std::vector<TestNode> out_vec;
  std::vector<int> out_id;
  std::vector<int> offset;
  std::vector<int> dest;
  srand(comm_rank + max_node_count);
  int node_count = rand() % max_node_count;
  offset.push_back(0);
  for (int i = 0; i < node_count; ++i) {
    // build node
    in_vec.push_back(randNode());
    // build offset and dest for nodes
    for (int j = 0; j < comm_size; ++j) {
      if (rand() & 1) {
        dest.push_back(j);
      }
    }
    offset.push_back(dest.size());
  }

  // run forwardScatter
  srim::forwardScatter<std::vector<TestNode>, TestNode>(in_vec, out_vec, offset,
                                                        dest);

  // now test the result by generating them again
  int idx = 0;
  for (int rank = 0; rank < comm_size; ++rank) {
    srand(rank + max_node_count);
    int rank_node_count = rand() % max_node_count;
    for (int i = 0; i < rank_node_count; ++i) {
      // build node
      TestNode rank_node = randNode();
      // see if the out_id matches
      bool match_found = false;
      for (int j = 0; j < comm_size; ++j) {
        if (rand() & 1) {
          if (j == comm_rank) {
            match_found = true;
          }
        }
      }

      if (match_found) {
        if (!(rank_node == out_vec[idx])) {
          printf("Error\n");
          exit(1);
        }
        ++idx;
      }
    }
  }
  if (idx != out_vec.size()) {
    printf("Error\n");
    exit(1);
  }
  MPI_Finalize();
  return 0;
}
