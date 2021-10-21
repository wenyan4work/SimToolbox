#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <omp.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  Logger::setup_mpi_spdlog();
  {
    const int size = 10000000;
    std::mt19937 gen(size);
    std::uniform_int_distribution<> dis(-10, 10);

    Kokkos::View<int *> array("array", size);
    std::vector<int> array_v(size);

    // init
    for (int i = 0; i < size; i++) {
      array(i) = dis(gen);
      array_v[i] = array[i];
    }

    Teuchos::RCP<Teuchos::Time> stdTimer =
        Teuchos::TimeMonitor::getNewCounter("single thread std::partial_sum");
    Teuchos::RCP<Teuchos::Time> kokkosTimer =
        Teuchos::TimeMonitor::getNewCounter(
            "multi thread kokkos::parallel_scan");

    { // in-place inclusive prefix sum
      Teuchos::TimeMonitor mon(*stdTimer);
      std::partial_sum(array_v.begin(), array_v.end(), array_v.begin());
    }

    { // multi-thread inclusive prefix sum
      Teuchos::TimeMonitor mon(*kokkosTimer);
      Kokkos::parallel_scan(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, size),
          KOKKOS_LAMBDA(const int i, int &update_value, const bool final) {
            const auto val_i = array(i);
            update_value += val_i; // if exclusive, move this line to the end
            if (final)
              array(i) = update_value;
          });
    }

    for (int i = 0; i < size; i++) {
      if (array_v[i] != array(i))
        spdlog::error("std {} kokkos {}", array_v[i], array(i));
    }
    Teuchos::TimeMonitor::summarize();
  }
  MPI_Finalize();

  return 0;
}