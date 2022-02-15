/**
 * @file SphereSystem_main.cpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-02-15
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "Sphere.hpp"
#include "mainloop_example.hpp"

#include "Util/Logger.hpp"

#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  std::string configFile = "RunConfig.toml";
  std::string posFile = "SphereInitial.dat";
  mainloop<Sphere>(configFile, posFile);
  MPI_Finalize();
  return 0;
}