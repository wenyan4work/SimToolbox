/**
 * @file BCQPSolver_test.cpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief test of Bound Constrained Quadratic Programming
 * @version 0.1
 * @date 2019-10-21
 *
 * @copyright Copyright (c) 2019
 *
 */

#include "BCQPSolver.hpp"
#include "Util/Logger.hpp"

#include <vector>

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    {
        Logger::setup_mpi_spdlog();
        const int localSize = argc > 1 ? atoi(argv[1]) : 500;
        const double diagonal = argc > 2 ? atof(argv[2]) : 0.0;
        const int maxIte = argc > 3 ? atoi(argv[3]) : localSize;
        // generate a test problem
        BCQPSolver test(localSize, diagonal);

        double tol = 1e-6;

        test.selfTest(tol, maxIte, 0); // BBPGD
        test.selfTest(tol, maxIte, 1); // APGD
    }
    MPI_Finalize();
    return 0;
}