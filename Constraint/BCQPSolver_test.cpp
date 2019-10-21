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

#include <vector>

#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    {
        const int globalSize = argc > 1 ? atoi(argv[1]) : 500;
        const double diagonal = argc > 2 ? atof(argv[2]) : 0.0;
        // generate a test problem
        BCQPSolver test(globalSize, diagonal);

        double tol = 1e-5;
        int maxIte = 1000;

        test.selfTest(tol, maxIte, 0); // BBPGD
        test.selfTest(tol, maxIte, 1); // APGD
    }
    MPI_Finalize();
    return 0;
}