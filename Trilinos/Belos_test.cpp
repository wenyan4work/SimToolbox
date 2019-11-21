#include "Preconditioner.hpp"
#include "TpetraUtil.hpp"

#include <random>

Teuchos::RCP<TCMAT> genMatrix(const int localSize, const double diagonal) {
    Teuchos::RCP<TCMAT> matrixRcp;

    // set up comm
    auto commRcp = getMPIWORLDTCOMM();
    // set up row and col maps, contiguous and evenly distributed
    Teuchos::RCP<const TMAP> rowMapRcp = getTMAPFromLocalSize(localSize, commRcp);
    auto mapRcp = rowMapRcp;

    if (commRcp->getRank() == 0) {
        std::cout << "Total number of processes: " << commRcp->getSize() << std::endl;
        std::cout << "rank: " << commRcp->getRank() << std::endl;
        std::cout << "global size: " << mapRcp->getGlobalNumElements() << std::endl;
        std::cout << "local size: " << mapRcp->getNodeNumElements() << std::endl;
        std::cout << "map: " << mapRcp->description() << std::endl;
    }

    // make sure A and b match the map and comm specified
    // set A and b randomly. maintain SPD of A
    // generate a local random matrix

    std::mt19937 gen(1234 + commRcp->getRank()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    // a random matrix
    Teuchos::SerialDenseMatrix<int, double> BLocal(localSize, localSize, true); // zeroOut
    for (int i = 0; i < localSize; i++) {
        for (int j = 0; j < localSize; j++) {
            // BLocal(i, j) = dis(gen) * (j == i ? 1 : pow(fabs(j - i), -1));
            BLocal(i, j) = dis(gen);
        }
    }
    // a random diagonal matrix
    Teuchos::SerialDenseMatrix<int, double> ALocal(localSize, localSize, true);
    Teuchos::SerialDenseMatrix<int, double> tempLocal(localSize, localSize, true);
    Teuchos::SerialDenseMatrix<int, double> DLocal(localSize, localSize, true);
    for (int i = 0; i < localSize; i++) {
        DLocal(i, i) = fabs(dis(gen)) + 2;
    }

    // compute B^T D B
    tempLocal.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, DLocal, BLocal, 0.0); // temp = DB
    ALocal.multiply(Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, BLocal, tempLocal, 0.0);    // A = B^T DB

    for (int i = 0; i < localSize; i++) {
        ALocal(i, i) += diagonal;
    }

    // use ALocal as local matrix to fill TCMAT A
    // block diagonal distribution of A
    double droptol = 1e-4;
    Kokkos::View<size_t *> rowCount("rowCount", localSize);
    Kokkos::View<size_t *> rowPointers("rowPointers", localSize + 1);
    for (int i = 0; i < localSize; i++) {
        rowCount[i] = 0;
        for (int j = 0; j < localSize; j++) {
            if (fabs(ALocal(i, j)) > droptol) {
                rowCount[i]++;
            }
        }
    }

    rowPointers[0] = 0;
    for (int i = 1; i < localSize + 1; i++) {
        rowPointers[i] = rowPointers[i - 1] + rowCount[i - 1];
    }
    Kokkos::View<int *> columnIndices("columnIndices", rowPointers[localSize]);
    Kokkos::View<double *> values("values", rowPointers[localSize]);
    int p = 0;
    for (int i = 0; i < localSize; i++) {
        for (int j = 0; j < localSize; j++) {
            if (fabs(ALocal(i, j)) > droptol) {
                columnIndices[p] = j;
                values[p] = ALocal(i, j);
                p++;
            }
        }
    }

    const int myRank = commRcp->getRank();
    const int colIndexCount = rowPointers[localSize];
    std::vector<int> colMapIndex(colIndexCount);
#pragma omp parallel for
    for (int i = 0; i < colIndexCount; i++) {
        colMapIndex[i] = columnIndices[i] + myRank * localSize;
    }

    // sort and unique
    std::sort(colMapIndex.begin(), colMapIndex.end());
    colMapIndex.erase(std::unique(colMapIndex.begin(), colMapIndex.end()), colMapIndex.end());

    Teuchos::RCP<TMAP> colMapRcp = Teuchos::rcp(
        new TMAP(Teuchos::OrdinalTraits<int>::invalid(), colMapIndex.data(), colMapIndex.size(), 0, commRcp));

    // fill matrix Aroot
    matrixRcp = Teuchos::rcp(new TCMAT(rowMapRcp, colMapRcp, rowPointers, columnIndices, values));
    matrixRcp->fillComplete(rowMapRcp, rowMapRcp);
    // // ARcp = Atemp;
    // std::cout << "ARcp" << matrixRcp->description() << std::endl;
    // // dump matrix
    // dumpTCMAT(matrixRcp, "Amat");

    return matrixRcp;
}

Teuchos::RCP<TV> genVector(const int dimension, const int seed) {
    std::vector<double> vecLocal(dimension);
    std::mt19937 gen(seed);
    std::normal_distribution<double> dis(1, 1);

    for (auto &x : vecLocal) {
        x = dis(gen);
    }

    auto commRcp = getMPIWORLDTCOMM();
    Teuchos::RCP<TV> vectorRcp = getTVFromVector(vecLocal, commRcp);

    return vectorRcp;
}

void genLinearProblem(const int dimension, const double diagonal, Teuchos::RCP<TCMAT> &ARcp, Teuchos::RCP<TV> &xRcp,
                      Teuchos::RCP<TV> &xTrueRcp, Teuchos::RCP<TV> &bRcp) {
    // Ax=b
    ARcp = genMatrix(dimension, diagonal);
    xTrueRcp = genVector(dimension, 5678 + ARcp->getComm()->getRank());
    bRcp = Teuchos::rcp<TV>(new TV(xTrueRcp->getMap(), true));
    ARcp->apply(*xTrueRcp, *bRcp);

    // xguess, random
    xRcp = genVector(dimension, 1 + ARcp->getComm()->getRank());
    ARcp->getComm()->barrier();
    dumpTCMAT(ARcp, "A");
    dumpTV(bRcp, "b");
    dumpTV(xTrueRcp, "xTrue");
    dumpTV(xRcp, "xGuess");
}

void testBelosSolver(Teuchos::RCP<TCMAT> &ARcp, Teuchos::RCP<TV> &xRcp, Teuchos::RCP<TV> &xTrueRcp,
                     Teuchos::RCP<TV> &bRcp, std::string solver) {
    auto problemRcp = Teuchos::rcp(new Belos::LinearProblem<TOP::scalar_type, TMV, TOP>(ARcp, xRcp, bRcp));

    Teuchos::RCP<Teuchos::ParameterList> solverParams = Teuchos::parameterList();
    solverParams->set("Timer Label", solver);
    solverParams->set("Maximum Iterations", 1000);
    solverParams->set("Convergence Tolerance", 1e-6);
    // solverParams->set("Maximum Restarts", 100);
    // solverParams->set("Num Blocks", 100); // larger values might trigger a std::bad_alloc inside Kokkos.
    solverParams->set("Orthogonalization", "IMGS");
    // solverParams->set("Output Style", Belos::OutputType::General);
    solverParams->set("Implicit Residual Scaling", "Norm of RHS");
    solverParams->set("Explicit Residual Scaling", "Norm of RHS");
    // default is preconditioned initial residual
    // solverParams->set("Implicit Residual Scaling", "Norm of Initial Residual");
    // solverParams->set("Explicit Residual Scaling", "Norm of Initial Residual");
    // solverParams->set("Implicit Residual Scaling", "None");
    // solverParams->set("Explicit Residual Scaling", "None");

    // all info except debug info
    solverParams->set("Verbosity", Belos::Errors + Belos::Warnings + +Belos::IterationDetails + Belos::OrthoDetails +
                                       Belos::FinalSummary + Belos::TimingDetails + Belos::StatusTestDetails);
    // solverParams->set("Output Frequency", 1);
    Belos::SolverFactory<TOP::scalar_type, TMV, TOP> factory;
    auto solverRcp = factory.create(solver, solverParams); // recycle Krylov space for collision

    bool set = problemRcp->setProblem();
    TEUCHOS_TEST_FOR_EXCEPTION(!set, std::runtime_error, "*** Belos::LinearProblem failed to set up correctly! ***");
    solverRcp->setProblem(problemRcp);

    Belos::ReturnType result = solverRcp->solve();
    int numIters = solverRcp->getNumIters();
    dumpTV(xRcp, std::string("xsol_") + solver);

    Teuchos::TimeMonitor::zeroOutTimers();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    {
        constexpr int dimension = 500;   // dimension per rank
        constexpr double diagonal = 0.1; // added to matrix diagonel, tune the condition number
        Teuchos::RCP<TCMAT> ARcp;
        Teuchos::RCP<TV> xRcp;
        Teuchos::RCP<TV> xTrueRcp;
        Teuchos::RCP<TV> bRcp;
        Teuchos::RCP<TV> xGuessRcp;

        genLinearProblem(dimension, diagonal, ARcp, xRcp, xTrueRcp, bRcp);
        xGuessRcp = Teuchos::rcp<TV>(new TV(*xRcp, Teuchos::Copy));
        testBelosSolver(ARcp, xGuessRcp, xTrueRcp, bRcp, "BICGSTAB");
        xGuessRcp = Teuchos::rcp<TV>(new TV(*xRcp, Teuchos::Copy));
        testBelosSolver(ARcp, xGuessRcp, xTrueRcp, bRcp, "GMRES");
    }
    MPI_Finalize();
    return 0;
}