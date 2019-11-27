#include "Preconditioner.hpp"
#include "TpetraUtil.hpp"

void genLinearProblem(Teuchos::RCP<TCMAT> &ARcp, Teuchos::RCP<TV> &xRcp, Teuchos::RCP<TV> &xTrueRcp,
                      Teuchos::RCP<TV> &bRcp) {
    // Ax=b
    auto commRcp = getMPIWORLDTCOMM();
    // A read from MatrixMarket mtx file
    Tpetra::MatrixMarket::Reader<TCMAT> mmReader;
    ARcp = mmReader.readSparseFile("A_TCMAT.mtx", commRcp);
    // dumpTMAP(ARcp->getDomainMap(), "A_DomainMap");
    // dumpTMAP(ARcp->getRangeMap(), "A_RangeMap");
    // dumpTMAP(ARcp->getRowMap(), "A_RowMap");
    // dumpTMAP(ARcp->getColMap(), "A_ColMap");

    // b, x
    auto xmap = ARcp->getDomainMap();
    auto bmap = ARcp->getRangeMap();
    xTrueRcp = Teuchos::rcp(new TV(xmap, true));
    xTrueRcp->randomize(-1, 1);
    bRcp = Teuchos::rcp<TV>(new TV(bmap, true));
    ARcp->apply(*xTrueRcp, *bRcp);

    // xguess, random
    xRcp = Teuchos::rcp(new TV(xmap, true));
    xRcp->randomize(-1, 1);
    commRcp->barrier();
    dumpTV(bRcp, "b");
    dumpTV(xTrueRcp, "xTrue");
    dumpTV(xRcp, "xGuess");
}

void testBelosSolver(Teuchos::RCP<TCMAT> &ARcp, Teuchos::RCP<TV> &xRcp, Teuchos::RCP<TV> &xTrueRcp,
                     Teuchos::RCP<TV> &bRcp, std::string solver) {
    auto problemRcp = Teuchos::rcp(new Belos::LinearProblem<TOP::scalar_type, TMV, TOP>(ARcp, xRcp, bRcp));

    Teuchos::RCP<Teuchos::ParameterList> solverParams = Teuchos::parameterList();
    solverParams->set("Timer Label", solver);
    solverParams->set("Maximum Iterations", 2000);
    solverParams->set("Convergence Tolerance", 1e-6);
    // solverParams->set("Maximum Restarts", 100);
    solverParams->set("Num Blocks", 100); // larger values might trigger a std::bad_alloc inside Kokkos.
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
    Belos::SolverFactory<TOP::scalar_type, TMV, TOP> factory;
    auto solverRcp = factory.create(solver, solverParams);

    auto prec = PrecondUtil::createILUTPreconditioner(ARcp, 1e-4, 5);
    problemRcp->setRightPrec(prec);

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
        Teuchos::RCP<TCMAT> ARcp;
        Teuchos::RCP<TV> xRcp;
        Teuchos::RCP<TV> xTrueRcp;
        Teuchos::RCP<TV> bRcp;
        Teuchos::RCP<TV> xGuessRcp;

        genLinearProblem(ARcp, xRcp, xTrueRcp, bRcp);
        xGuessRcp = Teuchos::rcp<TV>(new TV(*xRcp, Teuchos::Copy));
        testBelosSolver(ARcp, xGuessRcp, xTrueRcp, bRcp, "BICGSTAB");
        xGuessRcp = Teuchos::rcp<TV>(new TV(*xRcp, Teuchos::Copy));
        testBelosSolver(ARcp, xGuessRcp, xTrueRcp, bRcp, "GMRES");
    }
    MPI_Finalize();
    return 0;
}