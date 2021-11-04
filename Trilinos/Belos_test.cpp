#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

void genLinearProblem(Teuchos::RCP<TCMAT> &ARcp, Teuchos::RCP<TV> &xGuessRcp,
                      Teuchos::RCP<TV> &xTrueRcp, Teuchos::RCP<TV> &bRcp) {
  // Ax=b
  auto commRcp = Tpetra::getDefaultComm();
  // A read from MatrixMarket mtx file
  Tpetra::MatrixMarket::Reader<TCMAT> mmReader;
  ARcp = mmReader.readSparseFile("A_TCMAT.mtx", commRcp);

  describe(*ARcp);

  // b, x
  auto xmap = ARcp->getDomainMap();
  auto bmap = ARcp->getRangeMap();
  xTrueRcp = Teuchos::rcp(new TV(xmap, true));
  xTrueRcp->randomize(-1, 1);
  bRcp = Teuchos::rcp<TV>(new TV(bmap, true));
  ARcp->apply(*xTrueRcp, *bRcp);

  // xguess, random
  xGuessRcp = Teuchos::rcp(new TV(xmap, true));
  xGuessRcp->randomize(-1, 1);
  commRcp->barrier();

  dumpTV(bRcp, "b");
  dumpTV(xTrueRcp, "xTrue");
  dumpTV(xGuessRcp, "xGuess");
}

void testBelosSolver(Teuchos::RCP<TCMAT> &ARcp, Teuchos::RCP<TV> &xRcp,
                     Teuchos::RCP<TV> &xTrueRcp, Teuchos::RCP<TV> &bRcp,
                     std::string solver) {
  auto problemRcp = Teuchos::rcp(
      new Belos::LinearProblem<TOP::scalar_type, TMV, TOP>(ARcp, xRcp, bRcp));

  Teuchos::RCP<Teuchos::ParameterList> solverParams = Teuchos::parameterList();
  solverParams->set("Timer Label", solver);
  solverParams->set("Maximum Iterations", 2000);
  solverParams->set("Convergence Tolerance", 1e-6);
  // solverParams->set("Maximum Restarts", 100);
  // larger values might trigger a std::bad_alloc inside Kokkos.
  solverParams->set("Num Blocks", 100);
  solverParams->set("Orthogonalization", "IMGS");
  // default is preconditioned initial residual
  // solverParams->set("Implicit Residual Scaling", "Norm of RHS");
  // solverParams->set("Explicit Residual Scaling", "Norm of RHS");
  // solverParams->set("Implicit Residual Scaling", "Norm of Initial Residual");
  // solverParams->set("Explicit Residual Scaling", "Norm of Initial Residual");
  solverParams->set("Implicit Residual Scaling", "None");
  solverParams->set("Explicit Residual Scaling", "None");

  // all info except debug info
  solverParams->set("Verbosity",
                    Belos::Errors + Belos::Warnings + +Belos::IterationDetails +
                        Belos::OrthoDetails + Belos::FinalSummary +
                        Belos::TimingDetails + Belos::StatusTestDetails);
  Belos::SolverFactory<TOP::scalar_type, TMV, TOP> factory;
  auto solverRcp = factory.create(solver, solverParams);

  bool set = problemRcp->setProblem();
  TEUCHOS_TEST_FOR_EXCEPTION(
      !set, std::runtime_error,
      "*** Belos::LinearProblem failed to set up correctly! ***");

  Teuchos::RCP<TOP> prec = createILUTPreconditioner(ARcp);
  problemRcp->setRightPrec(prec);

  solverRcp->setProblem(problemRcp);

  Belos::ReturnType result = solverRcp->solve();
  int numIters = solverRcp->getNumIters();
  dumpTV(xRcp, std::string("xsol_") + solver);

  xRcp->update(1.0, *xTrueRcp, -1.0);
  double norm = xRcp->norm2();
  spdlog::info("|x-x_true|_2 {:g}", norm);
  if (norm > 1e-6)
    printf("Error, not converged\n");

  Teuchos::TimeMonitor::zeroOutTimers();
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  {
    Teuchos::RCP<TCMAT> ARcp;
    Teuchos::RCP<TV> xRcp;
    Teuchos::RCP<TV> xTrueRcp;
    Teuchos::RCP<TV> bRcp;
    Teuchos::RCP<TV> xGuessRcp;

    genLinearProblem(ARcp, xGuessRcp, xTrueRcp, bRcp);

    xRcp = Teuchos::rcp<TV>(new TV(*xGuessRcp, Teuchos::Copy));
    testBelosSolver(ARcp, xRcp, xTrueRcp, bRcp, "BICGSTAB");

    xRcp = Teuchos::rcp<TV>(new TV(*xGuessRcp, Teuchos::Copy));
    testBelosSolver(ARcp, xRcp, xTrueRcp, bRcp, "GMRES");
  }
  MPI_Finalize();
  return 0;
}