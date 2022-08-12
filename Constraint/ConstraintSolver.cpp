#include "ConstraintSolver.hpp"
#include "Util/Logger.hpp"

void ConstraintSolver::setup(ConstraintCollector &conCollector_, Teuchos::RCP<TOP> &mobOpRcp_, double dt_) {
    reset();

    conCollector = conCollector_;

    dt = dt_;
    mobOpRcp = mobOpRcp_;
    mobMapRcp = mobOpRcp->getDomainMap();

    conCollector.buildConstraintMatrixVector(mobMapRcp, DMatTransRcp, deltaRcp, 
                                             invKappaRcp, gammaRcp);

    deltaRcp->scale(1.0 / dt);
    invKappaRcp->scale(1.0 / dt);

    // the BCQP problem
    MOpRcp = Teuchos::rcp(new ConstraintOperator(mobOpRcp, DMatTransRcp, invKappaRcp));

    // result
    forceConRcp = Teuchos::rcp(new TV(mobMapRcp, true));
    velConRcp = Teuchos::rcp(new TV(mobMapRcp, true));
}

void ConstraintSolver::reset() {
    setControlParams(1e-5, 1000000, 0);

    mobMapRcp.reset(); ///< distributed map for obj mobility. 6 dof per obj
    mobOpRcp.reset();  ///< mobility operator, 6 dof per obj to 6 dof per obj
    forceConRcp.reset(); ///< force vec, 6 dof per obj, due to all constraints
    velConRcp.reset();   ///< velocity vec, 6 dof per obj. due to all constraints

    // composite vectors and operators
    DMatTransRcp.reset(); ///< D^Trans matrix
    invKappaRcp.reset();  ///< K^{-1} diagonal matrix
    deltaRcp.reset();    ///< the current (geometric) delta vector delta = deltau
                         ///< this is the constant part of BCQP problem. q = delta (the initial separation)

    // the constraint problem M gamma + q
    MOpRcp.reset();   ///< the operator of BCQP problem. M = D^T M D + 1/h K^{-1}
    gammaRcp.reset(); ///< the unknown constraint lagrange multiplier
}

void ConstraintSolver::solveConstraints() {
    ///////////
    // setup //
    ///////////
    const auto &commRcp = gammaRcp->getMap()->getComm();
    // solver
    BCQPSolver solver(MOpRcp, deltaRcp);
    spdlog::debug("solver constructed");

    // the bound of BCQP. 0 for all constraints
    Teuchos::RCP<TV> lbRcp = solver.getLowerBound();
    lbRcp->scale(0, *gammaRcp); // 0 for all flags and the same size as gamma
    spdlog::debug("bound constructed");

    ///////////
    // solve //
    ///////////
    IteHistory history;
    switch (solverChoice) {
    case 0:
        solver.solveBBPGD(gammaRcp, res, maxIte, history);
        break;
    case 1:
        solver.solveAPGD(gammaRcp, res, maxIte, history);
        break;
    default:
        solver.solveBBPGD(gammaRcp, res, maxIte, history);
        break;
    }

    ////////////
    // output //
    ////////////
    for (auto it = history.begin(); it != history.end() - 1; it++) {
        auto &p = *it;
        spdlog::debug("RECORD: BCQP history {:g}, {:g}, {:g}, {:g}, {:g}, {:g}", p[0], p[1], p[2], p[3], p[4], p[5]);
    }

    auto &p = history.back();
    spdlog::info("RECORD: BCQP residue {:g}, {:g}, {:g}, {:g}, {:g}, {:g}", p[0], p[1], p[2], p[3], p[4], p[5]);

    /////////////////////
    // process results //
    /////////////////////
    // calculate the reactinary, constraint vel/force with the solution
    Teuchos::RCP<TCMAT> DMatRcp = MOpRcp->getDMat();
    DMatRcp->apply(*gammaRcp, *forceConRcp);
    mobOpRcp->apply(*forceConRcp, *velConRcp);

    // calculate the final delta value 
    Teuchos::RCP<TCMAT> DMatTransRcp = MOpRcp->getDMatTrans();
    DMatTransRcp->apply(*velConRcp, *deltaRcp, Teuchos::NO_TRANS, dt, 0.0);
}

void ConstraintSolver::writebackGamma() { 
    conCollector.writeBackGamma(gammaRcp); 
}

void ConstraintSolver::writebackDelta() { 
    conCollector.writeBackDelta(deltaRcp); 
}