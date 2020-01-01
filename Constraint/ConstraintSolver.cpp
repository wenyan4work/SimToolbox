#include "ConstraintSolver.hpp"

void ConstraintSolver::setup(ConstraintCollector &uniConstraints_, ConstraintCollector &biConstraints_,
                             Teuchos::RCP<TOP> &mobOpRcp_, Teuchos::RCP<TV> &velncRcp_, double dt_) {
    reset();

    dt = dt_;
    uniConstraints = uniConstraints_;
    biConstraints = biConstraints_;
    mobOpRcp = mobOpRcp_;
    velncRcp = velncRcp_;

    mobMapRcp = mobOpRcp->getDomainMap();

    // unilateral block ops and vecs
    uniConstraints.buildConstraintMatrixVector(mobMapRcp, DuMatTransRcp, delta0uRcp, gammauRcp);
    delta0uRcp->scale(1.0 / dt);
    deltancuRcp = Teuchos::rcp(new TV(delta0uRcp->getMap(), true));
    DuMatTransRcp->apply(*velncRcp, *deltancuRcp);

    // bilateral block ops and vecs
    biConstraints.buildConstraintMatrixVector(mobMapRcp, DbMatTransRcp, delta0bRcp, gammabRcp);
    delta0bRcp->scale(1.0 / dt);
    biConstraints.buildInvKappa(invKappa);
    for (auto &v : invKappa) {
        v *= 1.0 / dt;
    }
    deltancbRcp = Teuchos::rcp(new TV(delta0bRcp->getMap(), true));
    DbMatTransRcp->apply(*velncRcp, *deltancbRcp);

    // assembled non-block vectors
    delta0Rcp = getTVFromTwoBlockTV(delta0uRcp, delta0bRcp);
    deltancRcp = getTVFromTwoBlockTV(deltancuRcp, deltancbRcp);

    // the BCQP problem
    gammaRcp = getTVFromTwoBlockTV(gammauRcp, gammabRcp); // initial guess
    MOpRcp = Teuchos::rcp(new ConstraintOperator(mobOpRcp, DuMatTransRcp, DbMatTransRcp, invKappa));
    qRcp = Teuchos::rcp(new TV(*delta0Rcp, Teuchos::DataAccess::Copy));

    // mobility vectors, allocated inside MOpRcp
    forceuRcp = MOpRcp->getForceUni();
    forcebRcp = MOpRcp->getForceBi();
    veluRcp = MOpRcp->getVelUni();
    velbRcp = MOpRcp->getVelBi();
}

void ConstraintSolver::reset() {
    setControlParams(1e-5, 1000000, 0);

    // mobility-map
    mobMapRcp.reset(); ///< distributed map for obj mobility. 6 dof per obj
    mobOpRcp.reset();  ///< mobility operator, 6 dof per obj to 6 dof per obj
    forceuRcp.reset(); ///< force vec, 6 dof per obj, due to unilateral constraints
    forcebRcp.reset(); ///< force vec, 6 dof per obj, due to bilateral constraints
    veluRcp.reset();   ///< velocity vec, 6 dof per obj. due to unilateral constraints
    velbRcp.reset();   ///< velocity vec, 6 dof per obj. due to bilateral constraints
    velncRcp.reset();  ///< the non-constraint velocity vel_nc

    // unilateral constraints block ops and vecs
    DuMatTransRcp.reset(); ///< unilateral constraint matrix
    gammauRcp.reset();     ///< the unknown unilateral constraint
    delta0uRcp.reset();    ///< unilateral delta0 vector, built with Du^Trans
    deltancuRcp.reset();   ///< delta_nc,u = Du^Trans vel_nc,u

    // bilateral constraints block ops and vecs
    DbMatTransRcp.reset(); ///< bilateral constraint matrix
    invKappa.clear();      ///< inverse of spring constant kappa
    gammabRcp.reset();     ///< the unknown bilateral constraint
    delta0bRcp.reset();    ///< bilateral delta0 vector, built with Dc^Trans
    deltancbRcp.reset();   ///< delta_nc,b = Db^Trans vel_nc,b

    // composite vectors and operators
    delta0Rcp.reset();  ///< the current (geometric) delta vector delta_0 = [delta_0u ; delta_0b]
    deltancRcp.reset(); ///< delta_nc = [Du^Trans vel_nc,u ; Db^Trans vel_nc,b]

    // the constraint problem
    gammaRcp.reset(); ///< the unknown constraint force magnitude gamma = [gamma_u;gamma_b]
    MOpRcp.reset();   ///< the operator of BCQP problem. M = [B,C;E,F]
    qRcp.reset();     ///< the constant part of BCQP problem. q = delta_0 + delta_n
}

void ConstraintSolver::solveConstraints() {
    const auto &commRcp = gammauRcp->getMap()->getComm();
    // solver
    BCQPSolver solver(MOpRcp, qRcp);

    // the bound of BCQP. 0 for gammau, unbound for gammab.
    Teuchos::RCP<TV> lbRcp = solver.getLowerBound();
    auto lbuRcp = lbRcp->offsetViewNonConst(gammauRcp->getMap(), 0);
    lbuRcp->putScalar(0);

    // solve
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

    if (commRcp->getRank() == 0 && history.size() > 0) {
        auto &p = history.back();
        std::cout << "RECORD: BCQP residue";
        for (auto &v : p) {
            std::cout << "," << v;
        }
        std::cout << std::endl;
    }

    // block view
    auto mapu = MOpRcp->getUniBlockMap();
    auto mapb = MOpRcp->getBiBlockMap();
    gammauRcp = gammaRcp->offsetViewNonConst(mapu, 0);
    gammabRcp = gammaRcp->offsetViewNonConst(mapb, mapu->getNodeNumElements());
}

void ConstraintSolver::writebackGamma() {
    uniConstraints.writeBackGamma(gammauRcp.getConst());
    biConstraints.writeBackGamma(gammabRcp.getConst());
}