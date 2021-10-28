#include "ConstraintSolver.hpp"
#include "BCQPSolver.hpp"

#include "Util/Logger.hpp"

void ConstraintSolver::setup(ConstraintCollector &conCollector_,
                             Teuchos::RCP<TOP> &mobOpRcp_,
                             Teuchos::RCP<TV> &velncRcp_, double dt_) {
  reset();

  conCollector = conCollector_;

  dt = dt_;
  mobOpRcp = mobOpRcp_;
  velncRcp = velncRcp_;

  mobMapRcp = mobOpRcp->getDomainMap();

  conCollector.buildConstraintMatrixVector(mobMapRcp, DMatTransRcp, delta0Rcp,
                                           invKappaRcp, biFlagRcp, gammaRcp);

  delta0Rcp->scale(1.0 / dt);
  invKappaRcp->scale(1.0 / dt);

  deltancRcp = Teuchos::rcp(new TV(delta0Rcp->getMap(), true));
  DMatTransRcp->apply(*velncRcp, *deltancRcp);

  // the BCQP problem
  MOpRcp = Teuchos::rcp(new ConstraintOperator(mobOpRcp,     //
                                               DMatTransRcp, //
                                               invKappaRcp));
  qRcp = Teuchos::rcp(new TV(delta0Rcp->getMap(), true));
  qRcp->update(1.0, *delta0Rcp, 1.0, *deltancRcp, 0.0);

  // result
  forcebRcp = Teuchos::rcp(new TV(mobMapRcp, true));
  forceuRcp = Teuchos::rcp(new TV(mobMapRcp, true));
  velbRcp = Teuchos::rcp(new TV(mobMapRcp, true));
  veluRcp = Teuchos::rcp(new TV(mobMapRcp, true));
}

void ConstraintSolver::reset() {
  setControlParams(1e-5, 1000000, 0);

  mobMapRcp.reset();
  mobOpRcp.reset();
  forceuRcp.reset();
  forcebRcp.reset();
  veluRcp.reset();
  velbRcp.reset();
  velncRcp.reset();

  // composite vectors and operators
  DMatTransRcp.reset();
  invKappaRcp.reset();
  biFlagRcp.reset();
  delta0Rcp.reset();
  deltancRcp.reset();

  // the constraint problem M gamma + q
  MOpRcp.reset();
  gammaRcp.reset();

  qRcp.reset();
}

void ConstraintSolver::solveConstraints() {
  const auto &commRcp = gammaRcp->getMap()->getComm();
  // solver
  BCQPSolver solver(MOpRcp, qRcp);
  spdlog::debug("solver constructed");

  // the bound of BCQP. 0 for gammau, unbound for gammab.
  Teuchos::RCP<TV> lbRcp = solver.getLowerBound();
  lbRcp->scale(-1e8, *biFlagRcp); // 0 if biFlag=0, -1e8 if biFlag=1
  spdlog::debug("bound constructed");

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

  for (auto it = history.begin(); it != history.end() - 1; it++) {
    auto &p = *it;
    spdlog::debug("RECORD: BCQP history {:g}, {:g}, {:g}, {:g}, {:g}, {:g}",
                  p[0], p[1], p[2], p[3], p[4], p[5]);
  }

  auto &p = history.back();
  spdlog::info("RECORD: BCQP residue {:g}, {:g}, {:g}, {:g}, {:g}, {:g}", p[0],
               p[1], p[2], p[3], p[4], p[5]);

  // calculate unilateral and bilateral vel/force with solution
  Teuchos::RCP<TCMAT> DMatRcp = MOpRcp->getDMat();
  // bilateral first
  Teuchos::RCP<TV> gammaBiRcp = Teuchos::rcp(new TV(gammaRcp->getMap(), true));
  gammaBiRcp->elementWiseMultiply(1.0, *gammaRcp, *biFlagRcp, 0.0);
  DMatRcp->apply(*gammaBiRcp, *forcebRcp);
  mobOpRcp->apply(*forcebRcp, *velbRcp);
  // unilateral second
  Teuchos::RCP<TV> forceRcp = MOpRcp->getForce();
  Teuchos::RCP<TV> velRcp = MOpRcp->getVel();
  // force_u = force - force_b
  forceuRcp->update(1.0, *forceRcp, -1.0, *forcebRcp, 0.0);
  // vel_u = vel - vel_b
  veluRcp->update(1.0, *velRcp, -1.0, *velbRcp, 0.0);
}

void ConstraintSolver::writebackGamma() {
  conCollector.writeBackGamma(gammaRcp.getConst());
}