#include "ConstraintSolver.hpp"

void ConstraintSolver::setup(ConstraintCollector &uniConstraints_, ConstraintCollector &biConstraints_,
                             Teuchos::RCP<TOP> &mobOpRcp_, Teuchos::RCP<TV> &velncRcp_, double dt_) {
    dt = dt_;
    uniConstraints = uniConstraints_;
    biConstraints = biConstraints_;
    mobOpRcp = mobOpRcp_;
    mobMapRcp = mobOpRcp->getDomainMap();
    velncRcp = velncRcp_;

    reset();

    // necessary operators and matrices
    uniConstraints.buildConstraintMatrixVector(mobMapRcp, DuMatTransRcp, delta0uRcp, gammauRcp);
    biConstraints.buildConstraintMatrixVector(mobMapRcp, DbMatTransRcp, delta0bRcp, gammabRcp);
    biConstraints.buildInvKappa(invKappa);
    conOpRcp = Teuchos::rcp(new ConstraintOperator(mobOpRcp, DuMatTransRcp, DbMatTransRcp, invKappa));

    // // assembled non-block vectors
    // gammaRcp = compositeVectors(gammauRcp, gammabRcp);
}

void ConstraintSolver::reset() {
    setControlParams(1e-5, 1000000);

    mobMapRcp.reset(); ///< distributed map for obj mobility. 6 dof per obj
    forceuRcp.reset(); ///< force vec, 6 dof per obj
    forcebRcp.reset(); ///< force vec, 6 dof per obj
    veluRcp.reset();   ///< velocity vec, 6 dof per obj
    velbRcp.reset();   ///< velocity vec, 6 dof per obj

    // unknown constraint force magnitude
    gammaRcp.reset(); ///< the unknown constraint force magnitude gamma
    gammauRcp.reset();
    gammabRcp.reset();

    // non-block vectors and operators
    conOpRcp.reset();   ///< the operator
    velncRcp.reset();   ///< the non-constraint velocity vel_nc
    delta0Rcp.reset();  ///< the current (geometric) delta vector delta_0 = [delta_0u ; delta_0b]
    deltancRcp.reset(); ///< delta_nc = [Du^Trans vel_nc,u ; Db^Trans vel_nc,b]
    bRcp.reset();       ///< the constant part of BCQP problem. b = delta_0 + delta_nc

    // block vectors and operators
    mobOpRcp.reset();      ///< mobility operator, 6 dof per obj to 6 dof per obj
    DuMatTransRcp.reset(); ///< unilateral constraint matrix
    DbMatTransRcp.reset(); ///< bilateral constraint matrix
    delta0uRcp.reset();    ///< unilateral delta0 vector, built with Du^Trans
    delta0bRcp.reset();    ///< bilateral delta0 vector, built with Dc^Trans
    deltancuRcp.reset();   ///< delta_nc,u = Du^Trans vel_nc,u
    deltancbRcp.reset();   ///< delta_nc,b = Db^Trans vel_nc,b
    invKappa.clear();
}
    void ConstraintSolver::solveConstraints(){
        
    }

    void ConstraintSolver::writebackGamma(){

    }