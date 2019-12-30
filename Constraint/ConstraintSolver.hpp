/**
 * @file ConstraintSolver.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Solve the uni/bi-lateral constraint problem
 * @version 0.1
 * @date 2019-11-14
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef CONSTRAINTSOLVER_HPP_
#define CONSTRAINTSOLVER_HPP_

#include "BCQPSolver.hpp"
#include "ConstraintCollector.hpp"
#include "ConstraintOperator.hpp"

#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <mpi.h>
#include <omp.h>

/**
 * @brief solver for the constrained dynamics for each timestep
 *
 */
class ConstraintSolver {

  public:
    // constructor
    ConstraintSolver() = default;
    ~ConstraintSolver() = default;

    // forbid copy
    ConstraintSolver(const ConstraintSolver &) = delete;
    ConstraintSolver(ConstraintSolver &&) = delete;
    const ConstraintSolver &operator=(const ConstraintSolver &) = delete;
    const ConstraintSolver &operator=(ConstraintSolver &&) = delete;

    /**
     * @brief reset the parameters and release all allocated spaces
     *
     */
    void reset();

    /**
     * @brief set the control parameters res, maxIte, and newton refinement
     *
     * @param res_ iteration residual
     * @param maxIte_ max iterations
     */
    void setControlParams(double res_, int maxIte_) {
        res = res_;
        maxIte = maxIte_;
    }

    /**
     * @brief setup this solver for solution
     *
     * @param uniConstraint_
     * @param biConstraint_
     * @param objMobMapRcp_
     * @param dt_
     */
    void setup(ConstraintCollector &uniConstraints_, ConstraintCollector &biConstraints_, //
               Teuchos::RCP<TOP> &mobOpRcp_, Teuchos::RCP<TV> &velncRcp_, double dt_);

    /**
     * @brief solve the constraint BCQP problem
     *
     */
    void solveConstraints();

    /**
     * @brief write the solution constraint force magnitude back to uniConstraints and biConstraints
     *
     */
    void writebackGamma();

    Teuchos::RCP<TV> getForceUni() const { return forceuRcp; }
    Teuchos::RCP<TV> getVelocityUni() const { return veluRcp; }
    Teuchos::RCP<TV> getForceBi() const { return forcebRcp; }
    Teuchos::RCP<TV> getVelocityBi() const { return velbRcp; }

  private:
    double dt;  ///< timestep size
    double res; ///< residual tolerance
    int maxIte; ///< max iterations

    ConstraintCollector uniConstraints; ///< unilateral constraints, i.e., collisions
    ConstraintCollector biConstraints;  ///< bilateral constraints, i.e., springs

    // mobility
    Teuchos::RCP<const TMAP> mobMapRcp; ///< distributed map for obj mobility. 6 dof per obj
    Teuchos::RCP<TV> forceuRcp;         ///< force vec, 6 dof per obj, due to unilateral constraints
    Teuchos::RCP<TV> forcebRcp;         ///< force vec, 6 dof per obj, due to bilateral constraints
    Teuchos::RCP<TV> veluRcp;           ///< velocity vec, 6 dof per obj. due to unilateral constraints
    Teuchos::RCP<TV> velbRcp;           ///< velocity vec, 6 dof per obj. due to bilateral constraints

    // unknown constraint force magnitude
    Teuchos::RCP<const TMAP> gammaMapRcp; ///< gamma map. gamma = [gammau; gammab]
    Teuchos::RCP<TV> gammaRcp;            ///< the unknown constraint force magnitude gamma
    Teuchos::RCP<TV> gammauRcp;           ///< the unknown unilateral constraint
    Teuchos::RCP<TV> gammabRcp;           ///< the unknown bilateral constraint

    // composite vectors and operators
    Teuchos::RCP<TOP> conOpRcp;  ///< the operator
    Teuchos::RCP<TV> velncRcp;   ///< the non-constraint velocity vel_nc
    Teuchos::RCP<TV> delta0Rcp;  ///< the current (geometric) delta vector delta_0 = [delta_0u ; delta_0b]
    Teuchos::RCP<TV> deltancRcp; ///< delta_nc = [Du^Trans vel_nc,u ; Db^Trans vel_nc,b]
    Teuchos::RCP<TV> bRcp;       ///< the constant part of BCQP problem. b = delta_0 + delta_nc

    // block vectors and operators
    Teuchos::RCP<TOP> mobOpRcp;        ///< mobility operator, 6 dof per obj to 6 dof per obj
    Teuchos::RCP<TCMAT> DuMatTransRcp; ///< unilateral constraint matrix
    Teuchos::RCP<TCMAT> DbMatTransRcp; ///< bilateral constraint matrix
    Teuchos::RCP<TV> delta0uRcp;       ///< unilateral delta0 vector, built with Du^Trans
    Teuchos::RCP<TV> delta0bRcp;       ///< bilateral delta0 vector, built with Dc^Trans
    Teuchos::RCP<TV> deltancuRcp;      ///< delta_nc,u = Du^Trans vel_nc,u
    Teuchos::RCP<TV> deltancbRcp;      ///< delta_nc,b = Db^Trans vel_nc,b
    std::vector<double> invKappa;      ///< inverse of spring constant kappa

    /**
     * @brief setup the constant \f$\delta\f$ vector in BCQP
     *
     */
    void setupDeltaVec();

};

#endif