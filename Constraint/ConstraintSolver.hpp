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
     * @param solver_ choice of solver
     */
    void setControlParams(double res_, int maxIte_, int solver_) {
        res = res_;
        maxIte = maxIte_;
        solverChoice = solver_;
    }

    /**
     * @brief setup this solver for solution
     *
     * @param constraint_
     * @param objMobMapRcp_
     * @param dt_
     */
    void setup(ConstraintCollector &conCollector_, Teuchos::RCP<TOP> &mobOpRcp_, double dt_);

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

    /**
     * @brief write the final (linearlized) separation distance back to uniConstraints and biConstraints
     *
     */
    void writebackDelta();

    Teuchos::RCP<const TV> getForceCon() const { return forceConRcp; }
    Teuchos::RCP<const TV> getVelocityCon() const { return velConRcp; }

  private:
    double dt;        ///< timestep size
    double res;       ///< residual tolerance
    int maxIte;       ///< max iterations
    int solverChoice; ///< which solver to use

    ConstraintCollector conCollector; ///< constraints

    // mobility-map
    Teuchos::RCP<const TMAP> mobMapRcp; ///< distributed map for obj mobility. 6 dof per obj
    Teuchos::RCP<TOP> mobOpRcp;         ///< mobility operator, 6 dof per obj to 6 dof per obj
    Teuchos::RCP<TV> forceConRcp; ///< force vec, 6 dof per obj, due to all constraints
    Teuchos::RCP<TV> velConRcp;   ///< velocity vec, 6 dof per obj. due to all constraints

    // composite vectors and operators
    Teuchos::RCP<TCMAT> DMatTransRcp; ///< D^Trans matrix
    Teuchos::RCP<TV> invKappaRcp;     ///< K^{-1} diagonal matrix
    Teuchos::RCP<TV> deltaRcp;        ///< the current (geometric) delta vector containing the constraint function values
                                      ///< the initial delta value is the constant part of BCQP problem

    // the constraint problem M gamma + q
    Teuchos::RCP<ConstraintOperator> MOpRcp; ///< the operator of BCQP problem. M = D^T M D + K^{-1}
    Teuchos::RCP<TV> gammaRcp;               ///< the unknown constraint lagrange multipliers
};

#endif