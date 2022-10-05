/**
 * @file PGDConstraintSolver.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Solve the uni/bi-lateral constraint problem
 * @version 0.1
 * @date 2019-11-14
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef PGDCONSTRAINTSOLVER_HPP_
#define PGDCONSTRAINTSOLVER_HPP_

#include "BCQPSolver.hpp"
#include "ConstraintCollector.hpp"
#include "ConstraintJacobianOp.hpp"
#include "Sylinder/SylinderSystem.hpp"

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
class PGDConstraintSolver {

  public:
    /**
     * @brief Construct a new PGDConstraintSolver object with particle system access
     *
     * This constructor calls setup() internally
     * @param config SylinderConfig object
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    PGDConstraintSolver(const Teuchos::RCP<const Teuchos::Comm<int>> &commRcp,
                        std::shared_ptr<ConstraintCollector> conCollectorPtr,
                        std::shared_ptr<SylinderSystem> ptcSystemPtr);

    // destructor
    ~PGDConstraintSolver() = default;

    // forbid copy
    PGDConstraintSolver(const PGDConstraintSolver &) = delete;
    PGDConstraintSolver(PGDConstraintSolver &&) = delete;
    const PGDConstraintSolver &operator=(const PGDConstraintSolver &) = delete;
    const PGDConstraintSolver &operator=(PGDConstraintSolver &&) = delete;

    /**
     * @brief reset the parameters and release all allocated spaces
     *
     */
    void reset();

    /**
     * @brief initializes all data structures
     *
     */
    void initialize();

    /**
     * @brief reinitializes all data structures that depend on the number of constraints
     *
     */
    void reinitialize();

    /**
     * @brief setup this solver. Initializes all data structures
     *
     * @param constraint_
     * @param objMobMapRcp_
     */
    void setup(const double dt, const double res = 1e-5, const int maxIterations = 1e5, const int maxRecursions = 100,
               const int solverChoice = 0);

    /**
     * @brief solve the recursively generated BCQP problem
     *
     */
    void solveConstraints();

    void recursionStep();

  private:
    double dt_;         ///< timestep size
    double res_;        ///< residual tolerance
    int maxIterations_; ///< max iterations
    int maxRecursions_; ///< max recursions
    int solverChoice_;  ///< which solver to use

    std::shared_ptr<ConstraintCollector> conCollectorPtr_; ///< pointer to ConstraintCollector
    std::shared_ptr<SylinderSystem> ptcSystemPtr_;         ///< pointer to SylinderSystem

    const Teuchos::RCP<const TCOMM> commRcp_; ///< TCOMM, set as a Teuchos::MpiComm object in constructor
    Teuchos::RCP<const TMAP> mobMapRcp_;      ///< distributed map for obj mobility. 6 dof per obj
    Teuchos::RCP<const TMAP> gammaMapRcp_;    ///< distributed map for constraints. 1 dof per constraint
    Teuchos::RCP<TCMAT> mobMatRcp_;           ///< mobility matrix, 6 dof per obj maps to 6 dof per obj

    Teuchos::RCP<TV> forceConRcp_; ///< constraints force vec, 6 dof per obj
    Teuchos::RCP<TV> velConRcp_;   ///< constraints velocity vec, 6 dof per obj
    Teuchos::RCP<TV> forceExtRcp_; ///< external (non-constraint) force, 6 dof per obj
    Teuchos::RCP<TV> velExtRcp_;   ///< external (non-constraint) velocity, 6 dof per obj

    Teuchos::RCP<TCMAT> DMatTransRcp_; ///< D^Trans matrix
    Teuchos::RCP<TCMAT>
        DMatRcp_; ///< D matrix TODO: is is better to build this explicitly or to use implicit tranpose of D^T?
    Teuchos::RCP<TV> constraintDiagonalRcp_; ///< K^{-1} diagonal matrix
    Teuchos::RCP<TV> biFlagRcp_; ///< constraint flag. 0 if unilaterial constraint, 1 if bilaterial constraint

    // the linear complementarity problem 0 <= A gamma + S0 _|_ gamma >= 0
    Teuchos::RCP<ConstraintJacobianOp>
        constraintJacobianOp_;  ///< operator taking gamma to change in sep w.r.t gamma
                                ///< this is the operator of BCQP problem. A = dt D^T M D + K^{-1}
    Teuchos::RCP<TV> gammaRcp_; ///< the unknown constraint Lagrange multipliers (overall)
    Teuchos::RCP<TV> sepRcp_;   ///< final, unconstrained constraint violation (A gamma + S0)
    Teuchos::RCP<TV> sep0Rcp_;  ///< initial, unconstrained constraint violation (S0)
                                ///< this is the constant part of BCQP problem
};

#endif