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
#ifndef NOXCONSTRAINTSOLVER_HPP_
#define NOXCONSTRAINTSOLVER_HPP_

// Our stuff
#include "ConstraintCollector.hpp"
#include "NOXConstraintEvaluator.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"

// regular C++
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
    /**
     * @brief Construct a new ConstraintSolver object with particle system access
     *
     * This constructor calls setup() internally
     * @param config SylinderConfig object
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    ConstraintSolver(const Teuchos::RCP<const Teuchos::Comm<int> >& commRcp,
                     std::shared_ptr<ConstraintCollector> conCollectorPtr, 
                     std::shared_ptr<SylinderSystem> ptcSystemPtr);

    // destructor
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
     * @brief setup this solver for solution
     *
     * @param constraint_
     * @param objMobMapRcp_
     * @param dt_
     */
    void setup(double dt);

    // /**
    //  * @brief dump the jacobian and constraint values to MatrixMarket files
    //  *
    //  */
    // void dumpConstraints();

    /**
     * @brief solve the nonlinear complementarity problem
     *
     */
    void solveConstraints();

    /**
     * @brief write the solution constraint force magnitude back to uniConstraints and biConstraints
     *
     */
    void writebackGamma();

  private:
    double dt_; ///< timestep size
    const Teuchos::RCP<const TCOMM> commRcp_;   ///< TCOMM, set as a Teuchos::MpiComm object in constructor
    Teuchos::RCP<const TOP> mobOpRcp_;         ///< mobility operator, 6 dof per obj to 6 dof per obj
    std::shared_ptr<ConstraintCollector> conCollectorPtr_; ///< pointer to ConstraintCollector
    std::shared_ptr<SylinderSystem> ptcSystemPtr_;         ///< pointer to SylinderSystem
    Teuchos::RCP<const TV> gammaRcp_; ///< constraint Lagrange multiplier

    // NOX stuff
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<Scalar>> lowsFactory_; ///< linear operator params/factory
    Teuchos::RCP<NOX::StatusTest::Combo> statusTests_;                      ///< status tests to check for convergence
    Teuchos::RCP<Teuchos::ParameterList> nonlinearParams_;                  ///< nonlinear params
};

#endif