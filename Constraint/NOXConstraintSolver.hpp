/**
 * @file NOXConstraintSolver.hpp
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

// Trilinos stuff
#include <Teuchos_AbstractFactoryStd.hpp>
#include <Thyra_Ifpack2PreconditionerFactory.hpp>

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
class NOXConstraintSolver {

  public:
    /**
     * @brief Construct a new NOXConstraintSolver object with particle system access
     *
     * This constructor calls setup() internally
     * @param config ParticleConfig object
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    NOXConstraintSolver(const Teuchos::RCP<const Teuchos::Comm<int>> &commRcp,
                     std::shared_ptr<ConstraintCollector> conCollectorPtr,
                     std::shared_ptr<ParticleSystem> ptcSystemPtr);

    // destructor
    ~NOXConstraintSolver() = default;

    // forbid copy
    NOXConstraintSolver(const NOXConstraintSolver &) = delete;
    NOXConstraintSolver(NOXConstraintSolver &&) = delete;
    const NOXConstraintSolver &operator=(const NOXConstraintSolver &) = delete;
    const NOXConstraintSolver &operator=(NOXConstraintSolver &&) = delete;

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
     */
    void setup(const double dt);

    /**
     * @brief solve the nonlinear complementarity problem
     *
     */
    void solveConstraints();

  private:
    double dt_;                               ///< timestep size
    const Teuchos::RCP<const TCOMM> commRcp_; ///< TCOMM, set as a Teuchos::MpiComm object in constructor
    Teuchos::RCP<const TCMAT> mobMatRcp_;     ///< mobility matrix, 6 dof per obj to 6 dof per obj
    Teuchos::RCP<const TMAP> mobMapRcp_;      ///< map for mobility matrix. 6 DOF per obj

    std::shared_ptr<ConstraintCollector> conCollectorPtr_; ///< pointer to ConstraintCollector
    std::shared_ptr<ParticleSystem> ptcSystemPtr_;         ///< pointer to ParticleSystem
    Teuchos::RCP<TV> gammaRcp_;                            ///< constraint Lagrange multiplier
    Teuchos::RCP<TV> velConRcp_;                           ///< constraint velocity
    Teuchos::RCP<TV> forceConRcp_;                         ///< constraint force

    // NOX stuff
    Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<Scalar>> lowsFactory_; ///< linear operator params/factory
    Teuchos::RCP<NOX::StatusTest::Combo> statusTests_;                      ///< status tests to check for convergence
    Teuchos::RCP<Teuchos::ParameterList> nonlinearParams_;                  ///< nonlinear params
};

#endif