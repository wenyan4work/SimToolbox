/**
 * @file BCQPSolver.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief Bound Constrained Quadratic Programming
 *        solver by projected gradient descent
 * @version 0.1
 * @date 2019-10-21
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef BCQPSOLVER_HPP_
#define BCQPSOLVER_HPP_

#include "ConstraintCollector.hpp"
#include "Trilinos/TpetraUtil.hpp"

#include <array>
#include <deque>
#include <vector>

#include <Tpetra_RowMatrixTransposer_decl.hpp>

using IteHistory = std::deque<std::array<double, 6>>; ///< recording iteration history

/**
 * @brief the CP problem \f$Ax+b\f$ solver class
 *
 */
class BCQPSolver {
  public:
    /**
     * @brief Construct a new CPSolver object with matrix
     *
     * @param A_ the linear operator \f$A\f$
     * @param b_ the vector \f$b\f$
     */
    BCQPSolver(ConstraintCollector &conCollector_, const Teuchos::RCP<const TOP> &ARcp, const Teuchos::RCP<const TV> &bRcp);

    /**
     * @brief Construct a new CPSolver object generating \f$A,b\f$ for internal test
     *
     * @param localSize
     * @param diagonal
     */
    BCQPSolver(int localSize, double diagonal = 0.0);

    /**
     * @brief Nesterov accelerated PGD
     *
     * @param xsolRcp initial guess and result
     * @param tol residual tolerance
     * @param iteMax max iteration number
     * @param history iteration history
     * @return int return error code. 0 for normal execution.
     */
    int solveAPGD(Teuchos::RCP<TV> &xsolRcp, const double tol, const int iteMax, IteHistory &history) const;

    /**
     * @brief Barzilai-Borwein PGD
     *
     * @param xsolRcp initial guess and result
     * @param tol residual tolerance
     * @param iteMax max iteration number
     * @param history iteration history
     * @return int return error code. 0 for normal execution.
     */
    int solveBBPGD(Teuchos::RCP<TV> &xsolRcp, const double tol, const int iteMax, IteHistory &history) const;

    /**
     * @brief self test
     *
     * @param tol
     * @param maxIte
     * @param solverChoice
     * @return int
     */
    int selfTest(double tol, int maxIte, int solverChoice);

  private:
    ConstraintCollector conCollector; ///< constraints
    Teuchos::RCP<const TOP> ARcp;      ///< linear operator \f$A\f$
    Teuchos::RCP<const TV> bRcp;       ///< vector \f$b\f$
    Teuchos::RCP<const TMAP> mapRcp;   ///< map for the distribution of xsolRcp, bRcp, and ARcp->rowMap
    Teuchos::RCP<const TCOMM> commRcp; ///< Teuchos::MpiComm
};

#endif