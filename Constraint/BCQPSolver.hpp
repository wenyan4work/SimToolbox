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
    BCQPSolver(const Teuchos::RCP<const TOP> &ARcp, const Teuchos::RCP<const TV> &bRcp);

    /**
     * @brief Construct a new CPSolver object generating \f$A,b\f$ for internal test
     *
     * @param localSize
     * @param diagonal
     */
    BCQPSolver(int localSize, double diagonal = 0.0);

    /**
     * @brief Set lb for the problem
     * bound must have compatible map to A & b
     * @param lbRcp
     */
    void setLowerBound(const Teuchos::RCP<TV> &lbRcp_) {
        lbSet = true;
        lbRcp = lbRcp_;
    }

    /**
     * @brief Set ub for the problem
     * bound must have compatible map to A & b
     * @param ubRcp
     */
    void setUpperBound(const Teuchos::RCP<TV> &ubRcp_) {
        ubSet = true;
        ubRcp = ubRcp_;
    };

    Teuchos::RCP<TV> getLowerBound(){
      return lbRcp;
    }

    Teuchos::RCP<TV> getUpperBound(){
      return ubRcp;
    }

    /**
     * @brief call this before any solve() functions
     *
     */
    void prepareSolver() { setDefaultBounds(); }

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
    Teuchos::RCP<const TOP> ARcp;      ///< linear operator \f$A\f$
    Teuchos::RCP<const TV> bRcp;       ///< vector \f$b\f$
    Teuchos::RCP<const TMAP> mapRcp;   ///< map for the distribution of xsolRcp, bRcp, and ARcp->rowMap
    Teuchos::RCP<const TCOMM> commRcp; ///< Teuchos::MpiComm
    Teuchos::RCP<TV> lbRcp;      ///< lower bound
    Teuchos::RCP<TV> ubRcp;      ///< upper bound
    bool lbSet = false;
    bool ubSet = false;

    /**
     * @brief Set default bounds (infinity) if no bounds set
     *
     */
    void setDefaultBounds();

    /**
     * @brief project the vector to [lb,ub]
     *
     * @param vecRcp
     */
    void boundProjection(Teuchos::RCP<TV> &vecRcp) const;

    /**
     * @brief check the residual with EQ 2.2 of Dai & Fletcher 2005
     *
     * @param XRcp X
     * @param YRcp Y=AX+b
     * @param QRcp temporary working space
     * @return double
     */
    double checkProjectionResidual(const Teuchos::RCP<const TV> &XRcp, const Teuchos::RCP<const TV> &YRcp,
                                   const Teuchos::RCP<TV> &QRcp) const;

    /**
     * @brief generate random lb and ub, used for internal tests only
     *
     */
    void generateRandomBounds();
};

#endif