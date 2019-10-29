/**
 * @file CPSolver.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief LCP Solver
 * @version 0.1
 * @date 2018-12-14
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef CPSOLVER_HPP_
#define CPSOLVER_HPP_

#include "Trilinos/TpetraUtil.hpp"

#include <array>
#include <deque>
#include <vector>

#include <Tpetra_RowMatrixTransposer_decl.hpp>

using IteHistory = std::deque<std::array<double, 6>>; ///< recording iteration history

/**
 * @brief the operator \f$F_c^T M F_c\f$ in the LCP problem
 *
 */
class CPMatOp : public TOP {
  public:
    /**
     * @brief Construct a new CPMatOp object
     *
     * @param mobRcp_ mobility operator derived from Tpetra::Operator
     * @param fcTransRcp_ \f$F_c^T\f$ matrix in Tpetra::CrsMatrix format
     */
    CPMatOp(Teuchos::RCP<TOP> mobRcp_, Teuchos::RCP<TCMAT> fcTransRcp_) : mobRcp(mobRcp_), fcTransRcp(fcTransRcp_) {
        // check map
        TEUCHOS_TEST_FOR_EXCEPTION(!(mobRcp->getRangeMap()->isSameAs(*(fcTransRcp->getDomainMap()))),
                                   std::invalid_argument, "Mob and Fc Maps not compatible.");
        this->forceVecRcp = Teuchos::rcp(new TV(mobRcp->getRangeMap().getConst(), true));
        this->velVecRcp = Teuchos::rcp(new TV(mobRcp->getRangeMap().getConst(), true));
        // explicit transpose
        Tpetra::RowMatrixTransposer<TCMAT::scalar_type, TCMAT::local_ordinal_type, TCMAT::global_ordinal_type>
            transposer(fcTransRcp);
        fcRcp = transposer.createTranspose();
    }

    /**
     * @brief apply this operator. 
     *
     * @param X
     * @param Y
     * @param mode if the operator should be applied as transposed
     * @param alpha
     * @param beta
     */
    void apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const {
#ifdef DEBUGLCPCOL
        TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                                   "X and Y do not have the same numbers of vectors (columns).");
        TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                                   "X and Y do not have the same Map.");
#endif
        const int numVecs = X.getNumVectors();
        for (int i = 0; i < numVecs; i++) {
            const Teuchos::RCP<const TV> XcolRcp = X.getVector(i);
            Teuchos::RCP<TV> YcolRcp = Y.getVectorNonConst(i);
            // step 1 force=Fc * Xcol
            Teuchos::RCP<Teuchos::Time> transTimer = Teuchos::TimeMonitor::getNewCounter("BBPGD::OP::Fc Apply");
            {
                Teuchos::TimeMonitor mon(*transTimer);
                // fcTransRcp->apply(*XcolRcp, *forceVecRcp, Teuchos::TRANS);
                fcRcp->apply(*XcolRcp, *forceVecRcp);
            }
            // step 2 vel = mob * Force
            Teuchos::RCP<Teuchos::Time> mobTimer = Teuchos::TimeMonitor::getNewCounter("BBPGD::OP::Mob Apply");
            {
                Teuchos::TimeMonitor mon(*mobTimer);
                mobRcp->apply(*forceVecRcp, *velVecRcp);
            }
            // step 3 Ycol = Fc^T * vel
            Teuchos::RCP<Teuchos::Time> fcTimer = Teuchos::TimeMonitor::getNewCounter("BBPGD::OP::FcTrans Apply");
            {
                Teuchos::TimeMonitor mon(*fcTimer);
                fcTransRcp->apply(*velVecRcp, *YcolRcp);
            }
            // fcTransRcp->apply(*XcolRcp, *forceVecRcp, Teuchos::TRANS);
            // mobRcp->apply(*forceVecRcp, *velVecRcp);
            // fcTransRcp->apply(*velVecRcp, *YcolRcp);
        }
    }
    /**
     * @brief Get the Domain Map object. interface required by Tpetra::Operator
     *
     * @return Teuchos::RCP<const TMAP>
     */
    Teuchos::RCP<const TMAP> getDomainMap() const {
        return this->fcTransRcp->getRangeMap(); // Get the domain Map of this Operator subclass.
    }
    /**
     * @brief Get the Range Map object. interface required by Tpetra::Operator
     *
     * @return Teuchos::RCP<const TMAP>
     */
    Teuchos::RCP<const TMAP> getRangeMap() const {
        return this->fcTransRcp->getRangeMap(); // Get the range Map of this Operator subclass.
    }
    /**
     * @brief return if this operator can be applied as transposed. interface required by Tpetra::Operator
     *
     * @return true
     * @return false
     */
    bool hasTransposeApply() const { return false; }

    Teuchos::RCP<TOP> mobRcp;       ///< mobility operator
    Teuchos::RCP<TCMAT> fcTransRcp; ///< \f$F_c^T\f$ matrix
    Teuchos::RCP<TV> forceVecRcp;   ///< force vector
    Teuchos::RCP<TV> velVecRcp;     ///< velocity vector \f$ V = M F\f$
    Teuchos::RCP<TCMAT> fcRcp;      ///< explicit transpose of fcTrans
};

/**
 * @brief the CP problem \f$Ax+b\f$ solver class
 *
 */
class CPSolver {
  public:
    /**
     * @brief Construct a new CPSolver object with matrix
     *
     * @param A_ the linear operator \f$A\f$
     * @param b_ the vector \f$b\f$
     */
    CPSolver(const Teuchos::RCP<const TOP> &A_, const Teuchos::RCP<const TV> &b_);

    /**
     * @brief Construct a new CPSolver object generating \f$A,b\f$ for internal test
     *
     * @param localSize
     * @param diagonal
     */
    CPSolver(int localSize, double diagonal = 0.0);

    /**
     * @brief Nesterov accelerated PGD
     *
     * @param xsolRcp initial guess and result
     * @param tol residual tolerance
     * @param iteMax max iteration number
     * @param history iteration history
     * @return int return error code. 0 for normal execution.
     */
    int LCP_APGD(Teuchos::RCP<TV> &xsolRcp, const double tol, const int iteMax, IteHistory &history) const;

    /**
     * @brief Barzilai-Borwein PGD
     *
     * @param xsolRcp initial guess and result
     * @param tol residual tolerance
     * @param iteMax max iteration number
     * @param history iteration history
     * @return int return error code. 0 for normal execution.
     */
    int LCP_BBPGD(Teuchos::RCP<TV> &xsolRcp, const double tol, const int iteMax, IteHistory &history) const;

    /**
     * @brief minimal-map Newton solver
     *
     * @param xsolRcp initial guess and result
     * @param tol residual tolerance
     * @param iteMax max iteration number
     * @param history iteration history
     * @return int return error code. 0 for normal execution.
     */
    int LCP_mmNewton(Teuchos::RCP<TV> &xsolRcp, const double tol, const int iteMax, IteHistory &history) const;

    /**
     * @brief self test
     *
     * @param tol
     * @param maxIte
     * @param solverChoice
     * @return int
     */
    int test_LCP(double tol, int maxIte, int solverChoice);

  private:
    Teuchos::RCP<const TOP> ARcp;      ///< linear operator \f$A\f$
    Teuchos::RCP<const TV> bRcp;       ///< vector \f$b\f$
    Teuchos::RCP<const TMAP> mapRcp;   ///< map for the distribution of xsolRcp, bRcp, and ARcp->rowMap
    Teuchos::RCP<const TCOMM> commRcp; ///< Teuchos::MpiComm

    // functions for internal use
    /**
     * @brief clip negative entries to zero
     *
     * @param vecRcp
     */
    void clipZero(Teuchos::RCP<TV> &vecRcp) const;

    /**
     * @brief compute \f$ Z_i = max(X_i,Y_i) \f$
     *
     * @param vecXRcp
     * @param vecYRcp
     * @param vecZRcp
     */
    void maxXY(const Teuchos::RCP<const TV> &vecXRcp, const Teuchos::RCP<const TV> &vecYRcp,
               const Teuchos::RCP<TV> &vecZRcp) const;

    /**
     * @brief compute \f$ Z_i = min(X_i,Y_i) \f$
     *
     * @param vecXRcp
     * @param vecYRcp
     * @param vecZRcp
     */
    void minXY(const Teuchos::RCP<const TV> &vecXRcp, const Teuchos::RCP<const TV> &vecYRcp,
               const Teuchos::RCP<TV> &vecZRcp) const;

    /**
     * @brief the minimal-map function \f$h\f$
     *
     * @param xRcp
     * @param yRcp
     * @param hRcp
     * @param maskRcp mask_i =  (x_i < y_i) ?1:0;
     */
    void hMinMap(const Teuchos::RCP<const TV> &xRcp, const Teuchos::RCP<const TV> &yRcp, const Teuchos::RCP<TV> &hRcp,
                 const Teuchos::RCP<TV> &maskRcp) const;
    /**
     * @brief   compute the residual function with Y = Ax
     *
     * @param vecXRcp
     * @param vecYRcp
     * @param vecbRcp
     * @param vecTempRcp temporary working space with map identical to both X and Y
     * @return double the residual
     */
    double checkResiduePhi(const Teuchos::RCP<const TV> &vecXRcp, const Teuchos::RCP<const TV> &vecYRcp,
                           const Teuchos::RCP<const TV> &vecbRcp, const Teuchos::RCP<TV> &vecTempRcp) const;

    /**
     * @brief   compute the residual function with Y = Ax+b
     *
     * @param vecXRcp
     * @param vecYRcp
     * @param vecTempRcp temporary working space with map identical to both X and Y
     * @return double the residual
     */
    double checkResiduePhi(const Teuchos::RCP<const TV> &vecXRcp, const Teuchos::RCP<const TV> &vecYRcp,
                           const Teuchos::RCP<TV> &vecTempRcp) const;
};

#endif