/**
 * @file ConstraintOperator.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-10-17
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef CONSTRAINTOPERATOR_HPP_
#define CONSTRAINTOPERATOR_HPP_

#include "Trilinos/TpetraUtil.hpp"

#include <array>
#include <deque>
#include <vector>

/**
 * @brief Constraint Operator is a block matrix assembled from four blocks:
 *    [Du^T M Du       Du^T M Db           ]
 *    [Db^T M Du       Db^T M Db  +  K^{-1}]
 * The operator is applied on block vectors: [gammau; gammab]^T
 * Du^T, Db^T, M, and K^{-1} are explicitly constructed before constructing this object
 */
class ConstraintOperator : public TOP {
  public:
    /**
     * @brief Construct a new ConstraintOperator object
     *
     * @param mobOp
     * @param uniDuMat
     * @param biDbMat
     * @param invKappaDiagMat
     */
    ConstraintOperator(Teuchos::RCP<TOP> &mobOp_, Teuchos::RCP<TCMAT> &DMatTransRcp_, Teuchos::RCP<TV> &invKappa_);

    /**
     * @brief apply this operator, ensuring the block structure
     *
     * @param X
     * @param Y
     * @param mode if the operator should be applied as transposed
     * @param alpha
     * @param beta
     */
    void apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const;

    /**
     * @brief Get the Domain Map object. interface required by Tpetra::Operator
     *
     * @return Teuchos::RCP<const TMAP>
     */
    Teuchos::RCP<const TMAP> getDomainMap() const;

    /**
     * @brief Get the Range Map object. interface required by Tpetra::Operator
     *
     * @return Teuchos::RCP<const TMAP>
     */
    Teuchos::RCP<const TMAP> getRangeMap() const;

    /**
     * @brief return if this operator can be applied as transposed. interface required by Tpetra::Operator
     *
     * @return true
     * @return false
     */
    bool hasTransposeApply() const { return false; }

    void enableTimer();
    void disableTimer();

    Teuchos::RCP<TV> getForce() { return forceRcp; }
    Teuchos::RCP<TV> getVel() { return velRcp; }
    Teuchos::RCP<TCMAT> getDMat() { return DMatRcp; }

  private:
    // comm
    Teuchos::RCP<const TCOMM> commRcp; ///< the mpi communicator
    // constant operators
    Teuchos::RCP<TOP> mobOpRcp; ///< mobility matrix
    Teuchos::RCP<TCMAT> DMatTransRcp;
    Teuchos::RCP<TCMAT> DMatRcp;
    Teuchos::RCP<TV> invKappa; ///< 1/h K^{-1} diagonal matrix

    Teuchos::RCP<const TMAP> mobMapRcp;   ///< map for mobility matrix. 6 DOF per obj
    Teuchos::RCP<const TMAP> gammaMapRcp; ///< map for combined vector [gammau; gammab]^T

    Teuchos::RCP<TV> forceRcp; ///< force = D gamma
    Teuchos::RCP<TV> velRcp;   ///< vel = M force

    // time monitor
    Teuchos::RCP<Teuchos::Time> transposeDMat;
    Teuchos::RCP<Teuchos::Time> applyMobMat;
    Teuchos::RCP<Teuchos::Time> applyDMat;
    Teuchos::RCP<Teuchos::Time> applyDTransMat;
};

#endif