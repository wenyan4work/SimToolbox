#ifndef CONSTRAINTJACOBIANOP_HPP_
#define CONSTRAINTJACOBIANOP_HPP_

#include "Trilinos/TpetraUtil.hpp"

/////////////////////////////////////////
// Change in sep w.r.t. gamma operator //
/////////////////////////////////////////

class ConstraintJacobianOp : public TOP {
  public:
    // Constructor
    ConstraintJacobianOp(const Teuchos::RCP<const TMAP> &xMapRcp);

    void initialize(const Teuchos::RCP<const TOP> &mobOpRcp, const Teuchos::RCP<const TCMAT> &DMatRcp,
                    const Teuchos::RCP<const TCMAT> &DMatTransRcp, const Teuchos::RCP<const TV> &invKappaDiagRcp,
                    const double dt);

    void unitialize();

    virtual Teuchos::RCP<const TMAP> getDomainMap() const;

    virtual Teuchos::RCP<const TMAP> getRangeMap() const;

    virtual bool hasTransposeApply() const;

    virtual void apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
                       Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
                       Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const;

    virtual void applyMatrixFree(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
                                 Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
                                 Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const;

    virtual void applyExplicitMatrix(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
                                     Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
                                     Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const;

  private:
    // misc
    double dt_;
    double matrixFree_; // switch between matrix free or explicit apply

    // constant operators
    Teuchos::RCP<const TMAP> xMapRcp_;       ///< map for constraint Lagrange multiplier
    Teuchos::RCP<const TOP> mobOpRcp_;       ///< mobility operator
    Teuchos::RCP<const TCMAT> mobMatRcp_;    ///< mobility matrix
    Teuchos::RCP<const TCMAT> DMatRcp_;      ///< sparce matrix mapping from COM velocity to change in sep
    Teuchos::RCP<const TCMAT> DMatTransRcp_; ///< sparce matrix mapping from constraint Lagrange multiplier to COM force
    Teuchos::RCP<const TV> invKappaDiagRcp_; ///< diagonal of the matrix added to the Jacobian.

    // internal temporary storage
    Teuchos::RCP<TCMAT> partialSepPartialGammaMatRcp_; ///< sparce matrix mapping x to change in sep w.r.t x
    Teuchos::RCP<TV> forceRcp_;                        ///< force = D x
    Teuchos::RCP<TV> velRcp_;                          ///< vel = M force
};

#endif
