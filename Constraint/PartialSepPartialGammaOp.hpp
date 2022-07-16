#ifndef PARTIALSEPPARTIALGAMMAOP_HPP_
#define PARTIALSEPPARTIALGAMMAOP_HPP_

#include "Trilinos/TpetraUtil.hpp"


/////////////////////////////////////////
// Change in sep w.r.t. gamma operator //
/////////////////////////////////////////

class PartialSepPartialGammaOp : public TOP
{
public:
  // Constructor
  PartialSepPartialGammaOp(const Teuchos::RCP<const TMAP> &xMapRcp);

  void initialize(const Teuchos::RCP<const TOP> &mobOpRcp, 
           const Teuchos::RCP<const TCMAT> &AMatTransRcp,
           const Teuchos::RCP<const TCMAT> &AMatRcp,
           const Teuchos::RCP<TV> &forceRcp,
           const Teuchos::RCP<TV> &velRcp,
           const double dt);

  void unitialize();

  virtual Teuchos::RCP<const TMAP> getDomainMap() const;

  virtual Teuchos::RCP<const TMAP> getRangeMap() const;

  virtual bool hasTransposeApply() const;

  virtual void
  apply(const TMV& X,
        TMV& Y,
        Teuchos::ETransp mode = Teuchos::NO_TRANS,
        Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
        Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const;

private:

  // misc
  double dt_;

  // constant operators
  Teuchos::RCP<const TOP> mobOpRcp_; ///< mobility matrix
  Teuchos::RCP<const TCMAT> AMatTransRcp_;
  Teuchos::RCP<const TCMAT> AMatRcp_;
  Teuchos::RCP<const TMAP> xMapRcp_; ///< map for combined vector [gammau; gammab]^T

  Teuchos::RCP<TV> forceMagRcp_; ///< force_mag = S gamma
  Teuchos::RCP<TV> forceRcp_; ///< force = A force_mag
  Teuchos::RCP<TV> velRcp_;   ///< vel = M force
};

#endif
