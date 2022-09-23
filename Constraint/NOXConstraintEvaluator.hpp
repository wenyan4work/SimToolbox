#ifndef NOXCONSTRAINTEVALUATOR_HPP
#define NOXCONSTRAINTEVALUATOR_HPP

// Our stuff
#include "ConstraintCollector.hpp"
#include "PartialSepPartialGammaOp.hpp"
#include "Sylinder/SylinderSystem.hpp"
#include "Trilinos/TpetraUtil.hpp"

// NOX Stuff
#include <Thyra_StateFuncModelEvaluatorBase.hpp>

// Trilinos Stuff
#include <Thyra_DefaultLinearOpSource.hpp>
#include <Thyra_Ifpack2PreconditionerFactory.hpp>
#include <TpetraExt_MatrixMatrix.hpp>
#include <TpetraExt_TripleMatrixMultiply.hpp>
#include <Tpetra_CrsMatrix.hpp>

/** \brief Bilateral constraint evaluator
 *
 * The Jacobian for this system is dense, so this creates a matrix-free Jacobian operator class.
 * The preconditioner this computes is the inverse diagonal of the Jacobian.
 */
class EvaluatorTpetraConstraint : public Thyra::StateFuncModelEvaluatorBase<Scalar> {
  public:
    // Constructor
    EvaluatorTpetraConstraint(const Teuchos::RCP<const TCOMM> &commRcp, const Teuchos::RCP<const TOP> &mobOpRcp,
                              std::shared_ptr<ConstraintCollector> conCollectorPtr,
                              std::shared_ptr<SylinderSystem> ptcSystemPtr, const Teuchos::RCP<TV> &forceRcp,
                              const Teuchos::RCP<TV> &velRcp, const double dt);

    /** \name Initializers/Accessors */
    //@{

    /** \brief . */
    void setShowGetInvalidArgs(bool showGetInvalidArg);

    void set_W_factory(const Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar>> &W_factoryRcp);

    //@}

    /** \name Public functions overridden from ModelEvaulator. */
    //@{

    /** \brief . */
    Teuchos::RCP<const thyra_vec_space> get_x_space() const;
    /** \brief . */
    Teuchos::RCP<const thyra_vec_space> get_f_space() const;
    /** \brief . */
    Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const;
    /** \brief . */
    Teuchos::RCP<thyra_op> create_W_op() const;
    // /** \brief . */
    // Teuchos::RCP<thyra_prec> create_W_prec() const;
    /** \brief . */
    Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar>> get_W_factory() const;
    /** \brief . */
    Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;
    //@}

    void recursionStep(const Teuchos::RCP<const TV> &gammaRcp);

  private:
    /** \name Private functions overridden from ModelEvaulatorDefaultBase. */
    //@{

    /** \brief . */
    Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;
    /** \brief . */
    void evalModelImpl(const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
                       const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs) const;

    //@}



    // debug functions
    void dumpToFile(const Teuchos::RCP<const TOP> &JRcp, const Teuchos::RCP<const TV> &fRcp,
                    const Teuchos::RCP<const TV> &projMaskRcp,
                    const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp, const Teuchos::RCP<const TV> &xRcp,
                    const Teuchos::RCP<const TOP> &mobOpRcp, const Teuchos::RCP<const TCMAT> &AMatTransRcp,
                    const Teuchos::RCP<const TV> &constraintDiagonalRcp) const;

  private: // data members
    mutable bool update_ = false;
    mutable int stepCount_ = 1;
    double dt_;

    std::shared_ptr<ConstraintCollector> conCollectorPtr_;
    std::shared_ptr<SylinderSystem> ptcSystemPtr_;
    const Teuchos::RCP<const TCOMM> commRcp_;
    const Teuchos::RCP<const TOP> mobOpRcp_;
    const Teuchos::RCP<const TOP> mobInvOpRcp_;
    Teuchos::RCP<const TMAP> mobMapRcp_; ///< map for mobility matrix. 6 DOF per obj

    Teuchos::RCP<TCMAT>
        AMatTransRcp_; ///< A^Trans matrix TODO: are these allowed to be mutable? No. The RCP object CANNOT change
                       ///< during computation. The contents can be changed without using mutablle
    Teuchos::RCP<PartialSepPartialGammaOp>
        partialSepPartialGammaOpRcp_; ///< dt S^T A^T M A S which takes gamma to change in sep w.r.t gamma
    Teuchos::RCP<TV> partialSepPartialGammaDiagRcp_; ///< the diagonal of dt S^T A^T M A S

    Teuchos::RCP<TV> statusRcp_;             ///< status mask: if 1 for active, otherwise 0.
    Teuchos::RCP<TV> forceRcp_;              ///< force = A forceMag
    Teuchos::RCP<TV> forceMagRcp_;           ///< forceMag = S gamma
    Teuchos::RCP<TV> velRcp_;                ///< vel = M force
    Teuchos::RCP<TV> constraintFlagRcp_;     ///< bilateral flag vector
    Teuchos::RCP<TV> constraintDiagonalRcp_; ///< Diagonal of the matrix to add to the jacobian
    Teuchos::RCP<TV> xGuessRcp_;             ///< initial guess
    Teuchos::RCP<TV> sep0Rcp_;               ///< initial unconstrained constraint violation
    Teuchos::RCP<TV> sepRcp_;                ///< dt D^T M D gamma

    Teuchos::RCP<const thyra_vec_space> xSpaceRcp_;
    Teuchos::RCP<const TMAP> xMapRcp_;

    Teuchos::RCP<const thyra_vec_space> fSpaceRcp_;
    Teuchos::RCP<const TMAP> fMapRcp_;

    Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar>> W_factoryRcp_;

    Thyra::ModelEvaluatorBase::InArgs<Scalar> nominalValues_;
    Teuchos::RCP<thyra_vec> x0Rcp_;
    bool showGetInvalidArg_;
    Thyra::ModelEvaluatorBase::InArgs<Scalar> prototypeInArgs_;
    Thyra::ModelEvaluatorBase::OutArgs<Scalar> prototypeOutArgs_;

    mutable Teuchos::RCP<Teuchos::Time> residTimer_;
    mutable Teuchos::RCP<Teuchos::Time> intOpTimer_;
};

class JacobianOperator : public TOP {
  public:
    // Constructor
    JacobianOperator(const Teuchos::RCP<const TMAP> &xMapRcp);

    void initialize(const Teuchos::RCP<const PartialSepPartialGammaOp> &PartialSepPartialGammaOpRcp,
                    const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp,
                    const Teuchos::RCP<const TV> &constraintDiagonalRcp, const Teuchos::RCP<const TV> &projMaskRcp,
                    const double dt);

    void unitialize();

    virtual Teuchos::RCP<const TMAP> getDomainMap() const;

    virtual Teuchos::RCP<const TMAP> getRangeMap() const;

    virtual bool hasTransposeApply() const;

    virtual void apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
                       Scalar alpha = Teuchos::ScalarTraits<Scalar>::one(),
                       Scalar beta = Teuchos::ScalarTraits<Scalar>::zero()) const;

  private:
    // misc
    double dt_;

    // constant operators
    const Teuchos::RCP<const TMAP> xMapRcp_;               ///< map for combined vector [gammau; gammab]^T
    Teuchos::RCP<const TV> statusRcp_;                     ///< projection mask: if 1 for apply projection, otherwise 0.
    Teuchos::RCP<const TV> constraintDiagonalRcp_;         ///< K^{-1} diagonal matrix
    Teuchos::RCP<const TV> partialSepPartialGammaDiagRcp_; ///< diagonal of dt A^T M A
    Teuchos::RCP<const PartialSepPartialGammaOp>
        partialSepPartialGammaOpRcp_; ///< dt S^T A^T M A S, which maps gamma to change in sep w.r.t gamma

    Teuchos::RCP<TV> activeXcolRcp_;  ///< active input
    Teuchos::RCP<TV> changeInSepRcp_; ///< force_mag = S gamma
};

#endif
