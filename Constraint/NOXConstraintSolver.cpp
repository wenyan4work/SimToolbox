
#include "NOXConstraintSolver.hpp"
#include "Util/Logger.hpp"

#include <NOX_StatusTest_NormF.H>
#include <mpi.h>

ConstraintSolver::ConstraintSolver(const Teuchos::RCP<const Teuchos::Comm<int> >& commRcp,
                                   std::shared_ptr<ConstraintCollector> conCollectorPtr,
                                   std::shared_ptr<SylinderSystem> ptcSystemPtr)
    : commRcp_(commRcp), conCollectorPtr_(std::move(conCollectorPtr)), ptcSystemPtr_(std::move(ptcSystemPtr)) {
    //////////////////////////////////////////
    // Create the NOX linear solver factory //
    //////////////////////////////////////////
    // TODO: switch to CG. J is SPD, so CG will be far faster than GMRES
    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::parameterList();
    params->set("Linear Solver Type", "Belos");
    Teuchos::ParameterList &belosList = params->sublist("Linear Solver Types").sublist("Belos");
    belosList.set("Solver Type", "Pseudo Block GMRES");
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Num Blocks", 30);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Maximum Restarts", 20);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Maximum Iterations", 1000);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Convergence Tolerance", 1e-8);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Timer Label", "NOX_Linear_Belos");
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Verbosity", Belos::Errors + Belos::Warnings + Belos::TimingDetails + Belos::FinalSummary);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Show Maximum Residual Norm Only", false);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Output Frequency", 100);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Implicit Residual Scaling", "Norm of RHS");
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Explicit Residual Scaling", "Norm of RHS");
    belosList.sublist("VerboseObject").set("Verbosity Level", "medium");
    params->set("Preconditioner Type", "None");

    Stratimikos::DefaultLinearSolverBuilder builder;
    builder.setParameterList(params);
    lowsFactory_ = builder.createLinearSolveStrategy("");

    /////////////////////////////////
    // Create the NOX status tests //
    /////////////////////////////////
    // TODO: play around with these params
    NOX::Abstract::Vector::NormType normType = NOX::Abstract::Vector::NormType::MaxNorm; // OneNorm, TwoNorm, MaxNorm
    NOX::StatusTest::NormF::ScaleType scaleType = NOX::StatusTest::NormF::Unscaled;
    Teuchos::RCP<NOX::StatusTest::NormF> absresid = 
        Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-6, normType, scaleType));
    Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters = Teuchos::rcp(new NOX::StatusTest::MaxIters(1000));
    Teuchos::RCP<NOX::StatusTest::FiniteValue> fv = Teuchos::rcp(new NOX::StatusTest::FiniteValue);
    
    statusTests_ = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
    statusTests_->addStatusTest(fv);
    statusTests_->addStatusTest(absresid);
    statusTests_->addStatusTest(maxiters);

    ///////////////////////////////
    // Create nox parameter list //
    ///////////////////////////////
    nonlinearParams_ = Teuchos::parameterList();
    
    // set solver params
    nonlinearParams_->set("Nonlinear Solver", "Line Search Based"); // Line Search Based or Trust Region Based
    nonlinearParams_->sublist("Direction").sublist("Newton").sublist("Linear Solver").set("Tolerance", 1.0e-4);
    nonlinearParams_->sublist("Line Search").set("Method", "Full Step");

    // Set output params
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Debug", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Warning", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Error", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Test Details", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Details", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Parameters", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Linear Solver Details", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Inner Iteration", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Outer Iteration", true);
    nonlinearParams_->sublist("Printing").sublist("Output Information").set("Outer Iteration StatusTest", true);
}

void ConstraintSolver::setup(double dt) {
    reset();

    dt_ = dt;
    mobOpRcp_ = ptcSystemPtr_->getMobOperator();
}

void ConstraintSolver::reset() {
    mobOpRcp_.reset();
}

void ConstraintSolver::solveConstraints() {
    /////////////////////////////////////////////////
    // Check if there are any constraints to solve //
    /////////////////////////////////////////////////
    const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfConstraints();
    int numGlobalConstraints;
    MPI_Allreduce(&numLocalConstraints, &numGlobalConstraints, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (numGlobalConstraints == 0) { 
        return;
    }

    ///////////////////////////////////////
    // Create the model evaluator object //
    ///////////////////////////////////////
    Teuchos::RCP<EvaluatorTpetraConstraint> model = Teuchos::rcp(new EvaluatorTpetraConstraint(commRcp_, mobOpRcp_, conCollectorPtr_, ptcSystemPtr_, dt_));
    model->set_W_factory(lowsFactory_);

    //////////////////////////////
    // Create the JFNK operator //
    //////////////////////////////
    Teuchos::ParameterList printParams;
    Teuchos::RCP<Teuchos::ParameterList> jfnkParams = Teuchos::parameterList();
    jfnkParams->set("Difference Type", "Forward");
    jfnkParams->set("Perturbation Algorithm", "KSP NOX 2001");
    jfnkParams->set("lambda", 1.0e-4);
    Teuchos::RCP<NOX::Thyra::MatrixFreeJacobianOperator<Scalar> > jfnkOp =
        Teuchos::rcp(new NOX::Thyra::MatrixFreeJacobianOperator<Scalar>(printParams));
    jfnkOp->setParameterList(jfnkParams);

    // Wrap the model evaluator in a JFNK Model Evaluator
    Teuchos::RCP<Thyra::ModelEvaluator<Scalar> > thyraModel =
        Teuchos::rcp(new NOX::MatrixFreeModelEvaluatorDecorator<Scalar>(model));

    //////////////////////////////
    // Create the initial guess //
    //////////////////////////////
    Teuchos::RCP<Thyra::VectorBase<Scalar>> initial_guess = model->getNominalValues().get_x()->clone_v();

    Teuchos::RCP<NOX::Thyra::Group> noxGroup = Teuchos::rcp(new NOX::Thyra::Group(
        *initial_guess, model, model->create_W_op(), lowsFactory_, Teuchos::null, Teuchos::null, Teuchos::null));

    // model, model->create_W_op(),
    // thyraModel, jfnkOp,

    // noxGroup->computeF(); //TODO: is this necessary? No, I don't think so
    // noxGroup->computeJacobian(); //TODO: is this necessary? No, I know think so

    // VERY IMPORTANT!!!  jfnk object needs base evaluation objects.
    // This creates a circular dependency, so use a weak pointer.
    jfnkOp->setBaseEvaluationToNOXGroup(noxGroup.create_weak());

    ///////////////////////////////
    // Create and run the solver //
    ///////////////////////////////
    Teuchos::RCP<NOX::Solver::Generic> solver = NOX::Solver::buildSolver(noxGroup, statusTests_, nonlinearParams_);
    NOX::StatusTest::StatusType solvStatus = solver->solve();
    TEUCHOS_ASSERT(solvStatus == NOX::StatusTest::Converged);
    Teuchos::TimeMonitor::summarize();

    //////////////////////////////////
    // Fetch amd store the solution //
    //////////////////////////////////  
    const auto& final_x_nox = solver->getSolutionGroup().getX();
    const auto& final_x_thyra = dynamic_cast<const NOX::Thyra::Vector&>(final_x_nox).getThyraVector();
    gammaRcp_ = dynamic_cast<const ::Thyra::TpetraVector<Scalar, LO, GO, Node>&>(final_x_thyra).getConstTpetraVector();
}

void ConstraintSolver::writebackGamma() { conCollectorPtr_->writeBackGamma(gammaRcp_.getConst()); }