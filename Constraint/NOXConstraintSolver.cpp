#include "NOXConstraintSolver.hpp"
#include "Util/Logger.hpp"

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
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set<int>("Maximum Iterations", 1000);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set<int>("Num Blocks", 200);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set<int>("Maximum Restarts", 100);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Verbosity", 0x7f);
    belosList.sublist("Solver Types").sublist("Pseudo Block GMRES").set("Output Frequency", 100);
    belosList.sublist("VerboseObject").set("Verbosity Level", "medium");
    params->set("Preconditioner Type", "None");

    Stratimikos::DefaultLinearSolverBuilder builder;
    builder.setParameterList(params);
    lowsFactory_ = builder.createLinearSolveStrategy("");

    /////////////////////////////////
    // Create the NOX status tests //
    /////////////////////////////////
    // TODO: play around with these params
    Teuchos::RCP<NOX::StatusTest::NormF> absresid = Teuchos::rcp(new NOX::StatusTest::NormF(1.0e-8));
    Teuchos::RCP<NOX::StatusTest::NormWRMS> wrms = Teuchos::rcp(new NOX::StatusTest::NormWRMS(1.0e-2, 1.0e-8));
    Teuchos::RCP<NOX::StatusTest::Combo> converged =
        Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::AND));
    converged->addStatusTest(absresid);
    converged->addStatusTest(wrms);
    Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters = Teuchos::rcp(new NOX::StatusTest::MaxIters(20));
    Teuchos::RCP<NOX::StatusTest::FiniteValue> fv = Teuchos::rcp(new NOX::StatusTest::FiniteValue);
    
    statusTests_ = Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));
    statusTests_->addStatusTest(fv);
    statusTests_->addStatusTest(converged);
    statusTests_->addStatusTest(maxiters);

    ///////////////////////////////
    // Create nox parameter list //
    ///////////////////////////////
    nonlinearParams_ = Teuchos::parameterList();
    nonlinearParams_->set("Nonlinear Solver", "Line Search Based");
    nonlinearParams_->sublist("Direction").sublist("Newton").sublist("Linear Solver").set("Tolerance", 1.0e-4);

    // Set output parameters
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

// TODO: give ConstraintSolver COMM access

void ConstraintSolver::solveConstraints() {
    ///////////////////////////////////////
    // Create the model evaluator object //
    ///////////////////////////////////////
    Teuchos::RCP<EvaluatorTpetraConstraint> model = Teuchos::rcp(new EvaluatorTpetraConstraint(commRcp_, mobOpRcp_, conCollectorPtr_, ptcSystemPtr_, dt_));
    model->set_W_factory(lowsFactory_);

    //////////////////////////////
    // Create the initial guess //
    //////////////////////////////
    Teuchos::RCP<Thyra::VectorBase<Scalar>> initial_guess = model->getNominalValues().get_x()->clone_v();
    Teuchos::RCP<NOX::Thyra::Group> noxGroup = Teuchos::rcp(new NOX::Thyra::Group(
        *initial_guess, model, model->create_W_op(), lowsFactory_, Teuchos::null, Teuchos::null, Teuchos::null));

    // noxGroup->computeF(); //TODO: is this necessary?
    // noxGroup->computeJacobian(); //TODO: is this necessary?


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
