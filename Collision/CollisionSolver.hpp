/**
 * @file CollisionSolver.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Solve the collision LCP problem
 * @version 0.1
 * @date 2018-12-14
 * 
 * @copyright Copyright (c) 2018
 * 
 */
#ifndef COLLISIONSOLVER_HPP_
#define COLLISIONSOLVER_HPP_

#include "CPSolver.hpp"
#include "CollisionCollector.hpp"

#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

class CollisionSolver {

  public:
    // constructor
    CollisionSolver() = default;
    ~CollisionSolver() = default;

    // forbid copy
    CollisionSolver(const CollisionSolver &) = delete;
    CollisionSolver(CollisionSolver &&) = delete;
    const CollisionSolver &operator=(const CollisionSolver &) = delete;
    const CollisionSolver &operator=(CollisionSolver &&) = delete;

    /**
     * @brief reset the parameters and release all allocated spaces
     *
     */
    void reset() {
        res = 1e-5;
        maxIte = 2000;
        newton = false;

        objMobMapRcp.reset();   // = Teuchos::null;
        forceColRcp.reset();    // force vec, 6 dof per obj
        velocityColRcp.reset(); // velocity vec, 6 dof per obj. velocity = mobity * forceCol

        gammaMapRcp.reset(); // distributed map for collision magnitude gamma
        gammaRcp.reset();    // the unknown

        phi0Rcp.reset();
        vnRcp.reset();
        bRcp.reset(); // the constant piece of LCP problem

        matMobilityRcp.reset(); // mobility operator, 6 dof per obj to 6 dof per obj
        matFcTransRcp.reset();  // FcTrans matrix, 6 dof per obj to gamma dof

        queueThreadIndex.clear();
    }

    /**
     * @brief return if no collision constraints are recorded on the local mpi rank
     *
     * @param colPool local CollisionPool object
     * @return true local pool is empty
     * @return false
     */
    bool collisionLocalIsEmpty(const CollisionBlockPool &colPool) const {
        if (colPool.empty()) {
            return true;
        } else {
            bool emptyFlag = true;
            for (auto &c : colPool) {
                emptyFlag = emptyFlag && c.empty();
            }
            return emptyFlag;
        }
    }

    /**
     * @brief set the control parameters res, maxIte, and newton refinement
     *
     * @param res_ iteration residual
     * @param maxIte_ max iterations
     * @param newton_ newton refinement flag
     */
    void setControlLCP(double res_, int maxIte_, bool newton_) {
        res = res_;
        maxIte = maxIte_;
        newton = newton_; // run mmNewton following the GD to refine the solution to 100x smaller residue
    }

    /**
     * @brief setup the solver object to prepare for solution
     *
     * @param collision_ the CollisionBlockPool object for local collision constraints
     * @param objMobMapRcp_ the map for object mobility
     * @param dt_ timestep dt
     * @param bufferGap_ overlap up to bufferGap is allowed
     */
    void setup(CollisionBlockPool &collision_, Teuchos::RCP<TMAP> &objMobMapRcp_, double dt_, double bufferGap_ = 0);

    /**
     * @brief solve the collision LCP problem
     *
     * @param matMobilityRcp_ mobility matrix
     * @param velocityKnownRcp_ known velocity besides collision
     */
    void solveCollision(Teuchos::RCP<TOP> &matMobilityRcp_, Teuchos::RCP<TV> &velocityKnownRcp_);

    /**
     * @brief write the solution collision force magnitude back to the ColliisonBlockPool
     *
     */
    void writebackGamma(CollisionBlockPool &collision_);

    // TODO: implement this in the future
    // void solveCollisionSplit(Teuchos::RCP<TOP> &matMobilityMajorRcp, Teuchos::RCP<TOP> &matMobilityOtherRcp);

    /**
     * @brief Get the Force Col object
     *
     * @return Teuchos::RCP<TV>
     */
    Teuchos::RCP<TV> getForceCol() const { return forceColRcp; }
    /**
     * @brief Get the Velocity Col object
     *
     * @return Teuchos::RCP<TV>
     */
    Teuchos::RCP<TV> getVelocityCol() const { return velocityColRcp; }
    /**
     * @brief Get the Force Col Magnitude object
     *
     * @return Teuchos::RCP<TV>
     */
    Teuchos::RCP<TV> getForceColMagnitude() const { return gammaRcp; }
    /**
     * @brief Get the Velocity Known object
     *
     * @return Teuchos::RCP<TV>
     */
    Teuchos::RCP<TV> getVelocityKnown() const { return vnRcp; }
    /**
     * @brief Get the Phi0 object
     *
     * @return Teuchos::RCP<TV>
     */
    Teuchos::RCP<TV> getPhi0() const { return phi0Rcp; }

  private:
    double res;  ///< residual tolerance
    int maxIte;  ///< max iterations
    bool newton; ///< flag for using newton refinement

    // mobility
    Teuchos::RCP<TMAP> objMobMapRcp; ///< distributed map for obj mobility. 6 dof per obj
    Teuchos::RCP<TV> forceColRcp;    ///< force vec, 6 dof per obj
    Teuchos::RCP<TV> velocityColRcp; ///< velocity vec, 6 dof per obj. velocityCol = mobity * forceCol

    // unknown collision force magnitude
    Teuchos::RCP<TMAP> gammaMapRcp; ///< distributed map for collision force magnitude gamma
    Teuchos::RCP<TV> gammaRcp;      ///< the unknown collision force magnitude gamma

    // known initial value of constraints
    Teuchos::RCP<TV> phi0Rcp; ///< the current minimal separation vector \f$\Phi_0\f$
    Teuchos::RCP<TV> vnRcp;   ///< the vector for velocity known
    Teuchos::RCP<TV> bRcp;    ///< the constant part of LCP problem

    // Mobility operator and FcTrans matrices
    Teuchos::RCP<TOP> matMobilityRcp;  ///< mobility operator, 6 dof per obj to 6 dof per obj
    Teuchos::RCP<TCMAT> matFcTransRcp; ///< FcTrans matrix, 6 dof per obj to gamma dof

    std::vector<int> queueThreadIndex;

    void dumpCollision(CollisionBlockPool &collision_) const;
    void setupCollisionBlockQueThreadIndex(CollisionBlockPool &collision_);

    /**
     * @brief setup the FcTrans matrix
     *
     * @param collision_
     */
    void setupFcTrans(CollisionBlockPool &collision_);
    /**
     * @brief setup the \f$\Phi_0\f$ vector
     *
     * @param collision_
     * @param dt_
     * @param bufferGap_
     */
    void setupPhi0Vec(CollisionBlockPool &collision_, double dt_, double bufferGap_);
    /**
     * @brief setup the initial guess of gamma vector
     *
     * @param collision_
     */
    void setupGammaVec(CollisionBlockPool &collision_);
    /**
     * @brief setup the known velocity vector
     *
     * @param collision_
     * @param velocity_
     */
    void setupVnVec(CollisionBlockPool &collision_, std::vector<double> &velocity_);

    /**
     * @brief setup the constant \f$b\f$ vector in LCP
     *
     */
    void setupBVec();

    /**
     * @brief Get the number of negative entries in a Tpetra Vector
     *
     * @param vecRcp_
     * @return int
     */
    int getNumNegElements(Teuchos::RCP<TV> &vecRcp_) const;
};

#endif