/**
 * @file SylinderNear.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Essential type for sylinder short range interactions
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef SylinderNear_HPP_
#define SylinderNear_HPP_

#include "Sylinder.hpp"

#include "Collision/DCPQuery.hpp"
#include "Constraint/ConstraintCollector.hpp"
#include "FDPS/particle_simulator.hpp"
#include "Util/EigenDef.hpp"
#include "Util/Logger.hpp"

#include <cassert>
#include <deque>
#include <limits>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>

/**
 *  Do not use PS::F64vec3 except in required function interfaces
 */

/**
 * @brief Essential type for sylinder short range interactions
 *
 * Essential Particle Class for FDPS
 */
struct SylinderNearEP {
  public:
    int gid;                            ///< global unique id
    int globalIndex;                    ///< sequentially ordered unique index in sylinder map
    int rank;                           ///< mpi rank of owning rank
    double radius;                      ///< radius
    double length;                      ///< length
    double radiusCollision;             ///< collision radius
    double lengthCollision;             ///< collision length
    double colBuf = GEO_DEFAULT_COLBUF; ///< collision search buffer

    double pos[3];       ///< position
    double direction[3]; ///< direction (unit norm vector)

    /**
     * @brief Get gid
     *
     * @return int
     */
    int getGid() const { return gid; }

    /**
     * @brief Get global index (sequentially ordered in sylinder map)
     *
     * @return int
     */
    int getGlobalIndex() const { return globalIndex; }

    /**
     * @brief copy data fields from full type Sylinder
     *
     * interface for FDPS
     * @param fp
     */
    void copyFromFP(const Sylinder &fp) {
        gid = fp.gid;
        globalIndex = fp.globalIndex;
        rank = fp.rank;
        colBuf = fp.colBuf;

        radius = fp.radius;
        length = fp.length;
        radiusCollision = fp.radiusCollision;
        lengthCollision = fp.lengthCollision;

        std::copy(fp.pos, fp.pos + 3, pos);
        Evec3 q = ECmapq(fp.orientation) * Evec3(0, 0, 1);
        direction[0] = q[0];
        direction[1] = q[1];
        direction[2] = q[2];
    }

    /**
     * @brief Get pos as a PS::F64vec3 object
     *
     * interface for FDPS
     * @return PS::F64vec
     */
    PS::F64vec getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }

    /**
     * @brief get search radius
     *
     * interface for FDPS
     * FDPS does not support search with rI+rJ.
     * Here length ensures contact is detected with Symmetry search mode
     * @return PS::F64
     */
    PS::F64 getRSearch() const {
        return std::max(length + 2 * radius, lengthCollision + 2 * lengthCollision) * (1 + colBuf);
    }

    /**
     * @brief Set pos with a PS::F64vec3 object
     *
     * interface for FDPS
     * @param newPos
     */
    void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }

    bool isSphere(bool collision = false) const {
        if (collision) {
            return lengthCollision < radiusCollision * 2;
        } else {
            return length < radius * 2;
        }
    }
};

static_assert(std::is_trivially_copyable<SylinderNearEP>::value, "");
static_assert(std::is_default_constructible<SylinderNearEP>::value, "");

/**
 * @brief collect short range interaction forces
 *
 */
class ForceNear {
  public:
    // double sepmin; ///< minimal separation
    double forceNear[3];
    double torqueNear[3];

    void clear() {
        std::fill(forceNear, forceNear + 3, 0);
        std::fill(torqueNear, torqueNear + 3, 0);
        // sepmin = std::numeric_limits<double>::max();
    }
};

static_assert(std::is_trivially_copyable<ForceNear>::value, "");
static_assert(std::is_default_constructible<ForceNear>::value, "");

/**
 * @brief callable object to collect collision blocks and compute near force
 *
 */
class CalcSylinderNearForce {

  public:
    std::shared_ptr<ConstraintPool> conPoolPtr;           ///< shared object for collecting constraints
    std::shared_ptr<ConstraintBlockPool> conBlockPoolPtr; ///< shared object for collecting constraint blocks (a.k.a constrained dof)

    /**
     * @brief Construct a new CalcSylinderNearForce object
     *
     */
    CalcSylinderNearForce() = default;

    /**
     * @brief Construct a new CalcSylinderNearForce object
     *
     * @param colPoolPtr_ the CollisionBlockPool object to write to
     */
    CalcSylinderNearForce(std::shared_ptr<ConstraintPool> &conPoolPtr_, 
                          std::shared_ptr<ConstraintBlockPool> &conBlockPoolPtr_) {
        spdlog::debug("stress recoder size: {}", conBlockPoolPtr_->size());

        conPoolPtr = conPoolPtr_;
        conBlockPoolPtr = conBlockPoolPtr_;
        assert(conPoolPtr);
        assert(conBlockPoolPtr);
    }

    /**
     * @brief interaction functor called by FDPS internally
     *   when a sylinder has length < diameter, it is treated as a sphere with radius_eff = radius + 0.5*length,
     *   i.e., a big sphere that completely encapsules this sylinder.
     *   collision stress is also calculated in this way
     * @param ep_i target
     * @param Nip number of target
     * @param ep_j source
     * @param Njp number of source
     * @param forceNear computed force
     */
    void operator()(const SylinderNearEP *const ep_i, const PS::S32 Nip, const SylinderNearEP *const ep_j,
                    const PS::S32 Njp, ForceNear *const forceNear) {
        const int myThreadId = omp_get_thread_num();
        auto &conQue = (*conPoolPtr)[myThreadId];
        auto &conBlockQue = (*conBlockPoolPtr)[myThreadId];

        for (PS::S32 i = 0; i < Nip; ++i) {
            auto &syI = ep_i[i];
            auto &forceI = forceNear[i];
            forceI.clear();

            if (isSphere(syI)) { // sphereI collisions
                for (int j = 0; j < Njp; j++) {
                    auto &syJ = ep_j[j];
                    if (syI.gid >= syJ.gid)
                        continue;
                    bool collision = false;
                    if (isSphere(syJ)) {
                        collision = sp_sp(syI, syJ, forceI, conQue, conBlockQue);
                    } else {
                        collision = sp_sy(syI, syJ, forceI, conQue, conBlockQue);
                    }
                }
            } else { // sylinderI collisions
                for (int j = 0; j < Njp; j++) {
                    auto &syJ = ep_j[j];
                    if (syI.gid >= syJ.gid)
                        continue;
                    bool collision = false;
                    if (isSphere(syJ)) {
                        collision = sp_sy(syJ, syI, forceI, conQue, conBlockQue, true);
                    } else {
                        collision = sy_sy(syI, syJ, forceI, conQue, conBlockQue);
                    }
                }
            }
        }
    }

    bool isSphere(const SylinderNearEP &sy) const { return sy.lengthCollision < 2 * sy.radiusCollision; }

    /**
     * @brief
     *
     * @param spI
     * @param spJ
     * @param forceI
     * @param conBlock
     * @return true
     * @return false
     */
    bool sp_sp(const SylinderNearEP &spI, const SylinderNearEP &spJ, ForceNear &forceI,
               ConstraintQue &conQue, ConstraintBlockQue &conBlockQue) const {
        // TODO: Switch all our pi's to M_PI
        const double pi = 3.141592653589793238;

        // sphere collide with sphere
        const Evec3 centerI = ECmap3(spI.pos);
        const Evec3 centerJ = ECmap3(spJ.pos);

        // effective radius of sphere = sp.lengthCollision*0.5 + sp.radiusCollision
        const double radI = spI.lengthCollision * 0.5 + spI.radiusCollision;
        const double radJ = spJ.lengthCollision * 0.5 + spJ.radiusCollision;

        const Evec3 rIJ = centerJ - centerI;
        const double rnorm = rIJ.norm();

        bool collision = false;

        const double sep = rnorm - (radI + radJ); // goal of constraint is sep >=0

        if (sep < (radI * spI.colBuf + radJ * spJ.colBuf)) {
            collision = true;
            std::unique_ptr<Collision> collisionCon = std::make_unique<Collision>();
            conQue.push_back(std::move(collisionCon));

            const Evec3 &Ploc = centerI;
            const Evec3 &Qloc = centerJ;
            const double gamma = sep < 0 ? -sep : 0;
            const Evec3 normI = (Ploc - Qloc).normalized();
            const Evec3 normJ = -normI;
            const Evec3 posI = Ploc - centerI;
            const Evec3 posJ = Qloc - centerJ;

            // Collisions generate three DOF, one for no-penetration and two for tangency
            // TODO: Make this tangent computation a component of the Sylinder class! Here, we should simply call a compute function
            // step 1: compute the tangent vectors to spI at posI 
            // step 1.1: convert posI into spherical polar coordinates
            //           default to using coordinates centered around the z-axis (chart A)
            //           if chart A is degenerage (has a tangent vector of near-zero length), 
            //           then use coordinates centered around the x-axis (chart B)
            // chart A
            double theta = std::atan2(std::sqrt(std::pow(posI[0], 2) + std::pow(posI[1], 2)), posI[2]);
            double phi = std::atan2(posI[1], posI[0]);

            // compute the tangent vectors in a non-degenerate chart
            Evec3 tangent1;
            Evec3 tangent2; 
            if ((theta < 0.8 * pi / 4.0 ) || (theta > 0.8 * pi)) {
                // chart A is potentially degenerate, use chart B
                theta = std::atan2(std::sqrt(std::pow(posI[1], 2) + std::pow(posI[2], 2)), posI[0]);
                phi = std::atan2(posI[1], posI[2]);

                // compute the tangent
                tangent1 = Evec3(-std::sin(theta), std::cos(theta) * std::sin(phi), std::cos(theta) * std::cos(phi)); 
                tangent2 = Evec3(0.0, std::sin(theta) * std::cos(phi), -std::sin(theta) * std::sin(phi)); 
            } else {
                // use chart A by default
                // compute the tangent
                tangent1 = Evec3(std::cos(theta) * std::cos(phi), std::cos(theta) * std::sin(phi), -std::sin(theta)); 
                tangent2 = Evec3(-std::sin(theta) * std::sin(phi), std::sin(theta) * std::cos(phi), 0.0); 
            }

            // fill the constraint dof
            { 
                // no-penetration constraint
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI = normI;
                const Evec3 unscaledForceComJ = normJ;
                const Evec3 unscaledTorqueComI(
                    normI[2] * posI[1] - normI[1] * posI[2],
                    normI[0] * posI[2] - normI[2] * posI[0],
                    normI[1] * posI[0] - normI[0] * posI[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * posJ[1] - normJ[1] * posJ[2],
                    normJ[0] * posJ[2] - normJ[2] * posJ[0],
                    normJ[1] * posJ[0] - normJ[0] * posJ[1]); 
                const double delta0 = sep;
                ConstraintBlock conBlock(delta0, gamma,             // current separation, initial guess of gamma
                                        spI.gid, spJ.gid,           //
                                        spI.globalIndex,            //
                                        spJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                Emat3 stressIJ = Emat3::Zero();
                collideStress(Evec3(0, 0, 1), Evec3(0, 0, 1), centerI, centerJ, 0, 0, // length = 0, degenerates to sphere
                            radI, radJ, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
            {
                // tangency constraint 1
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
                const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
                const Evec3 unscaledTorqueComI(
                    normI[2] * tangent1[1] - normI[1] * tangent1[2],
                    normI[0] * tangent1[2] - normI[2] * tangent1[0],
                    normI[1] * tangent1[0] - normI[0] * tangent1[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * tangent1[1] - normJ[1] * tangent1[2],
                    normJ[0] * tangent1[2] - normJ[2] * tangent1[0],
                    normJ[1] * tangent1[0] - normJ[0] * tangent1[1]);

                // current angle between normal and tangent1 is pi, keep it that way
                ConstraintBlock conBlock(0.0, gamma,                // desired change in angle, initial guess of gamma
                                        spI.gid, spJ.gid,           //
                                        spI.globalIndex,            //
                                        spJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                Emat3 stressIJ = Emat3::Zero();
                collideStress(Evec3(0, 0, 1), Evec3(0, 0, 1), centerI, centerJ, 0, 0, // length = 0, degenerates to sphere
                            radI, radJ, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
            {
                // tangency constraint 2
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
                const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
                const Evec3 unscaledTorqueComI(
                    normI[2] * tangent2[1] - normI[1] * tangent2[2],
                    normI[0] * tangent2[2] - normI[2] * tangent2[0],
                    normI[1] * tangent2[0] - normI[0] * tangent2[1]);      
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * tangent2[1] - normJ[1] * tangent2[2],
                    normJ[0] * tangent2[2] - normJ[2] * tangent2[0],
                    normJ[1] * tangent2[0] - normJ[0] * tangent2[1]);

                // current angle between normal and tangent1 is pi, keep it that way
                ConstraintBlock conBlock(0.0, gamma,                // desired change in angle, initial guess of gamma
                                        spI.gid, spJ.gid,           //
                                        spI.globalIndex,            //
                                        spJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                Emat3 stressIJ = Emat3::Zero();
                collideStress(Evec3(0, 0, 1), Evec3(0, 0, 1), centerI, centerJ, 0, 0, // length = 0, degenerates to sphere
                            radI, radJ, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
        }
        return collision;
    }

    /**
     * @brief
     *
     * @param sp the sylinder treated as a sphere
     * @param sy the sylinder
     * @param forceI
     * @param conBlock
     * @param reverseIJ default = false. if true, reverse I and J when adding the constraint block
     * @return true
     * @return false
     */
    bool sp_sy(const SylinderNearEP &spI, const SylinderNearEP &syJ, ForceNear &forceI, 
               ConstraintQue &conQue, ConstraintBlockQue &conBlockQue,
               bool reverseIJ = false) const {
        // TODO: Switch all our pi's to M_PI
        const double pi = 3.141592653589793238;

        // sphere collide with sylinder
        // effective radius of sphere = sp.lengthCollision*0.5 + sp.radiusCollision

        const Evec3 centerI = ECmap3(spI.pos);
        const double radI = spI.lengthCollision * 0.5 + spI.radiusCollision;

        const Evec3 centerJ = ECmap3(syJ.pos);

        const Evec3 directionJ = ECmap3(syJ.direction);
        const Evec3 Qm = centerJ - directionJ * (0.5 * syJ.lengthCollision); // minus end
        const Evec3 Qp = centerJ + directionJ * (0.5 * syJ.lengthCollision); // plus end

        Evec3 Ploc = centerI;
        Evec3 Qloc = Evec3::Zero();
        double distMin = DistPointSeg<Evec3>(centerI, Qm, Qp, Qloc);

        bool collision = false;

        const double sep = distMin - (radI + syJ.radiusCollision); // goal of constraint is sep >=0

        if (sep < (radI * spI.colBuf + syJ.radiusCollision * syJ.colBuf)) {
            collision = true;
            std::unique_ptr<Collision> collisionCon = std::make_unique<Collision>();
            // std::unique_ptr<NoPenetration> collisionCon = std::make_unique<NoPenetration>();
            conQue.push_back(std::move(collisionCon));

            const double delta0 = sep;
            const double gamma = sep < 0 ? -sep : 0;
            const Evec3 normI = (Ploc - Qloc).normalized();
            const Evec3 normJ = -normI;
            const Evec3 posI = Ploc - centerI;
            const Evec3 posJ = Qloc - centerJ;


            // Collisions generate three constraints, one for no-penetration and two for tangency
            // TODO: Make this computation a component of the Sylinder class! Here, we should simply call a compute function
            // step 1: compute the tangent vectors to spI at posI 
            // step 1.1: convert posI into spherical polar coordinates
            //           default to using coordinates centered around the z-axis (chart A)
            //           if chart A is degenerage (has a tangent vector of near-zero length), 
            //           then use coordinates centered around the x-axis (chart B)
            // chart A
            double theta = std::atan2(std::sqrt(std::pow(posI[0], 2) + std::pow(posI[1], 2)), posI[2]);
            double phi = std::atan2(posI[1], posI[0]);

            // compute the tangent vectors in a non-degenerate chart
            Evec3 tangent1;
            Evec3 tangent2; 
            if ((theta < 0.8 * pi / 4.0 ) || (theta > 0.8 * pi)) {
                // chart A is potentially degenerate, use chart B
                theta = std::atan2(std::sqrt(std::pow(posI[1], 2) + std::pow(posI[2], 2)), posI[0]);
                phi = std::atan2(posI[1], posI[2]);

                // compute the tangent (in the current config!)
                tangent1 = Evec3(-std::sin(theta), std::cos(theta) * std::sin(phi), std::cos(theta) * std::cos(phi)); 
                tangent2 = Evec3(0.0, std::sin(theta) * std::cos(phi), -std::sin(theta) * std::sin(phi)); 
            } else {
                // use chart A by default
                // compute the tangent (in the current config!)
                tangent1 = Evec3(std::cos(theta) * std::cos(phi), std::cos(theta) * std::sin(phi), -std::sin(theta)); 
                tangent2 = Evec3(-std::sin(theta) * std::sin(phi), std::sin(theta) * std::cos(phi), 0.0); 
            }

            // fill the constraints
            { 
                // no-penetration constraint
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI = normI;
                const Evec3 unscaledForceComJ = normJ;
                const Evec3 unscaledTorqueComI(
                    normI[2] * posI[1] - normI[1] * posI[2],
                    normI[0] * posI[2] - normI[2] * posI[0],
                    normI[1] * posI[0] - normI[0] * posI[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * posJ[1] - normJ[1] * posJ[2],
                    normJ[0] * posJ[2] - normJ[2] * posJ[0],
                    normJ[1] * posJ[0] - normJ[0] * posJ[1]); 
                ConstraintBlock conBlock(delta0, gamma,           // current separation, initial guess of gamma
                                        spI.gid, syJ.gid,           //
                                        spI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                if (reverseIJ) {
                    conBlock.reverseIJ();
                }
                Emat3 stressIJ = Emat3::Zero();
                collideStress(Evec3(0, 0, 1), directionJ, centerI, centerJ, 0, syJ.lengthCollision, radI,
                            syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
            {
                // tangency constraint 1
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
                const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
                const Evec3 unscaledTorqueComI(
                    normI[2] * tangent1[1] - normI[1] * tangent1[2],
                    normI[0] * tangent1[2] - normI[2] * tangent1[0],
                    normI[1] * tangent1[0] - normI[0] * tangent1[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * tangent1[1] - normJ[1] * tangent1[2],
                    normJ[0] * tangent1[2] - normJ[2] * tangent1[0],
                    normJ[1] * tangent1[0] - normJ[0] * tangent1[1]);
                // current angle between normal and tangent1 is pi, keep it that way
                ConstraintBlock conBlock(0.0, gamma,                // desired change in angle, initial guess of gamma
                                        spI.gid, syJ.gid,           //
                                        spI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                if (reverseIJ) {
                    conBlock.reverseIJ();
                }
                Emat3 stressIJ = Emat3::Zero();
                collideStress(Evec3(0, 0, 1), directionJ, centerI, centerJ, 0, syJ.lengthCollision, radI,
                            syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
            {
                // tangency constraint 2
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
                const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
                const Evec3 unscaledTorqueComI(
                    normI[2] * tangent2[1] - normI[1] * tangent2[2],
                    normI[0] * tangent2[2] - normI[2] * tangent2[0],
                    normI[1] * tangent2[0] - normI[0] * tangent2[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * tangent2[1] - normJ[1] * tangent2[2],
                    normJ[0] * tangent2[2] - normJ[2] * tangent2[0],
                    normJ[1] * tangent2[0] - normJ[0] * tangent2[1]);
                // current angle between normal and tangent1 is pi, keep it that way
                ConstraintBlock conBlock(0.0, gamma,                // desired change in angle, initial guess of gamma
                                        spI.gid, syJ.gid,           //
                                        spI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                if (reverseIJ) {
                    conBlock.reverseIJ();
                }
                Emat3 stressIJ = Emat3::Zero();
                collideStress(Evec3(0, 0, 1), directionJ, centerI, centerJ, 0, syJ.lengthCollision, radI,
                            syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
        }
        return collision;
    }

    /**
     * @brief
     *
     * @param syI
     * @param syJ
     * @param forceI
     * @param conBlock
     * @return true
     * @return false
     */
    bool sy_sy(const SylinderNearEP &syI, const SylinderNearEP &syJ, ForceNear &forceI,
               ConstraintQue &conQue, ConstraintBlockQue &conBlockQue) const {
        // TODO: Switch all our pi's to M_PI
        const double pi = 3.141592653589793238;

        // sylinder collide with sylinder
        DCPQuery<3, double, Evec3> DistSegSeg3;

        const Evec3 centerI = ECmap3(syI.pos);
        const Evec3 directionI = ECmap3(syI.direction);
        const Evec3 Pm = centerI - directionI * (0.5 * syI.lengthCollision); // minus end
        const Evec3 Pp = centerI + directionI * (0.5 * syI.lengthCollision); // plus end

        const Evec3 centerJ = ECmap3(syJ.pos);
        Evec3 Ploc = Evec3::Zero();
        Evec3 Qloc = Evec3::Zero();

        const Evec3 directionJ = ECmap3(syJ.direction);
        const Evec3 Qm = centerJ - directionJ * (0.5 * syJ.lengthCollision); // minus end
        const Evec3 Qp = centerJ + directionJ * (0.5 * syJ.lengthCollision); // plus end
        double s, t = 0;
        double distMin = DistSegSeg3(Pm, Pp, Qm, Qp, Ploc, Qloc, s, t);

        bool collision = false;

        const double sep = distMin - (syI.radiusCollision + syJ.radiusCollision); // goal of constraint is sep >=0

        if (sep < (syI.radiusCollision * syI.colBuf + syJ.radiusCollision * syJ.colBuf)) {
            collision = true;
            std::unique_ptr<Collision> collisionCon = std::make_unique<Collision>();
            // std::unique_ptr<NoPenetration> collisionCon = std::make_unique<NoPenetration>();
            conQue.push_back(std::move(collisionCon));

            const double delta0 = sep;
            const double gamma = sep < 0 ? -sep : 0;
            const Evec3 normI = (Ploc - Qloc).normalized();
            const Evec3 normJ = -normI;
            const Evec3 posI = Ploc - centerI;
            const Evec3 posJ = Qloc - centerJ;

            // Collisions generate three constraints, one for no-penetration and two for tangency
            // TODO: Make this computation a component of the Sylinder class! Here, we should simply call a compute function
            // step 1: compute the tangent vectors to spI at posI 
            // step 1.1: convert posI into spherical polar coordinates
            //           default to using coordinates centered around the z-axis (chart A)
            //           if chart A is degenerage (has a tangent vector of near-zero length), 
            //           then use coordinates centered around the x-axis (chart B)
            Equatn quatI = Equatn::FromTwoVectors(Evec3(0, 0, 1), directionI);
            const Evec3 normIRefConfig = quatI.inverse() * normI; 

            // compute the tangent vectors in a non-degenerate chart
            Evec3 tangent1;
            Evec3 tangent2; 

            // chart A
            double theta = std::atan2(std::sqrt(std::pow(normIRefConfig[0], 2) + std::pow(normIRefConfig[1], 2)), normIRefConfig[2]);
            double phi = std::atan2(normIRefConfig[1], normIRefConfig[0]);

            if ((theta < 0.8 * pi / 4.0 ) || (theta > 0.8 * pi)) {
                // chart A is potentially degenerate, use chart B
                theta = std::atan2(std::sqrt(std::pow(normIRefConfig[1], 2) + std::pow(normIRefConfig[2], 2)), normIRefConfig[0]);
                phi = std::atan2(normIRefConfig[1], normIRefConfig[2]);

                // compute the tangent (in the current config!)
                tangent1 = quatI * Evec3(-std::sin(theta), std::cos(theta) * std::sin(phi), std::cos(theta) * std::cos(phi)).normalized(); 
                tangent2 = quatI * Evec3(0.0, std::sin(theta) * std::cos(phi), -std::sin(theta) * std::sin(phi)).normalized(); 
            } else {
                // use chart A
                // compute the tangent (in the current config!)
                tangent1 = quatI * Evec3(std::cos(theta) * std::cos(phi), std::cos(theta) * std::sin(phi), -std::sin(theta)).normalized(); 
                tangent2 = quatI * Evec3(-std::sin(theta) * std::sin(phi), std::sin(theta) * std::cos(phi), 0.0).normalized(); 
            }

            // fill the constraints
            { 
                // no-penetration constraint
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI = normI;
                const Evec3 unscaledForceComJ = normJ;
                const Evec3 unscaledTorqueComI(
                    normI[2] * posI[1] - normI[1] * posI[2],
                    normI[0] * posI[2] - normI[2] * posI[0],
                    normI[1] * posI[0] - normI[0] * posI[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * posJ[1] - normJ[1] * posJ[2],
                    normJ[0] * posJ[2] - normJ[2] * posJ[0],
                    normJ[1] * posJ[0] - normJ[0] * posJ[1]); 
                ConstraintBlock conBlock(delta0, gamma,             // current separation, initial guess of gamma
                                        syI.gid, syJ.gid,           //
                                        syI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                Emat3 stressIJ = Emat3::Zero();
                collideStress(directionI, directionJ, centerI, centerJ, syI.lengthCollision, syJ.lengthCollision,
                            syI.radiusCollision, syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
            {
                // tangency constraint 1
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
                const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
                const Evec3 unscaledTorqueComI(
                    normI[2] * tangent1[1] - normI[1] * tangent1[2],
                    normI[0] * tangent1[2] - normI[2] * tangent1[0],
                    normI[1] * tangent1[0] - normI[0] * tangent1[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * tangent1[1] - normJ[1] * tangent1[2],
                    normJ[0] * tangent1[2] - normJ[2] * tangent1[0],
                    normJ[1] * tangent1[0] - normJ[0] * tangent1[1]);
                // current angle between normal and tangent1 is pi, keep it that way
                ConstraintBlock conBlock(0.0, gamma,                // desired change in angle, initial guess of gamma
                                        syI.gid, syJ.gid,           //
                                        syI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                Emat3 stressIJ = Emat3::Zero();
                collideStress(directionI, directionJ, centerI, centerJ, syI.lengthCollision, syJ.lengthCollision,
                            syI.radiusCollision, syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
            {
                // tangency constraint 2
                // unscaledForce = normal vector
                // unscaledTorque = relative position cross normal vector
                const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
                const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
                const Evec3 unscaledTorqueComI(
                    normI[2] * tangent2[1] - normI[1] * tangent2[2],
                    normI[0] * tangent2[2] - normI[2] * tangent2[0],
                    normI[1] * tangent2[0] - normI[0] * tangent2[1]);         
                const Evec3 unscaledTorqueComJ(
                    normJ[2] * tangent2[1] - normJ[1] * tangent2[2],
                    normJ[0] * tangent2[2] - normJ[2] * tangent2[0],
                    normJ[1] * tangent2[0] - normJ[0] * tangent2[1]);

                // current angle between normal and tangent1 is pi, keep it that way
                ConstraintBlock conBlock(0.0, gamma,                // desired change in angle, initial guess of gamma
                                        syI.gid, syJ.gid,           //
                                        syI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        unscaledForceComI.data(), unscaledForceComJ.data(),   // com force induced by this constraint for unit gamma
                                        unscaledTorqueComI.data(), unscaledTorqueComJ.data(), // com torque induced by this constraint for unit gamma
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, 0, 0.0);
                Emat3 stressIJ = Emat3::Zero();
                collideStress(directionI, directionJ, centerI, centerJ, syI.lengthCollision, syJ.lengthCollision,
                            syI.radiusCollision, syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                conBlock.setStress(stressIJ);
                conBlockQue.push_back(conBlock);
            }
        }
        return collision;
    }

    /**
     * @brief compute collision stress for a pair of sylinders
     *
     * @param dirI direction of I
     * @param dirJ direction of J
     * @param posI center location of I (lab frame)
     * @param posJ center location of J (lab frame)
     * @param hI length (cylindrical part) of I
     * @param hJ length (cylindrical part) of J
     * @param rI radius of I
     * @param rJ radius of J
     * @param rho mass density (set to 1)
     * @param Ploc location of force on I (lab frame)
     * @param Qloc location of force on J (lab frame)
     * @param StressIJ [out] pairwise stress if gamma = 1
     */
    static void collideStress(const Evec3 &dirI, const Evec3 &dirJ,                   //
                              const Evec3 &centerI, const Evec3 &centerJ,             //
                              double hI, double hJ, const double rI, const double rJ, //
                              const double rho,                                       //
                              const Evec3 &Ploc, const Evec3 &Qloc, Emat3 &StressIJ) {
        Emat3 NI, GAMMAI, NJ, GAMMAJ, InvGAMMAI, InvGAMMAJ;
        InitializeSyN(NI, rI, hI, rho);
        InitializeSyGA(GAMMAI, rI, hI, rho);
        InitializeSyN(NJ, rJ, hJ, rho);
        InitializeSyGA(GAMMAJ, rJ, hJ, rho);

        double aI = NI(0, 0), bI = NI(2, 2), aJ = NJ(0, 0), bJ = NJ(2, 2);

        NI = aI * Emat3::Identity() + (bI - aI) * (dirI * dirI.transpose());
        NJ = aJ * Emat3::Identity() + (bJ - aJ) * (dirJ * dirJ.transpose());

        aI = 1.0 / GAMMAI(0, 0);
        bI = 1.0 / GAMMAI(2, 2);
        aJ = 1.0 / GAMMAJ(0, 0);
        bJ = 1.0 / GAMMAJ(2, 2);

        InvGAMMAI = aI * Emat3::Identity() + (bI - aI) * (dirI * dirI.transpose());
        InvGAMMAJ = aJ * Emat3::Identity() + (bJ - aJ) * (dirJ * dirJ.transpose());

        Evec3 F1 = (Qloc - Ploc).normalized();
        Emat3 rIf = (centerI) * (-F1.transpose()); // Newton's law
        Emat3 rJf = (centerJ) * (F1.transpose());
        Evec3 xICf = (Ploc - centerI).cross(-F1); // Newton's law
        Evec3 xJCf = (Qloc - centerJ).cross(F1);

        // Levi-Civita symbol epsilon
        constexpr double epsilon[3][3][3] = {{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, //
                                             {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, //
                                             {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}};

        Emat3 SGI, SGJ;
        SGI.setZero();
        SGJ.setZero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        for (int r = 0; r < 3; r++) {
                            SGI(i, j) = SGI(i, j) + NI(i, l) * epsilon[j][k][l] * InvGAMMAI(k, r) * xICf(r);
                            SGJ(i, j) = SGJ(i, j) + NJ(i, l) * epsilon[j][k][l] * InvGAMMAJ(k, r) * xJCf(r);
                        }
                    }
                }
            }
        }

        StressIJ = rIf + rJf + SGI + SGJ;
    }

    /**
     * @brief initialize the tensor integral N for sylinder aligned with z axis
     *
     * @param NI [out] tensor integral N
     * @param r
     * @param h
     * @param rho
     */
    static void InitializeSyN(Emat3 &NI, double r, double h, double rho) {
        NI.setZero();
        double beta = h / 2.0 / r;
        NI(0, 0) = 1.0 / 30.0 * (15.0 * beta + 8);
        NI(1, 1) = NI(0, 0);
        NI(2, 2) = 1.0 / 15.0 * (10.0 * beta * beta * beta + 20.0 * beta * beta + 15.0 * beta + 4.0);
        NI = NI * rho * r * r * r * r * r * M_PI;
    }

    /**
     * @brief initialize the tensor integral GAMMAI for sylinder aligned with z axis
     *
     * @param GAMMAI
     * @param r
     * @param h
     * @param rho
     */
    static void InitializeSyGA(Emat3 &GAMMAI, double r, double h, double rho) {
        GAMMAI.setZero();
        double beta = h / 2.0 / r;
        GAMMAI(0, 0) = 1.0 / 30.0 * (20.0 * beta * beta * beta + 40.0 * beta * beta + 45.0 * beta + 16.0);
        GAMMAI(1, 1) = GAMMAI(0, 0);
        GAMMAI(2, 2) = 1.0 / 15.0 * (15 * beta + 8);
        GAMMAI = GAMMAI * M_PI * r * r * r * r * r * rho;
    }
};

/**
 * @brief tree type for computing near interaction of sylinders
 *
 */
using TreeSylinderNear = PS::TreeForForceShort<ForceNear, SylinderNearEP, SylinderNearEP>::Symmetry;

#endif
