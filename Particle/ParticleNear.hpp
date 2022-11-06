/**
 * @file ParticleNear.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Essential type for particle short range interactions
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef ParticleNear_HPP_
#define ParticleNear_HPP_

#include "Particle.hpp"

#include "Constraint/ConstraintCollector.hpp"
#include "Constraint/DCPQuery.hpp"
#include "FDPS/particle_simulator.hpp"
#include "Util/EigenDef.hpp"
#include "Util/GeoUtil.hpp"
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
 * @brief Essential type for particle short range interactions
 *
 * Essential Particle Class for FDPS
 */
struct ParticleNearEP {
  public:
    int gid;                            ///< global unique id
    int globalIndex;                    ///< sequentially ordered unique index in particle map
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
     * @brief Get global index (sequentially ordered in particle map)
     *
     * @return int
     */
    int getGlobalIndex() const { return globalIndex; }

    /**
     * @brief copy data fields from full type Particle
     *
     * interface for FDPS
     * @param fp
     */
    void copyFromFP(const Particle &fp) {
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
     * FDPS gives the following discription of Symmetry search mode:
     *      This type is used when its force decays to zero at a finite distance and when the distance is
     *      determined by the larger of the sizes of i- and j-particles.
     *
     * The member variable search radius is a radius of a particle. Neighbor particles of the
     * particle are defined as those inside the particle.
     * @return PS::F64
     */
    PS::F64 getRSearch() const {
        const double boundingSphereRad =  0.5 * std::max(length + 2 * radius, lengthCollision + 2 * radiusCollision);
        const double searchRadius = boundingSphereRad + colBuf;
        return searchRadius;
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

static_assert(std::is_trivially_copyable<ParticleNearEP>::value, "");
static_assert(std::is_default_constructible<ParticleNearEP>::value, "");

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
class CalcParticleNearForce {

  public:
    std::shared_ptr<ConstraintPool> conPoolPtr; ///< shared object for collecting constraints
    bool simBoxPBC[3];
    double simBoxLow[3];
    double simBoxHigh[3];

    /**
     * @brief Construct a new CalcParticleNearForce object
     *
     */
    CalcParticleNearForce() = default;

    /**
     * @brief Construct a new CalcParticleNearForce object
     *
     * @param colPoolPtr_ the CollisionBlockPool object to write to
     */
    CalcParticleNearForce(std::shared_ptr<ConstraintPool> &conPoolPtr_, const bool *simBoxPBC_,
                          const double *simBoxLow_, const double *simBoxHigh_) {
        spdlog::debug("stress recoder size: {}", conPoolPtr_->size());

        for (int i = 0; i < 3; i++) {
            simBoxPBC[i] = simBoxPBC_[i];
            simBoxLow[i] = simBoxLow_[i];
            simBoxHigh[i] = simBoxHigh_[i];
        }
        conPoolPtr = conPoolPtr_;
        assert(conPoolPtr);
    }

    /**
     * @brief interaction functor called by FDPS internally
     *   when a particle has length < diameter, it is treated as a sphere with radius_eff = radius + 0.5*length,
     *   i.e., a big sphere that completely encapsules this particle.
     *   collision stress is also calculated in this way
     * @param ep_i target
     * @param Nip number of target
     * @param ep_j source
     * @param Njp number of source
     * @param forceNear computed force
     */
    void operator()(const ParticleNearEP *const ep_i, const PS::S32 Nip, const ParticleNearEP *const ep_j,
                    const PS::S32 Njp, ForceNear *const forceNear) {
        const int myThreadId = omp_get_thread_num();
        auto &conQue = (*conPoolPtr)[myThreadId];

        for (PS::S32 i = 0; i < Nip; ++i) {
            auto &ptcI = ep_i[i];
            auto &forceI = forceNear[i];
            forceI.clear();

            if (isSphere(ptcI)) { // sphereI collisions
                for (int j = 0; j < Njp; j++) {
                    auto &ptcJ = ep_j[j];
                    if (ptcI.gid >= ptcJ.gid)
                        continue;
                    Constraint con;
                    bool collision = false;
                    if (isSphere(ptcJ)) {
                        collision = sp_sp(ptcI, ptcJ, forceI, con, false);
                    } else {
                        collision = sp_sy(ptcI, ptcJ, forceI, con, false);
                    }
                    if (collision)
                        conQue.push_back(con);
                }
            } else { // particleI collisions
                for (int j = 0; j < Njp; j++) {
                    auto &ptcJ = ep_j[j];
                    if (ptcI.gid >= ptcJ.gid)
                        continue;
                    Constraint con;
                    bool collision = false;
                    if (isSphere(ptcJ)) {
                        collision = sp_sy(ptcJ, ptcI, forceI, con, false, true);
                    } else {
                        collision = sy_sy(ptcI, ptcJ, forceI, con, false);
                    }
                    if (collision)
                        conQue.push_back(con);
                }
            }
        }
    }

    /**
     * @brief Update a collision constraint between two known ParticleNearEPs
     *   when a particle has length < diameter, it is treated as a sphere with radius_eff = radius + 0.5*length,
     *   i.e., a big sphere that completely encapsules this particle!
     *   collision stress is also calculated in this way
     * @param ptcI target
     * @param ptcJ source
     * @param con block to update
     */
    void updateCollisionBlock(const ParticleNearEP ptcI, const ParticleNearEP ptcJ, Constraint &con) {
        ForceNear forceI;
        if (isSphere(ptcI)) { // sphereI collisions
            if (isSphere(ptcJ)) {
                sp_sp(ptcI, ptcJ, forceI, con, true);
            } else {
                sp_sy(ptcI, ptcJ, forceI, con, true);
            }
        } else { // particleI collisions
            if (isSphere(ptcJ)) {
                sp_sy(ptcJ, ptcI, forceI, con, true, true);
            } else {
                sy_sy(ptcI, ptcJ, forceI, con, true);
            }
        }
    }

    bool isSphere(const ParticleNearEP &ptc) const { return ptc.lengthCollision < 2 * ptc.radiusCollision; }

    /**
     * @brief
     *
     * @param spI
     * @param spJ
     * @param forceI
     * @param con
     * @return true
     * @return false
     */
    bool sp_sp(const ParticleNearEP &spI, const ParticleNearEP &spJ, ForceNear &forceI, Constraint &con,
               bool update = false) const {
        // sphere collide with sphere

        // apply PBC on centerJ
        const Evec3 centerI = ECmap3(spI.pos);
        Evec3 centerJ = ECmap3(spJ.pos);
        for (int k = 0; k < 3; k++) {
            if (!simBoxPBC[k])
                continue;
            double trg = centerI[k];
            double xk = centerJ[k];
            findPBCImage(simBoxLow[k], simBoxHigh[k], xk, trg);
            centerJ[k] = xk;
            // error check
            if (fabs(trg - xk) > 0.5 * (simBoxHigh[k] - simBoxLow[k])) {
                spdlog::critical("pbc image error in near collision");
                std::exit(1);
            }
        }

        // effective radius of sphere = sp.lengthCollision*0.5 + sp.radiusCollision
        const double radI = spI.lengthCollision * 0.5 + spI.radiusCollision;
        const double radJ = spJ.lengthCollision * 0.5 + spJ.radiusCollision;

        const Evec3 rIJ = centerJ - centerI;
        const double rnorm = rIJ.norm();

        bool collision = false;
        const double sep = rnorm - (radI + radJ); // goal of collision constraint is sep >=0

        const double buffer = std::max(spI.colBuf, spJ.colBuf);
        if ((sep < buffer) || (update)) {
            collision = true;
            const Evec3 &Ploc = centerI;
            const Evec3 &Qloc = centerJ;
            const Evec3 normI = (Ploc - Qloc).normalized();
            const Evec3 normJ = -normI;
            const Evec3 posI = Ploc - centerI;
            const Evec3 posJ = Qloc - centerJ;

            Emat3 stressIJ = Emat3::Zero();
            collideStress(Evec3(0, 0, 1), Evec3(0, 0, 1), centerI, centerJ, 0,
                          0, // length = 0, degenerates to sphere
                          radI, radJ, 1.0, Ploc, Qloc, stressIJ);

            // fill the constraint
            noPenetrationConstraint(con,                      // constraint object,
                                    sep,                      // amount of overlap,
                                    spI.gid, spJ.gid,         //
                                    spI.globalIndex,          //
                                    spJ.globalIndex,          //
                                    posI.data(), posJ.data(), // location of collision relative to particle center
                                    Ploc.data(), Qloc.data(), // location of collision in lab frame
                                    normI.data(),             // direction of collision force
                                    stressIJ.data(), false);
        }
        return collision;
    }

    /**
     * @brief
     *
     * @param sp the particle treated as a sphere
     * @param ptc the particle
     * @param forceI
     * @param con
     * @param reverseIJ default = false. if true, reverse I and J when adding the constraint block
     * @return true
     * @return false
     */
    bool sp_sy(const ParticleNearEP &spI, const ParticleNearEP &ptcJ, ForceNear &forceI, Constraint &con,
               bool update = false, const bool reverseIJ = false) const {
        // sphere collide with particle

        // apply PBC on centerJ
        const Evec3 centerI = ECmap3(spI.pos);
        Evec3 centerJ = ECmap3(ptcJ.pos);
        for (int k = 0; k < 3; k++) {
            if (!simBoxPBC[k])
                continue;
            double trg = centerI[k];
            double xk = centerJ[k];
            findPBCImage(simBoxLow[k], simBoxHigh[k], xk, trg);
            centerJ[k] = xk;
            // error check
            if (fabs(trg - xk) > 0.5 * (simBoxHigh[k] - simBoxLow[k])) {
                spdlog::critical("pbc image error in near collision");
                std::exit(1);
            }
        }

        // effective radius of sphere = sp.lengthCollision*0.5 + sp.radiusCollision
        const double radI = spI.lengthCollision * 0.5 + spI.radiusCollision;
        const Evec3 directionJ = ECmap3(ptcJ.direction);
        const Evec3 Qm = centerJ - directionJ * (0.5 * ptcJ.lengthCollision); // minus end
        const Evec3 Qp = centerJ + directionJ * (0.5 * ptcJ.lengthCollision); // plus end

        Evec3 Ploc = centerI;
        Evec3 Qloc = Evec3::Zero();
        double distMin = DistPointSeg<Evec3>(centerI, Qm, Qp, Qloc);

        bool collision = false;
        const double sep = distMin - (radI + ptcJ.radiusCollision); // goal of constraint is sep >=0

        const double buffer = std::max(spI.colBuf, ptcJ.colBuf);
        if ((sep < buffer) || (update)) {
            collision = true;
            const Evec3 normI = (Ploc - Qloc).normalized();
            const Evec3 normJ = -normI;
            const Evec3 posI = Ploc - centerI;
            const Evec3 posJ = Qloc - centerJ;

            Emat3 stressIJ = Emat3::Zero();
            collideStress(Evec3(0, 0, 1), directionJ, centerI, centerJ, 0, ptcJ.lengthCollision, radI,
                          ptcJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);

            // fill the constraint
            noPenetrationConstraint(con,                      // constraint object,
                                    sep,                      // amount of overlap,
                                    spI.gid, ptcJ.gid,         //
                                    spI.globalIndex,          //
                                    ptcJ.globalIndex,          //
                                    posI.data(), posJ.data(), // location of collision relative to particle center
                                    Ploc.data(), Qloc.data(), // location of collision in lab frame
                                    normI.data(),             // direction of collision force
                                    stressIJ.data(), false);
            if (reverseIJ) {
                con.reverseIJ();
            }
        }
        return collision;
    }

    /**
     * @brief
     *
     * @param ptcI
     * @param ptcJ
     * @param forceI
     * @param con
     * @return true
     * @return false
     */
    bool sy_sy(const ParticleNearEP &ptcI, const ParticleNearEP &ptcJ, ForceNear &forceI, Constraint &con,
               bool update = false) const {
        // particle collide with particle
        DCPQuery<3, double, Evec3> DistSegSeg3;

        // apply PBC on centerJ
        const Evec3 centerI = ECmap3(ptcI.pos);
        Evec3 centerJ = ECmap3(ptcJ.pos);
        for (int k = 0; k < 3; k++) {
            if (!simBoxPBC[k])
                continue;
            double trg = centerI[k];
            double xk = centerJ[k];
            findPBCImage(simBoxLow[k], simBoxHigh[k], xk, trg);
            centerJ[k] = xk;
            // error check
            if (fabs(trg - xk) > 0.5 * (simBoxHigh[k] - simBoxLow[k])) {
                spdlog::critical("pbc image error in near collision");
                std::exit(1);
            }
        }

        // compute the separation
        const Evec3 directionI = ECmap3(ptcI.direction);
        const Evec3 Pm = centerI - directionI * (0.5 * ptcI.lengthCollision); // minus end
        const Evec3 Pp = centerI + directionI * (0.5 * ptcI.lengthCollision); // plus end

        Evec3 Ploc = Evec3::Zero();
        Evec3 Qloc = Evec3::Zero();

        const Evec3 directionJ = ECmap3(ptcJ.direction);
        const Evec3 Qm = centerJ - directionJ * (0.5 * ptcJ.lengthCollision); // minus end
        const Evec3 Qp = centerJ + directionJ * (0.5 * ptcJ.lengthCollision); // plus end
        double s, t = 0;
        double distMin = DistSegSeg3(Pm, Pp, Qm, Qp, Ploc, Qloc, s, t);

        bool collision = false;
        const double sep = distMin - (ptcI.radiusCollision + ptcJ.radiusCollision); // goal of constraint is sep >=0

        const double buffer = std::max(ptcI.colBuf, ptcJ.colBuf);
        if ((sep < buffer) || (update)) {
            collision = true;
            const Evec3 normI = (Ploc - Qloc).normalized();
            const Evec3 normJ = -normI;
            const Evec3 posI = Ploc - centerI;
            const Evec3 posJ = Qloc - centerJ;

            Emat3 stressIJ = Emat3::Zero();
            collideStress(directionI, directionJ, centerI, centerJ, ptcI.lengthCollision, ptcJ.lengthCollision,
                          ptcI.radiusCollision, ptcJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);

            // fill the constraint
            noPenetrationConstraint(con,                      // constraint object
                                    sep,                      // amount of overlap,
                                    ptcI.gid, ptcJ.gid,         //
                                    ptcI.globalIndex,          //
                                    ptcJ.globalIndex,          //
                                    posI.data(), posJ.data(), // location of collision relative to particle center
                                    Ploc.data(), Qloc.data(), // location of collision in lab frame
                                    normI.data(),             // direction of collision force
                                    stressIJ.data(), false);
        }
        return collision;
    }

    /**
     * @brief compute collision stress for a pair of particles
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
     * @brief initialize the tensor integral N for particle aligned with z axis
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
     * @brief initialize the tensor integral GAMMAI for particle aligned with z axis
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
 * @brief tree type for computing near interaction of particles
 * 
 * FDPS gives the following discription of Symmetry search mode:
 *      This type is used when its force decays to zero at a finite distance, and when the distance is
 *      determined by the larger of the sizes of i- and j-particles.
 *
 */
using TreeParticleNear = PS::TreeForForceShort<ForceNear, ParticleNearEP, ParticleNearEP>::Symmetry;

#endif