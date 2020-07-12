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
    int gid;                ///< global unique id
    int globalIndex;        ///< sequentially ordered unique index in sylinder map
    int rank;               ///< mpi rank of owning rank
    double radius;          ///< radius
    double radiusCollision; ///< collision radius
    double length;          ///< length
    double lengthCollision; ///< collision length
    Link link;              ///< linkage of this sylinder

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
    template<class T>
    void copyFromFP(T &fp) {
        gid = fp.gid;
        globalIndex = fp.globalIndex;
        rank = fp.rank;

        radius = fp.radius;
        radiusCollision = fp.radiusCollision;
        length = fp.length;
        lengthCollision = fp.lengthCollision;

        link = fp.link;

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
     * Here length*2 ensures contact is detected with Symmetry search mode
     * @return PS::F64
     */
    PS::F64 getRSearch() const { return (length + 2 * radiusCollision) * (1 + 0.5 * GEO_DEFAULT_COLBUF); }

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
};

static_assert(std::is_trivially_copyable<SylinderNearEP>::value, "");
static_assert(std::is_default_constructible<SylinderNearEP>::value, "");

/**
 * @brief collect short range interaction forces
 *
 */
class ForceNear {
  public:
    double sepmin; ///< minimal separation
    double forceNear[3];
    double torqueNear[3];

    void clear() {
        std::fill(forceNear, forceNear + 3, 0);
        std::fill(torqueNear, torqueNear + 3, 0);
        sepmin = std::numeric_limits<double>::max();
    }
};

static_assert(std::is_trivially_copyable<ForceNear>::value, "");
static_assert(std::is_default_constructible<ForceNear>::value, "");

/**
 * @brief callable object to collect collision blocks and compute near force
 *
 */
class CalcSylinderNearForce {
    double colbuf;

  public:
    std::shared_ptr<ConstraintBlockPool> conPoolPtr; ///< shared object for collecting collision constraints

    /**
     * @brief Construct a new CalcSylinderNearForce object
     *
     */
    CalcSylinderNearForce(const double colbuf_ = GEO_DEFAULT_COLBUF) : colbuf(colbuf_) {}

    /**
     * @brief Construct a new CalcSylinderNearForce object
     *
     * @param colPoolPtr_ the CollisionBlockPool object to write to
     */
    CalcSylinderNearForce(std::shared_ptr<ConstraintBlockPool> &conPoolPtr_, const double colbuf_ = GEO_DEFAULT_COLBUF)
        : colbuf(colbuf_) {
#ifdef DEBUGSYLINDERNEAR
        std::cout << "stress recoder size:" << colPoolPtr_->size() << std::endl;
#endif

        conPoolPtr = conPoolPtr_;
        assert(conPoolPtr);
    }

    /**
     * @brief interaction functor called by FDPS internally
     *
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
        DCPQuery<3, double, Evec3> DistSegSeg3;

        for (PS::S32 i = 0; i < Nip; ++i) {
            auto &forceI = forceNear[i];
            forceI.clear();

            auto &syI = ep_i[i];
            const Evec3 centerI = ECmap3(syI.pos);
            const Evec3 directionI = ECmap3(syI.direction);
            const Evec3 Pm = centerI - directionI * (0.5 * syI.lengthCollision); // minus end
            const Evec3 Pp = centerI + directionI * (0.5 * syI.lengthCollision); // plus end

            for (PS::S32 j = 0; j < Njp; ++j) {
                auto &syJ = ep_j[j];
                const Evec3 centerJ = ECmap3(syJ.pos);
                const Evec3 directionJ = ECmap3(syJ.direction);
                const Evec3 Qm = centerJ - directionJ * (0.5 * syJ.lengthCollision); // minus end
                const Evec3 Qp = centerJ + directionJ * (0.5 * syJ.lengthCollision); // plus end

                Evec3 Ploc = Evec3::Zero();
                Evec3 Qloc = Evec3::Zero();
                double s, t = 0;
                const double distMin = DistSegSeg3(Pm, Pp, Qm, Qp, Ploc, Qloc, s, t);
                forceI.sepmin = std::min(forceI.sepmin, distMin);

                // add to collision processing list
                // save collision block
                // save only block gidI < gidJ, and gid
                const double sep = distMin - (syI.radiusCollision + syJ.radiusCollision); // target is sep >=0

                // record collision blocks
                if (sep < colbuf * (syI.radiusCollision + syJ.radiusCollision) && syI.gid < syJ.gid) {
                    const double delta0 = sep;
                    const double gamma = sep < 0 ? -sep : 0;
                    const Evec3 normI = (Ploc - Qloc).normalized();
                    const Evec3 normJ = -normI;
                    const Evec3 posI = Ploc - centerI;
                    const Evec3 posJ = Qloc - centerJ;
                    conQue.emplace_back(delta0, gamma,              // current separation, initial guess of gamma
                                        syI.gid, syJ.gid,           //
                                        syI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        normI.data(), normJ.data(), // direction of collision force
                                        posI.data(), posJ.data(),   // location of collision relative to particle center
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, false, 0.0, 0.0);
                    Emat3 stressIJ;
                    collideStress(directionI, directionJ, centerI, centerJ, syI.lengthCollision, syJ.lengthCollision,
                                  syI.radiusCollision, syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                    conQue.back().setStress(stressIJ);
                }

                // record bilateral blocks
                if (syI.link.next == syJ.gid) {
                    const double delta0 = distMin - (syI.radiusCollision + syJ.radiusCollision) * 1.01;
                    const double gamma = 0;
                    const Evec3 Ploc = Pp;
                    const Evec3 Qloc = Qm;
                    const Evec3 normI = (Ploc - Qloc).normalized();
                    const Evec3 normJ = -normI;
                    const Evec3 posI = Ploc - centerI;
                    const Evec3 posJ = Qloc - centerJ;
                    conQue.emplace_back(delta0, gamma,              // current separation, initial guess of gamma
                                        syI.gid, syJ.gid,           //
                                        syI.globalIndex,            //
                                        syJ.globalIndex,            //
                                        normI.data(), normJ.data(), // direction of collision force
                                        posI.data(), posJ.data(),   // location of collision relative to particle center
                                        Ploc.data(), Qloc.data(),   // location of collision in lab frame
                                        false, true, -1.0, 0.0);    // kappa will be set to link kappa later
                    Emat3 stressIJ;
                    collideStress(directionI, directionJ, centerI, centerJ, syI.lengthCollision, syJ.lengthCollision,
                                  syI.radiusCollision, syJ.radiusCollision, 1.0, Ploc, Qloc, stressIJ);
                    conQue.back().setStress(stressIJ);
                }
            }
        }
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
