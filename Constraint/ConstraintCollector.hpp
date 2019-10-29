/**
 * @file ConstraintCollector.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Collect all constraints
 * @version 0.1
 * @date 2018-12-14
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef CONSTRAINTCOLLECTOR_HPP_
#define CONSTRAINTCOLLECTOR_HPP_

#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"
#include "Util/IOHelper.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <type_traits>
#include <vector>

#include <omp.h>

/**
 * @brief collision constraint information block
 *
 * Each block stores the information for one collision constraint.
 * The blocks are collected by ConstraintCollector and then used to construct the sparse fcTrans matrix
 */
struct ConstraintBlock {
  public:
    double phi0 = 0;                        ///< constraint initial value
    double gamma = 0;                       ///< force magnitude, could be an initial guess
    int gidI = 0, gidJ = 0;                 ///< global ID of the two constrained objects
    int globalIndexI = 0, globalIndexJ = 0; ///< the global index of the two objects in mobility matrix
    bool oneSide = false;                   ///< flag for one side constraint. body J does not appear in mobility matrix
    double kappa = 0;                       ///< spring constant. =0 means no spring
    double normI[3] = {0, 0, 0};
    double normJ[3] = {0, 0, 0}; ///< surface norm vector at the location of constraints (minimal separation).
    double posI[3] = {0, 0, 0};
    double posJ[3] = {0, 0, 0}; ///< the relative constraint position on bodies I and J.
    double endI[3] = {0, 0, 0};
    double endJ[3] = {0, 0, 0}; ///< the labframe location of collision points endI and endJ
    double stress[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    ///< stress 3x3 matrix (row-major) for unit constraint force gamma

    /**
     * @brief Construct a new empty collision block
     *
     */
    ConstraintBlock() = default;

    /**
     * @brief Construct a new Constraint Block object
     *
     * @param phi0_ current value of the constraint
     * @param gamma_ initial guess of constraint force magnitude
     * @param gidI_
     * @param gidJ_
     * @param globalIndexI_
     * @param globalIndexJ_
     * @param normI_
     * @param normJ_
     * @param posI_
     * @param posJ_
     * @param oneSide_ flag for one side collision
     *
     * If oneside = true, the gidJ, globalIndexJ, normJ, posJ will be ignored when constructing the fcTrans matrix
     * so any value of gidJ, globalIndexJ, normJ, posJ can be used in that case.
     *
     */
    ConstraintBlock(double phi0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
                    const Evec3 &normI_, const Evec3 &normJ_, const Evec3 &posI_, const Evec3 &posJ_,
                    const Evec3 &endI_, const Evec3 &endJ_, bool oneSide_ = false)
        : ConstraintBlock(phi0_, gamma_, gidI_, gidJ_, globalIndexI_, globalIndexJ_, normI_.data(), normJ_.data(),
                          posI_.data(), posJ_.data(), endI_.data(), endJ_.data(), oneSide_) {}

    ConstraintBlock(double phi0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
                    const double normI_[3], const double normJ_[3], const double posI_[3], const double posJ_[3],
                    const double endI_[3], const double endJ_[3], bool oneSide_ = false)
        : phi0(phi0_), gamma(gamma_), gidI(gidI_), gidJ(gidJ_), globalIndexI(globalIndexI_),
          globalIndexJ(globalIndexJ_), oneSide(oneSide_) {
        for (int d = 0; d < 3; d++) {
            normI[d] = normI_[d];
            normJ[d] = normJ_[d];
            posI[d] = posI_[d];
            posJ[d] = posJ_[d];
            endI[d] = endI_[d];
            endJ[d] = endJ_[d];
        }
        std::fill(stress, stress + 9, 0);
    }

    void setStress(const Emat3 &stress_) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                stress[i * 3 + j] = stress_(i, j);
            }
        }
    }

    void setStress(const double *stress_) {
        for (int i = 0; i < 9; i++) {
            stress[i] = stress_[i];
        }
    }

    const double *getStress() const { return stress; }

    void getStress(Emat3 &stress_) const {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                stress_(i, j) = stress[i * 3 + j];
            }
        }
    }
};

static_assert(std::is_trivially_copyable<ConstraintBlock>::value, "");
static_assert(std::is_default_constructible<ConstraintBlock>::value, "");

using ConstraintBlockQue = std::vector<ConstraintBlock>;     ///< a queue contains blocks collected by one thread
using ConstraintBlockPool = std::vector<ConstraintBlockQue>; ///< a pool contains queues on different threads

/**
 * @brief collecter of collision blocks
 *
 */
class ConstraintCollector {
  public:
    std::shared_ptr<ConstraintBlockPool> constraintPoolPtr; /// all copy of collector share a pointer to collision pool

    /**
     * @brief Construct a new Constraint Collector object
     *
     */
    ConstraintCollector();

    /**
     * @brief Construct a new Constraint Collector object
     *
     * @param obj
     */
    ConstraintCollector(const ConstraintCollector &obj) = default;
    ConstraintCollector(ConstraintCollector &&obj) = default;
    ConstraintCollector &operator=(const ConstraintCollector &obj) = default;
    ConstraintCollector &operator=(ConstraintCollector &&obj) = default;
    ~ConstraintCollector() = default;

    /**
     * @brief if the shared pointer to collision pool is not allocated
     *
     * After the constructor this should always be false
     * @return true
     * @return false
     */
    bool empty() const { return constraintPoolPtr->empty(); }

    /**
     * @brief clear the blocks ang get an empty collision pool
     *
     * The collision pool still contains (the number of openmp threads) queues
     *
     */
    void clear() {
        assert(constraintPoolPtr);
        for (int i = 0; i < constraintPoolPtr->size(); i++) {
            (*constraintPoolPtr)[i].clear();
        }
    }

    /**
     * @brief get the number of collision constraints on the local node
     *
     * @return int
     */
    int getLocalNumberOfConstraints() {
        int sum = 0;
        for (int i = 0; i < constraintPoolPtr->size(); i++) {
            sum += (*constraintPoolPtr)[i].size();
        }
        return sum;
    }

    /**
     * @brief compute the total collision stress of all constraints (blocks)
     *
     * @param stress the sum of all stress blocks for all threads on the local rank
     * @param withOneSide include the stress (without proper definition) of one side collisions
     */
    void sumLocalConstraintStress(Emat3 &stress, bool withOneSide = false) const;

    /**
     * @brief write VTK XML PVTP Header file from rank 0
     *
     * the files will be written as folder/prefixConBlock_rX_postfix.vtp
     * @param folder
     * @param prefix
     * @param postfix
     * @param nProcs
     */
    void writePVTP(const std::string &folder, const std::string &prefix, const std::string &postfix,
                   const int nProcs) const;

    /**
     * @brief write VTK XML binary base64 VTP data file from every MPI rank
     *
     * files are written as
     * folder + '/' + prefix + ("ColBlock_") + "r" + (rank) + ("_") + postfix + (".vtp")
     *
     * Procedure for dumping constraint blocks in the system:
     * Each block writes a polyline with two (connected) points.
     * Points are labeled with float -1 and 1
     * Constraint data fields are written as point and cell data
     * Rank 0 writes the parallel header , then each rank write its own serial vtp file
     *
     * @param folder
     * @param prefix
     * @param postfix
     * @param rank
     */
    void writeVTP(const std::string &folder, const std::string &prefix, const std::string &postfix, int rank) const;

    Teuchos::RCP<TCMAT> assembleConstraintMatrix() const;
};

#endif