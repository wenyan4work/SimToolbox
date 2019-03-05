/**
 * @file CollisionCollector.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Collect collision constraints
 * @version 0.1
 * @date 2018-12-14
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef COLLISIONCOLLECTOR_HPP_
#define COLLISIONCOLLECTOR_HPP_

#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>
#include <memory>

#include <omp.h>

/**
 * @brief collision constraint information block
 *
 * Each block stores the information for one collision constraint.
 * The blocks are collected by CollisionCollector and then used to construct the sparse fcTrans matrix
 */
struct CollisionBlock {
  public:
    double phi0;                    ///< constraint initial value
    double gamma;                   ///< force magnitude, could be an initial guess
    int gidI, gidJ;                 ///< global ID of the two colliding objects
    int globalIndexI, globalIndexJ; ///< the global index of the two objects in mobility matrix
    bool oneSide = false;           ///< flag for one side collision. body J does not appear in mobility matrix
    Evec3 normI, normJ;             ///< surface norm vector at the location of minimal separation.
    Evec3 posI, posJ;               ///< the collision position on bodies I and J. useless for spheres.
    Emat3 stress;                   ///< stress 3x3 matrix, to be scaled by solution gamma for the actual stress

    /**
     * @brief Construct a new empty collision block
     *
     */
    CollisionBlock() : gidI(0), gidJ(0), globalIndexI(0), globalIndexJ(0), phi0(0), gamma(0) {
        // default constructor
        normI.setZero();
        normJ.setZero();
        posI.setZero();
        posJ.setZero();
        oneSide = false;
        stress.setZero();
    }

    /**
     * @brief Construct a new Collision Block object
     *
     * @param phi0_ current value of the constraint (current minimal separation)
     * @param gamma_ initial guess of collision force magnitude
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
    CollisionBlock(double phi0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
                   const Evec3 &normI_, const Evec3 &normJ_, const Evec3 &posI_, const Evec3 &posJ_,
                   bool oneSide_ = false)
        : phi0(phi0_), gamma(gamma_), gidI(gidI_), gidJ(gidJ_), globalIndexI(globalIndexI_),
          globalIndexJ(globalIndexJ_), normI(normI_), normJ(normJ_), posI(posI_), posJ(posJ_), oneSide(oneSide_) {
        stress.setZero();
    }
};

using CollisionBlockQue = std::vector<CollisionBlock>;     ///< a queue contains blocks collected by one thread
using CollisionBlockPool = std::vector<CollisionBlockQue>; ///< a pool contains queues on different threads

/**
 * @brief collecter of collision blocks
 *
 */
class CollisionCollector {
  public:
    std::shared_ptr<CollisionBlockPool> collisionPoolPtr; /// all copy of collector share a pointer to collision pool

    /**
     * @brief Construct a new Collision Collector object
     *
     */
    CollisionCollector() {
        const int totalThreads = omp_get_max_threads();
        collisionPoolPtr = std::make_shared<CollisionBlockPool>();
        collisionPoolPtr->resize(totalThreads);
        for (auto &queue : *collisionPoolPtr) {
            queue.resize(0);
            queue.reserve(50);
        }
        std::cout << "stress recoder size:" << collisionPoolPtr->size() << std::endl;
    }

    /**
     * @brief Construct a new Collision Collector object
     *
     * @param obj
     */
    CollisionCollector(const CollisionCollector &obj) = default;
    CollisionCollector(CollisionCollector &&obj) = default;
    CollisionCollector &operator=(const CollisionCollector &obj) = default;
    CollisionCollector &operator=(CollisionCollector &&obj) = default;
    ~CollisionCollector() = default;

    /**
     * @brief if the shared pointer to collision pool is not allocated
     *
     * After the constructor this should always be false
     * @return true
     * @return false
     */
    bool empty() const { return collisionPoolPtr->empty(); }

    /**
     * @brief clear the blocks ang get an empty collision pool
     *
     * The collision pool still contains (the number of openmp threads) queues
     *
     */
    void clear() {
        assert(collisionPoolPtr);
        for (int i = 0; i < collisionPoolPtr->size(); i++) {
            (*collisionPoolPtr)[i].clear();
        }
    }

    /**
     * @brief get the number of collision constraints on the local node
     *
     * @return int
     */
    int getLocalCollisionNumber() {
        int sum = 0;
        for (int i = 0; i < collisionPoolPtr->size(); i++) {
            sum += (*collisionPoolPtr)[i].size();
        }
        return sum;
    }

    /**
     * @brief compute the average of collision stress of all constraints (blocks)
     *
     * @param stress the result
     */
    void computeCollisionStress(Emat3 &stress) {
        const auto &colPool = *collisionPoolPtr;
        const int poolSize = colPool.size();
        std::vector<Emat3> stressPool(poolSize);
// reduction of stress
#pragma omp parallel for schedule(static, 1)
        for (int que = 0; que < poolSize; que++) {
            Emat3 stress = Emat3::Zero();
            for (const auto &col : colPool[que]) {
                stress = stress + (col.stress * col.gamma);
            }
            stressPool[que] = stress * (1.0 / colPool.size());
        }

        stress = Emat3::Zero();
        for (int i = 0; i < poolSize; i++) {
            stress = stress + stressPool[i];
        }
        stress = stress * (1.0 / poolSize);
    }

    /**
     * @brief process each collision i,j and record them to ColBlocks
     *
     * @tparam Trg
     * @tparam Src
     * @param trg
     * @param src
     * @param srcShift shift of Src position. used in periodic boundary condition.
     */
    template <class Trg, class Src>
    void operator()(Trg &trg, const Src &src, const std::array<double, 3> &srcShift) {
        const int threadId = omp_get_thread_num();
        auto &colque = (*collisionPoolPtr)[threadId];
        // construct a collision block to threadId
        CollisionBlock block;
        bool collide = trg.collide(src, block, srcShift);
        if (collide) {
            colque.push_back(block);
        }
    }
};

#endif