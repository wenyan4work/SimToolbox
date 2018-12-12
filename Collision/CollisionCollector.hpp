#ifndef COLLISIONCOLLECTOR_HPP_
#define COLLISIONCOLLECTOR_HPP_

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

#include <omp.h>

#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"

/**
 * @brief collision constraint information block
 *
 * Each block stores the information for one collision constraint.
 * The blocks are collected by CollisionCollector and then used to construct the sparse fcTrans matrix
 */
struct CollisionBlock {
  public:
    double phi0;                    /// constraint initial value
    double gamma;                   /// force magnitude, could be an initial guess
    int gidI, gidJ;                 /// global ID of the two colliding objects
    int globalIndexI, globalIndexJ; /// the global index of the two objects in mobility matrix
    bool oneSide = false; /// flag for one side collision, where one body of the pair does not appear in mobility matrix
    Evec3 normI, normJ; /// surface norm vector at the location of minimal separation for each particle. normJ = - normI
    Evec3 posI, posJ;   /// the collision position on bodies I and J. useless for spheres.
    Emat3 stress;       /// stress 3x3 matrix, to be scaled by solution gamma for the actual stress

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

using CollisionBlockQue = std::vector<CollisionBlock>; // can be changed to other containers, e.g., deque
using CollisionBlockPool = std::vector<CollisionBlockQue>;

// process each collision i,j and record them to ColBlocks
// interface operator(obj i, obj j)
class CollisionCollector {
  public:
    std::shared_ptr<CollisionBlockPool> collisionPoolPtr;

    // constructor
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

    // copy constructor
    CollisionCollector(const CollisionCollector &obj) = default;
    CollisionCollector(CollisionCollector &&obj) = default;
    CollisionCollector &operator=(const CollisionCollector &obj) = default;
    CollisionCollector &operator=(CollisionCollector &&obj) = default;
    ~CollisionCollector() = default;

    bool empty() const {
        // after the constructor this should always be false
        return collisionPoolPtr->empty();
    }

    void clear() {
        assert(collisionPoolPtr);
        for (int i = 0; i < collisionPoolPtr->size(); i++) {
            (*collisionPoolPtr)[i].clear();
        }
    }

    int getLocalCollisionNumber() {
        int sum = 0;
        for (int i = 0; i < collisionPoolPtr->size(); i++) {
            sum += (*collisionPoolPtr)[i].size();
        }
        return sum;
    }

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