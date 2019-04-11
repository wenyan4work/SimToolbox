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
 * The blocks are collected by CollisionCollector and then used to construct the sparse fcTrans matrix
 */
struct CollisionBlock {
  public:
    double phi0 = 0;                        ///< constraint initial value
    double gamma = 0;                       ///< force magnitude, could be an initial guess
    int gidI = 0, gidJ = 0;                 ///< global ID of the two colliding objects
    int globalIndexI = 0, globalIndexJ = 0; ///< the global index of the two objects in mobility matrix
    bool oneSide = false;                   ///< flag for one side collision. body J does not appear in mobility matrix
    double normI[3] = {0, 0, 0};
    double normJ[3] = {0, 0, 0}; ///< surface norm vector at the location of collision (minimal separation).
    double posI[3] = {0, 0, 0};
    double posJ[3] = {0, 0, 0}; ///< the collision position on bodies I and J. useless for spheres.
    double endI[3] = {0, 0, 0};
    double endJ[3] = {0, 0, 0}; ///< the labframe location of collision points endI and endJ
    double stress[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    ///< stress 3x3 matrix (row-major), to be scaled by solution gamma for the actual stress

    /**
     * @brief Construct a new empty collision block
     *
     */
    CollisionBlock() = default;

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
                   const Evec3 &normI_, const Evec3 &normJ_, const Evec3 &posI_, const Evec3 &posJ_, const Evec3 &endI_,
                   const Evec3 &endJ_, bool oneSide_ = false)
        : CollisionBlock(phi0_, gamma_, gidI_, gidJ_, globalIndexI_, globalIndexJ_, normI_.data(), normJ_.data(),
                         posI_.data(), posJ_.data(), endI_.data(), endJ_.data(), oneSide_) {}

    CollisionBlock(double phi0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
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
};

static_assert(std::is_trivially_copyable<CollisionBlock>::value, "");
static_assert(std::is_default_constructible<CollisionBlock>::value, "");

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
    void computeCollisionStress(Emat3 &stress, bool withOneSide = false) {
        const auto &colPool = *collisionPoolPtr;
        const int poolSize = colPool.size();
        std::vector<Emat3> stressPool(poolSize);
// reduction of stress
#pragma omp parallel for schedule(static, 1)
        for (int que = 0; que < poolSize; que++) {
            Emat3 stress = Emat3::Zero();
            for (const auto &colBlock : colPool[que]) {
                if (colBlock.oneSide && withOneSide) {
                    continue;
                }
                Emat3 stressColBlock;
                stressColBlock << colBlock.stress[0], colBlock.stress[1], colBlock.stress[2], //
                    colBlock.stress[3], colBlock.stress[4], colBlock.stress[5],               //
                    colBlock.stress[6], colBlock.stress[7], colBlock.stress[8];
                stress = stress + (stressColBlock * colBlock.gamma);
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

    /**
     * @brief write VTK XML PVTP Header file from rank 0
     *
     * @param prefix
     * @param postfix
     * @param nProcs
     */
    void writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs) {
        std::vector<std::string> pieceNames;

        std::vector<IOHelper::FieldVTU> pointDataFields;
        pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "gid");
        pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "globalIndex");
        pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "posIJ");
        pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "normIJ");

        std::vector<IOHelper::FieldVTU> cellDataFields;
        cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "oneSide");
        cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "phi0");
        cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "gamma");
        // cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "StressRowX");
        // cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "StressRowY");
        // cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "StressRowZ");
        cellDataFields.emplace_back(9, IOHelper::IOTYPE::Float32, "Stress");

        for (int i = 0; i < nProcs; i++) {
            pieceNames.emplace_back(std::string("ColBlock_") + std::string("r") + std::to_string(i) + "_" + postfix +
                                    ".vtp");
        }

        IOHelper::writePVTPFile(prefix + "ColBlock_" + postfix + ".pvtp", pointDataFields, cellDataFields, pieceNames);
    }

    /**
     * @brief write VTK XML binary base64 VTP data file from every MPI rank
     *
     * Procedure for dumping collision blocks in the system:
     * Each block writes a polyline with two (connected) points.
     * Points are labeled with float -1 and 1
     * Collision data fields are written as cell data
     * Rank 0 writes the parallel header , then each rank write its own serial vtp file
     *
     * @param prefix
     * @param postfix
     * @param rank
     */
    void writeVTP(const std::string &prefix, const std::string &postfix, int rank) {
        const auto &colPool = *collisionPoolPtr;
        const int nColQue = colPool.size();
        std::vector<int> colQueSize(nColQue, 0);
        std::vector<int> colQueIndex(nColQue + 1, 0);
        for (int i = 0; i < nColQue; i++) {
            colQueSize[i] = colPool[i].size();
        }
        for (int i = 1; i < nColQue + 1; i++) {
            colQueIndex[i] = colQueSize[i - 1] + colQueIndex[i - 1];
        }

        const int nColBlock = colQueIndex.back();

        // for each block:

        // write VTP for basic data
        //  use float to save some space
        // point and point data
        std::vector<double> pos(6 * nColBlock); // position always in Float64
        std::vector<int32_t> gid(2 * nColBlock);
        std::vector<int32_t> globalIndex(2 * nColBlock);
        std::vector<float> posIJ(6 * nColBlock);
        std::vector<float> normIJ(6 * nColBlock);

        // point connectivity of line
        std::vector<int32_t> connectivity(2 * nColBlock);
        std::vector<int32_t> offset(nColBlock);

        // cell data for ColBlock
        std::vector<int32_t> oneSide(nColBlock);
        std::vector<float> phi0(nColBlock);
        std::vector<float> gamma(nColBlock);
        // std::vector<float> StressRowX(3 * nColBlock);
        // std::vector<float> StressRowY(3 * nColBlock);
        // std::vector<float> StressRowZ(3 * nColBlock);
        std::vector<float> Stress(9 * nColBlock);

// connectivity
#pragma omp parallel for
        for (int i = 0; i < nColBlock; i++) {
            connectivity[2 * i] = 2 * i;         // index of point 0 in line
            connectivity[2 * i + 1] = 2 * i + 1; // index of point 1 in line
            offset[i] = 2 * i + 2;               // offset is the end of each line. in fortran indexing
        }

// data
#pragma omp parallel for
        for (int q = 0; q < nColQue; q++) {
            const int queSize = colQueSize[q];
            const int queIndex = colQueIndex[q];
            for (int c = 0; c < queSize; c++) {
                const int colIndex = queIndex + c;
                const auto &block = colPool[q][c];
                // point position
                pos[6 * colIndex + 0] = block.endI[0];
                pos[6 * colIndex + 1] = block.endI[1];
                pos[6 * colIndex + 2] = block.endI[2];
                pos[6 * colIndex + 3] = block.endJ[0];
                pos[6 * colIndex + 4] = block.endJ[1];
                pos[6 * colIndex + 5] = block.endJ[2];
                // point data
                gid[2 * colIndex + 0] = block.gidI;
                gid[2 * colIndex + 1] = block.gidJ;
                globalIndex[2 * colIndex + 0] = block.globalIndexI;
                globalIndex[2 * colIndex + 1] = block.globalIndexJ;
                posIJ[6 * colIndex + 0] = block.posI[0];
                posIJ[6 * colIndex + 1] = block.posI[1];
                posIJ[6 * colIndex + 2] = block.posI[2];
                posIJ[6 * colIndex + 3] = block.posJ[0];
                posIJ[6 * colIndex + 4] = block.posJ[1];
                posIJ[6 * colIndex + 5] = block.posJ[2];
                normIJ[6 * colIndex + 0] = block.normI[0];
                normIJ[6 * colIndex + 1] = block.normI[1];
                normIJ[6 * colIndex + 2] = block.normI[2];
                normIJ[6 * colIndex + 3] = block.normJ[0];
                normIJ[6 * colIndex + 4] = block.normJ[1];
                normIJ[6 * colIndex + 5] = block.normJ[2];
                // cell data
                oneSide[colIndex] = block.oneSide ? 1 : 0;
                phi0[colIndex] = block.phi0;
                gamma[colIndex] = block.gamma;
                // StressRowX[3 * colIndex + 0] = block.stress[0];
                // StressRowX[3 * colIndex + 1] = block.stress[1];
                // StressRowX[3 * colIndex + 2] = block.stress[2];
                // StressRowY[3 * colIndex + 0] = block.stress[3];
                // StressRowY[3 * colIndex + 1] = block.stress[4];
                // StressRowY[3 * colIndex + 2] = block.stress[5];
                // StressRowZ[3 * colIndex + 0] = block.stress[6];
                // StressRowZ[3 * colIndex + 1] = block.stress[7];
                // StressRowZ[3 * colIndex + 2] = block.stress[8];
                for (int kk = 0; kk < 9; kk++) {
                    Stress[9 * colIndex + kk] = block.stress[kk] * block.gamma;
                }
            }
        }

        std::ofstream file(prefix + std::string("ColBlock_") + "r" + std::to_string(rank) + std::string("_") + postfix +
                               std::string(".vtp"),
                           std::ios::out);

        IOHelper::writeHeadVTP(file);

        file << "<Piece NumberOfPoints=\"" << nColBlock * 2 << "\" NumberOfLines=\"" << nColBlock << "\">\n";
        // Points
        file << "<Points>\n";
        IOHelper::writeDataArrayBase64(pos, "position", 3, file);
        file << "</Points>\n";
        // cell definition
        file << "<Lines>\n";
        IOHelper::writeDataArrayBase64(connectivity, "connectivity", 1, file);
        IOHelper::writeDataArrayBase64(offset, "offsets", 1, file);
        file << "</Lines>\n";
        // point data
        file << "<PointData Scalars=\"scalars\">\n";
        IOHelper::writeDataArrayBase64(gid, "gid", 1, file);
        IOHelper::writeDataArrayBase64(globalIndex, "globalIndex", 1, file);
        IOHelper::writeDataArrayBase64(posIJ, "posIJ", 3, file);
        IOHelper::writeDataArrayBase64(normIJ, "normIJ", 3, file);
        file << "</PointData>\n";
        // cell data
        file << "<CellData Scalars=\"scalars\">\n";
        IOHelper::writeDataArrayBase64(oneSide, "oneSide", 1, file);
        IOHelper::writeDataArrayBase64(phi0, "phi0", 1, file);
        IOHelper::writeDataArrayBase64(gamma, "gamma", 1, file);
        IOHelper::writeDataArrayBase64(Stress, "Stress", 9, file);
        // IOHelper::writeDataArrayBase64(StressRowX, "StressRowX", 3, file);
        // IOHelper::writeDataArrayBase64(StressRowY, "StressRowY", 3, file);
        // IOHelper::writeDataArrayBase64(StressRowZ, "StressRowZ", 3, file);
        file << "</CellData>\n";
        file << "</Piece>\n";

        IOHelper::writeTailVTP(file);
        file.close();
    }
};

#endif