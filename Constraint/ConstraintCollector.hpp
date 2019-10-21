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
    ConstraintCollector() {
        const int totalThreads = omp_get_max_threads();
        constraintPoolPtr = std::make_shared<ConstraintBlockPool>();
        constraintPoolPtr->resize(totalThreads);
        for (auto &queue : *constraintPoolPtr) {
            queue.resize(0);
            queue.reserve(50);
        }
        std::cout << "ConstraintCollector constructed for:" << constraintPoolPtr->size() << " threads" << std::endl;
    }

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
    void computeTotalConstraintStress(Emat3 &stress, bool withOneSide = false) {
        const auto &cPool = *constraintPoolPtr;
        const int poolSize = cPool.size();

        // std::vector<Emat3> stressPool(poolSize);
        Emat3 stressTotal = Emat3::Zero();

#pragma omp parallel for
        for (int que = 0; que < poolSize; que++) {
            // reduction of stress, one que for each thread

            Emat3 stressSumQue = Emat3::Zero();
            for (const auto &cBlock : cPool[que]) {
                if (cBlock.oneSide && !withOneSide) {
                    // skip counting oneside collision blocks
                    continue;
                } else if (cBlock.gamma > 0) {
                    Emat3 stressBlock;
                    cBlock.getStress(stressBlock);
                    stressSumQue = stressSumQue + (stressBlock * cBlock.gamma);
                }
            }
#pragma omp critical
            { stressTotal = stressTotal + stressSumQue; }
        }

        // std::cout << stressTotal << std::endl;
    }

    /**
     * @brief write VTK XML PVTP Header file from rank 0
     *
     * the files will be written as folder/prefixConBlock_rX_postfix.vtp
     * @param folder
     * @param prefix
     * @param postfix
     * @param nProcs
     */
    void writePVTP(const std::string &folder, const std::string &prefix, const std::string &postfix, const int nProcs) {
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
        cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "kappa");
        cellDataFields.emplace_back(9, IOHelper::IOTYPE::Float32, "Stress");

        for (int i = 0; i < nProcs; i++) {
            pieceNames.emplace_back(prefix + std::string("ConBlock_") + std::string("r") + std::to_string(i) + "_" +
                                    postfix + ".vtp");
        }

        IOHelper::writePVTPFile(folder + "/" + prefix + std::string("ConBlock_") + postfix + ".pvtp", pointDataFields,
                                cellDataFields, pieceNames);
    }

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
    void writeVTP(const std::string &folder, const std::string &prefix, const std::string &postfix, int rank) {
        const auto &cPool = *constraintPoolPtr;
        const int cQueNum = cPool.size();
        std::vector<int> cQueSize(cQueNum, 0);
        std::vector<int> cQueIndex(cQueNum + 1, 0);
        for (int i = 0; i < cQueNum; i++) {
            cQueSize[i] = cPool[i].size();
        }
        for (int i = 1; i < cQueNum + 1; i++) {
            cQueIndex[i] = cQueSize[i - 1] + cQueIndex[i - 1];
        }

        const int cBlockNum = cQueIndex.back();

        // for each block:

        // write VTP for basic data, use float to save some space
        // point and point data
        std::vector<double> pos(6 * cBlockNum); // position always in Float64
        std::vector<int32_t> gid(2 * cBlockNum);
        std::vector<int32_t> globalIndex(2 * cBlockNum);
        std::vector<float> posIJ(6 * cBlockNum);
        std::vector<float> normIJ(6 * cBlockNum);

        // point connectivity of line
        std::vector<int32_t> connectivity(2 * cBlockNum);
        std::vector<int32_t> offset(cBlockNum);

        // cell data for ColBlock
        std::vector<int32_t> oneSide(cBlockNum);
        std::vector<float> phi0(cBlockNum);
        std::vector<float> gamma(cBlockNum);
        std::vector<float> kappa(cBlockNum);
        std::vector<float> Stress(9 * cBlockNum);

#pragma omp parallel for
        // connectivity
        for (int i = 0; i < cBlockNum; i++) {
            connectivity[2 * i] = 2 * i;         // index of point 0 in line
            connectivity[2 * i + 1] = 2 * i + 1; // index of point 1 in line
            offset[i] = 2 * i + 2;               // offset is the end of each line. in fortran indexing
        }

#pragma omp parallel for
        // data
        for (int q = 0; q < cQueNum; q++) {
            const int queSize = cQueSize[q];
            const int queIndex = cQueIndex[q];
            for (int c = 0; c < queSize; c++) {
                const int cIndex = queIndex + c;
                const auto &block = cPool[q][c];
                // point position
                pos[6 * cIndex + 0] = block.endI[0];
                pos[6 * cIndex + 1] = block.endI[1];
                pos[6 * cIndex + 2] = block.endI[2];
                pos[6 * cIndex + 3] = block.endJ[0];
                pos[6 * cIndex + 4] = block.endJ[1];
                pos[6 * cIndex + 5] = block.endJ[2];
                // point data
                gid[2 * cIndex + 0] = block.gidI;
                gid[2 * cIndex + 1] = block.gidJ;
                globalIndex[2 * cIndex + 0] = block.globalIndexI;
                globalIndex[2 * cIndex + 1] = block.globalIndexJ;
                posIJ[6 * cIndex + 0] = block.posI[0];
                posIJ[6 * cIndex + 1] = block.posI[1];
                posIJ[6 * cIndex + 2] = block.posI[2];
                posIJ[6 * cIndex + 3] = block.posJ[0];
                posIJ[6 * cIndex + 4] = block.posJ[1];
                posIJ[6 * cIndex + 5] = block.posJ[2];
                normIJ[6 * cIndex + 0] = block.normI[0];
                normIJ[6 * cIndex + 1] = block.normI[1];
                normIJ[6 * cIndex + 2] = block.normI[2];
                normIJ[6 * cIndex + 3] = block.normJ[0];
                normIJ[6 * cIndex + 4] = block.normJ[1];
                normIJ[6 * cIndex + 5] = block.normJ[2];
                // cell data
                oneSide[cIndex] = block.oneSide ? 1 : 0;
                phi0[cIndex] = block.phi0;
                gamma[cIndex] = block.gamma;
                kappa[cIndex] = block.kappa;
                for (int kk = 0; kk < 9; kk++) {
                    Stress[9 * cIndex + kk] = block.stress[kk] * block.gamma;
                }
            }
        }

        std::ofstream file(folder + '/' + prefix + std::string("ColBlock_") + "r" + std::to_string(rank) +
                               std::string("_") + postfix + std::string(".vtp"),
                           std::ios::out);

        IOHelper::writeHeadVTP(file);

        // Piece starts
        file << "<Piece NumberOfPoints=\"" << cBlockNum * 2 << "\" NumberOfLines=\"" << cBlockNum << "\">\n";
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
        IOHelper::writeDataArrayBase64(kappa, "kappa", 1, file);
        IOHelper::writeDataArrayBase64(Stress, "Stress", 9, file);
        file << "</CellData>\n";
        // Piece ends
        file << "</Piece>\n";

        IOHelper::writeTailVTP(file);
        file.close();
    }
};

#endif