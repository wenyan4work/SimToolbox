#ifndef COLLISIONCOLLECTOR_HPP
#define COLLISIONCOLLECTOR_HPP

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

#include <omp.h>

#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"
#include "Util/IOHelper.hpp"

struct CollisionBlock { // the information for each collision
  public:
    double phi0;  // constraint value
    double gamma; // force magnitude , could be an initial guess
    int gidI, gidJ;
    int globalIndexI, globalIndexJ;
    bool oneSide = false; // one side collision, e.g. moving obj collide with a boundary, and the boundary does not
                          // appear in the mobility matrix
    Evec3 normI, normJ;   // norm vector for each particle. gvecJ = - gvecI
    Evec3 posI, posJ;     // the collision position on I and J. set as zero for spheres
    Evec3 endI, endJ;     // end point for the force chain
    Emat3 stress;

    CollisionBlock() : gidI(0), gidJ(0), globalIndexI(0), globalIndexJ(0), phi0(0), gamma(0) {
        // default constructor
        normI.setZero();
        normJ.setZero();
        posI.setZero();
        posJ.setZero();
        endI.setZero();
        endJ.setZero();
        oneSide = false;
        stress.setZero();
    }

    CollisionBlock(double phi0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
                   const Evec3 &normI_, const Evec3 &normJ_, const Evec3 &posI_, const Evec3 &posJ_,
                   bool oneSide_ = false)
        : phi0(phi0_), gamma(gamma_), gidI(gidI_), gidJ(gidJ_), globalIndexI(globalIndexI_),
          globalIndexJ(globalIndexJ_), normI(normI_), normJ(normJ_), posI(posI_), posJ(posJ_), oneSide(oneSide_) {
        // if oneside = true, the gidJ, globalIndexJ, normJ, posJ will be ignored
        stress.setZero();
        endI.setZero();
        endJ.setZero();
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

    // write collision blocks as VTK vertices
    static void writeVTP(const CollisionBlockPool &colBlockPool, const std::string &prefix, const std::string &postfix,
                         int rank) {
        CollisionBlockQue colBlockQue;
        for (auto &que : colBlockPool) {
            const int curSize = colBlockQue.size();
            const int queSize = que.size();
            colBlockQue.resize(curSize + queSize);
            std::copy(que.cbegin(), que.cend(), colBlockQue.begin() + curSize);
        }
        if (colBlockQue.size() == 0) {
            // fill some fake data otherwise paraview crashes on empty data files
            colBlockQue.resize(5);
        }
        // write VTP for basic data
        //  use float to save some space
        const int nLocal = colBlockQue.size();
        std::vector<float> phi0(nLocal);
        std::vector<float> gamma(nLocal);
        std::vector<int> gid(2 * nLocal);
        std::vector<int> globalIndex(2 * nLocal);
        std::vector<double> pos(3 * nLocal); // position always in double
        std::vector<float> direction(3 * nLocal);
        std::vector<float> length(nLocal);

#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            const auto &block = colBlockQue[i];
            phi0[i] = block.phi0;
            gamma[i] = block.gamma;
            gid[2 * i] = block.gidI;
            gid[2 * i + 1] = block.gidJ;
            globalIndex[2 * i] = block.globalIndexI;
            globalIndex[2 * i + 1] = block.globalIndexJ;
            Evec3 center = 0.5 * (block.endI + block.endJ);
            Evec3 rIJ = block.endJ - block.endI;
            length[i] = rIJ.norm();
            for (int j = 0; j < 3; j++) {
                pos[3 * i + j] = center[j];
                direction[3 * i + j] = rIJ[j] * (1 / length[i]);
            }
        }

        std::ofstream file(prefix + std::string("ColBlock_") + "r" + std::to_string(rank) + std::string("_") + postfix +
                               std::string(".vtp"),
                           std::ios::out);

        IOHelper::writeHeadVTP(file);
        std::string contentB64; // data converted to base64 format
        contentB64.reserve(4 * nLocal);

        file << "<Piece NumberOfPoints=\"" << nLocal << "\" NumberOfCells=\"" << 0 << "\">\n";
        // point data
        file << "<PointData Scalars=\"scalars\">\n";
        IOHelper::writeDataArrayBase64(phi0, "phi0", 1, file);
        IOHelper::writeDataArrayBase64(gamma, "gamma", 1, file);
        IOHelper::writeDataArrayBase64(gid, "gid", 2, file);
        IOHelper::writeDataArrayBase64(globalIndex, "globalIndex", 2, file);
        IOHelper::writeDataArrayBase64(length, "length", 1, file);
        IOHelper::writeDataArrayBase64(direction, "direction", 3, file);

        file << "</PointData>\n";
        // no cell data
        // Points
        file << "<Points>\n";
        IOHelper::writeDataArrayBase64(pos, "points", 3, file);
        file << "</Points>\n";
        file << "</Piece>\n";

        IOHelper::writeTailVTP(file);
        file.close();
    }

    static void writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs) {
        std::vector<IOHelper::FieldVTU> dataFields;
        dataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "phi0");
        dataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "gamma");
        dataFields.emplace_back(2, IOHelper::IOTYPE::Int32, "gid");
        dataFields.emplace_back(2, IOHelper::IOTYPE::Int32, "globalIndex");
        dataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "length");
        dataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "direction");

        std::vector<std::string> pieceNames;
        for (int i = 0; i < nProcs; i++) {
            pieceNames.emplace_back(std::string("ColBlock_") + std::string("r") + std::to_string(i) + "_" + postfix +
                                    ".vtp");
        }

        IOHelper::writePVTPFile(prefix + "ColBlock_" + postfix + ".pvtp", dataFields, pieceNames);
    }
};

#endif
