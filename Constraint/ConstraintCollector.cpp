#include "ConstraintCollector.hpp"
#include "spdlog/spdlog.h"

#include <cstdlib>

ConstraintCollector::ConstraintCollector() {
    const int totalThreads = omp_get_max_threads();

    constraintPoolPtr = std::make_shared<ConstraintPool>();
    constraintPoolPtr->resize(totalThreads);
    for (auto &queue : *constraintPoolPtr) {
        queue.clear();
    }

    constraintBlockPoolPtr = std::make_shared<ConstraintBlockPool>();
    constraintBlockPoolPtr->resize(totalThreads);
    for (auto &queue : *constraintBlockPoolPtr) {
        queue.clear();
    }

    spdlog::debug("ConstraintCollector constructed for {} threads", constraintBlockPoolPtr->size());
}

bool ConstraintCollector::valid() const { return (constraintPoolPtr->empty() && constraintBlockPoolPtr->empty()); }

void ConstraintCollector::clear() {
    assert(constraintPoolPtr);
    for (int i = 0; i < constraintPoolPtr->size(); i++) {
        (*constraintPoolPtr)[i].clear();
    }

    assert(constraintBlockPoolPtr);
    for (int i = 0; i < constraintBlockPoolPtr->size(); i++) {
        (*constraintBlockPoolPtr)[i].clear();
    }

    // keep the total number of queues
    const int totalThreads = omp_get_max_threads();
    constraintPoolPtr->resize(totalThreads); 
    constraintBlockPoolPtr->resize(totalThreads);
}

int ConstraintCollector::getLocalNumberOfConstraints() {
    int sum = 0;
    for (int i = 0; i < constraintPoolPtr->size(); i++) {
        sum += (*constraintPoolPtr)[i].size();
    }
    return sum;
}


int ConstraintCollector::getLocalNumberOfBlocks() {
    int sum = 0;
    for (int i = 0; i < constraintBlockPoolPtr->size(); i++) {
        sum += (*constraintBlockPoolPtr)[i].size();
    }
    return sum;
}

void ConstraintCollector::sumLocalConstraintStress(Emat3 &conStress, bool withOneSide) const {
    // TODO: this needs thread parallelized 
    const auto &cPool = *constraintBlockPoolPtr;
    const int poolSize = cPool.size();

    Emat3 conStressTotal = Emat3::Zero();

#pragma omp parallel for
    for (int que = 0; que < poolSize; que++) {
        // reduction of stress, one que for each thread

        Emat3 conStressSumQue = Emat3::Zero();
        const auto &conQue = cPool[que];
        for (const auto &cBlock : conQue) {
            if (cBlock.oneSide && !withOneSide) {
                // skip counting oneside collision blocks
                continue;
            } else {
                Emat3 stressBlock;
                cBlock.getStress(stressBlock);
                conStressSumQue = conStressSumQue + stressBlock;
            }
        }
#pragma omp critical
        {
            conStressTotal = conStressTotal + conStressSumQue;
        }
    }
    conStress = conStressTotal;
}

void ConstraintCollector::writePVTP(const std::string &folder, const std::string &prefix, const std::string &postfix,
                                    const int nProcs) const {
    std::vector<std::string> pieceNames;

    std::vector<IOHelper::FieldVTU> pointDataFields;
    pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "gid");
    pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "globalIndex");
    pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "forceComIJ");
    pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "torqueComIJ");

    std::vector<IOHelper::FieldVTU> cellDataFields;
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "oneSide");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "id");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "delta");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "gamma");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "invKappa");
    cellDataFields.emplace_back(9, IOHelper::IOTYPE::Float32, "Stress");

    for (int i = 0; i < nProcs; i++) {
        pieceNames.emplace_back(prefix + std::string("ConBlock_") + std::string("r") + std::to_string(i) + "_" +
                                postfix + ".vtp");
    }

    IOHelper::writePVTPFile(folder + "/" + prefix + std::string("ConBlock_") + postfix + ".pvtp", pointDataFields,
                            cellDataFields, pieceNames);
}

void ConstraintCollector::writeVTP(const std::string &folder, const std::string &prefix, const std::string &postfix,
                                   int rank) const {
    const auto &cPool = *constraintBlockPoolPtr;
    const int cQueNum = cPool.size();
    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);
    const int cBlockNum = cQueIndex.back();

    // for each block:

    // write VTP for basic data, use float to save some space
    // point and point data
    std::vector<double> pos(6 * cBlockNum); // position always in Float64
    std::vector<int32_t> gid(2 * cBlockNum);
    std::vector<int32_t> globalIndex(2 * cBlockNum);
    std::vector<float> forceComIJ(6 * cBlockNum);
    std::vector<float> torqueComIJ(6 * cBlockNum);

    // point connectivity of line
    std::vector<int32_t> connectivity(2 * cBlockNum);
    std::vector<int32_t> offset(cBlockNum);

    // cell data for ColBlock
    std::vector<int32_t> oneSide(cBlockNum);
    std::vector<int32_t> id(cBlockNum);
    std::vector<float> delta(cBlockNum);
    std::vector<float> gamma(cBlockNum);
    std::vector<float> invKappa(cBlockNum);
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
            pos[6 * cIndex + 0] = block.labI[0];
            pos[6 * cIndex + 1] = block.labI[1];
            pos[6 * cIndex + 2] = block.labI[2];
            pos[6 * cIndex + 3] = block.labJ[0];
            pos[6 * cIndex + 4] = block.labJ[1];
            pos[6 * cIndex + 5] = block.labJ[2];

            // point data
            gid[2 * cIndex + 0] = block.gidI;
            gid[2 * cIndex + 1] = block.gidJ;
            globalIndex[2 * cIndex + 0] = block.globalIndexI;
            globalIndex[2 * cIndex + 1] = block.globalIndexJ;
            forceComIJ[6 * cIndex + 0] = block.unscaledForceComI[0] * block.gamma;
            forceComIJ[6 * cIndex + 1] = block.unscaledForceComI[1] * block.gamma;
            forceComIJ[6 * cIndex + 2] = block.unscaledForceComI[2] * block.gamma;
            forceComIJ[6 * cIndex + 3] = block.unscaledForceComJ[0] * block.gamma;
            forceComIJ[6 * cIndex + 4] = block.unscaledForceComJ[1] * block.gamma;
            forceComIJ[6 * cIndex + 5] = block.unscaledForceComJ[2] * block.gamma;
            torqueComIJ[6 * cIndex + 0] = block.unscaledTorqueComI[0] * block.gamma;
            torqueComIJ[6 * cIndex + 1] = block.unscaledTorqueComI[1] * block.gamma;
            torqueComIJ[6 * cIndex + 2] = block.unscaledTorqueComI[2] * block.gamma;
            torqueComIJ[6 * cIndex + 3] = block.unscaledTorqueComJ[0] * block.gamma;
            torqueComIJ[6 * cIndex + 4] = block.unscaledTorqueComJ[1] * block.gamma;
            torqueComIJ[6 * cIndex + 5] = block.unscaledTorqueComJ[2] * block.gamma;

            // cell data
            oneSide[cIndex] = block.oneSide ? 1 : 0;
            id[cIndex] = block.id;
            delta[cIndex] = block.delta;
            gamma[cIndex] = block.gamma;
            invKappa[cIndex] = block.invKappa;
            for (int kk = 0; kk < 9; kk++) {
                Stress[9 * cIndex + kk] = block.stress[kk];
            }
        }
    }

    std::ofstream file(folder + '/' + prefix + std::string("ConBlock_") + "r" + std::to_string(rank) +
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
    IOHelper::writeDataArrayBase64(forceComIJ, "forceComIJ", 3, file);
    IOHelper::writeDataArrayBase64(torqueComIJ, "torqueComIJ", 3, file);
    file << "</PointData>\n";
    // cell data
    file << "<CellData Scalars=\"scalars\">\n";
    IOHelper::writeDataArrayBase64(oneSide, "oneSide", 1, file);
    IOHelper::writeDataArrayBase64(id, "id", 1, file);
    IOHelper::writeDataArrayBase64(delta, "delta", 1, file);
    IOHelper::writeDataArrayBase64(gamma, "gamma", 1, file);
    IOHelper::writeDataArrayBase64(invKappa, "invKappa", 1, file);
    IOHelper::writeDataArrayBase64(Stress, "Stress", 9, file);
    file << "</CellData>\n";
    // Piece ends
    file << "</Piece>\n";

    IOHelper::writeTailVTP(file);
    file.close();
}

void ConstraintCollector::dumpBlocks() const {
    std::cout << "number of constraint block queues: " << constraintBlockPoolPtr->size() << std::endl;
    // dump constraint blocks
    for (const auto &conBlockQue : (*constraintBlockPoolPtr)) {
        std::cout << conBlockQue.size() << " DOF in this queue" << std::endl;
        for (const auto &block : conBlockQue) {
            std::cout << block.globalIndexI << " " << block.globalIndexJ << "  delta:" << block.delta << std::endl;
        }
    }
}

void ConstraintCollector::dumpConstraints() const {
    std::cout << "number of constraint queues: " << constraintPoolPtr->size() << std::endl;
    // dump constraint blocks
    for (const auto &conQue : (*constraintPoolPtr)) {
        std::cout << conQue.size() << " constraints in this queue" << std::endl;
        for (const auto &con : conQue) {
            const int id = con->getID();
            const int numDOF = con->getNumDOF();  
            std::cout << "ID: " << id << " | NumDOF: " << numDOF << std::endl;
        }
    }
}

/*
TODO: Improve constraimnts to have multiple degrees of freedom 

    Each constraint may have more than 1 degree of freedom. This is specified within each constraint block and accessed using getLocalDOF()
    Local DOF can also be obtained using buildConIndex, which outputs cQueSize, cQueIndex, cDofSize, cDofIndex. 
    Constraints can have the following types:
      id=0: collision              | 3 constrained DOF | prevents penetration and enforces tangency of colliding surfaces
      id=1: hookean spring         | 3 constrained DOF | resists relative translational motion between two points
      id=2: angular hookean spring | 3 constrained DOF | resists relative rotational motion between two vectors
      id=3: ball and socket        | 3 constrained DOF | prevents the separation of two points
    id 0 and 1 are implemented. 2 and 3 need finished


    Steps to make this all possible:
    (Skip for now) We should create an abstract constraint class, which specifies the number of degrees of freedom and thier corresponding non-zero terms within the D matrix 
        This step will be circumvented, for now, by using if statements and hard-coding such things. 
    (Done) We must update the constraints to include a type identidier that is more complex then bilateral or non-bilaterial. 
    We must modify the constraints to specify the number of degrees of freedom. 
        This should be determined solely by their type modifier. Instead, we should modify ConstraintCollector to know their DOF based on their type. 
        This is temporary and will be replaced when we abstractify using a constraint class. 
    (Done) We should simplify the Sylinder class to only have constraint forces/torques/velocities, rather than breaking things into bilaterial and collision. 
    We must create getLocalDOF to return the total local constraint degrees of freedom.
    We must update buildConIndex to output both constraint and DOF indexing/sizes.
    We must update buildConstraintMatrixVector to use the correct direction depending on the constraint type identifier. 



*/






int ConstraintCollector::buildConstraintMatrixVector(const Teuchos::RCP<const TMAP> &mobMapRcp, //
                                                     Teuchos::RCP<TCMAT> &DMatTransRcp,         //
                                                     Teuchos::RCP<TV> &deltaRcp,                //
                                                     Teuchos::RCP<TV> &invKappaRcp,             //
                                                     Teuchos::RCP<TV> &gammaGuessRcp) const {
    Teuchos::RCP<const TCOMM> commRcp = mobMapRcp->getComm();

    const auto &cPool = *constraintBlockPoolPtr; // the constraint pool
    const int cQueNum = cPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> cBlockQueSize;
    std::vector<int> cBlockQueIndex;
    buildConBlockIndex(cBlockQueSize, cBlockQueIndex);

    // prepare 2, allocate the map and vectors
    const int localGammaSize = cBlockQueIndex.back();
    Teuchos::RCP<const TMAP> gammaMapRcp = getTMAPFromLocalSize(localGammaSize, commRcp);

    // step 1, count the number of entries to each row
    // each constraint block, correspoding to a gamma, occupies a row
    // 12 entries for two side constraint blocks
    // 6 entries for one side constraint blocks
    Kokkos::View<size_t *> rowPointers("rowPointers", localGammaSize + 1); // last entry is the total nnz in this matrix
    rowPointers[0] = 0;

    std::vector<int> colIndexPool(cQueNum + 1, 0); // The beginning index in crs columnIndices for each constraint queue

    int rowPointerIndex = 0;
    int colIndexCount = 0;
    for (int i = 0; i < cQueNum; i++) {
        const auto &queue = cPool[i];
        const int jsize = queue.size();
        for (int j = 0; j < jsize; j++) {
            rowPointerIndex++;
            const int cBlockNNZ = (queue[j].oneSide ? 6 : 12);
            rowPointers[rowPointerIndex] = rowPointers[rowPointerIndex - 1] + cBlockNNZ;
            colIndexCount += cBlockNNZ;
        }
        colIndexPool[i + 1] = colIndexCount;
    }

    if (rowPointerIndex != localGammaSize) {
        spdlog::critical("rowPointerIndexError in collision solver");
        std::exit(1);
    }

    // step 2, fill the values to each row
    Kokkos::View<int *> columnIndices("columnIndices", rowPointers[localGammaSize]);
    Kokkos::View<double *> values("values", rowPointers[localGammaSize]);
    // multi-thread filling. nThreads = poolSize, each thread process a queue
    const int nThreads = cPool.size();
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        // each thread process a queue
        const auto &cBlockQue = cPool[threadId];
        const int cBlockNum = cBlockQue.size();
        const int cBlockIndexBase = cBlockQueIndex[threadId];
        int kk = colIndexPool[threadId];

        for (int j = 0; j < cBlockNum; j++) {
            // each 6nnz for an object: gx.ux+gy.uy+gz.uz+(gzpy-gypz)wx+(gxpz-gzpx)wy+(gypx-gxpy)wz
            // 6 nnz for I
            columnIndices[kk + 0] = 6 * cBlockQue[j].globalIndexI;
            columnIndices[kk + 1] = 6 * cBlockQue[j].globalIndexI + 1;
            columnIndices[kk + 2] = 6 * cBlockQue[j].globalIndexI + 2;
            columnIndices[kk + 3] = 6 * cBlockQue[j].globalIndexI + 3;
            columnIndices[kk + 4] = 6 * cBlockQue[j].globalIndexI + 4;
            columnIndices[kk + 5] = 6 * cBlockQue[j].globalIndexI + 5;
            values[kk + 0] = cBlockQue[j].unscaledForceComI[0];
            values[kk + 1] = cBlockQue[j].unscaledForceComI[1];
            values[kk + 2] = cBlockQue[j].unscaledForceComI[2];
            values[kk + 3] = cBlockQue[j].unscaledTorqueComI[0];
            values[kk + 4] = cBlockQue[j].unscaledTorqueComI[1];
            values[kk + 5] = cBlockQue[j].unscaledTorqueComI[2];
            kk += 6;
            if (!cBlockQue[j].oneSide) {
                // 6 nnz for J
                columnIndices[kk + 0] = 6 * cBlockQue[j].globalIndexJ;
                columnIndices[kk + 1] = 6 * cBlockQue[j].globalIndexJ + 1;
                columnIndices[kk + 2] = 6 * cBlockQue[j].globalIndexJ + 2;
                columnIndices[kk + 3] = 6 * cBlockQue[j].globalIndexJ + 3;
                columnIndices[kk + 4] = 6 * cBlockQue[j].globalIndexJ + 4;
                columnIndices[kk + 5] = 6 * cBlockQue[j].globalIndexJ + 5;
                values[kk + 0] = cBlockQue[j].unscaledForceComJ[0];
                values[kk + 1] = cBlockQue[j].unscaledForceComJ[1];
                values[kk + 2] = cBlockQue[j].unscaledForceComJ[2];
                values[kk + 3] = cBlockQue[j].unscaledTorqueComJ[0];
                values[kk + 4] = cBlockQue[j].unscaledTorqueComJ[1];
                values[kk + 5] = cBlockQue[j].unscaledTorqueComJ[2];
                kk += 6;
            }
        }
    }

    // step 3 prepare the partitioned column map
    // Each process own some columns. In the map, processes share entries.
    // 3.1 column map has to cover the contiguous range of the mobility map locally owned
    const int mobMin = mobMapRcp->getMinGlobalIndex();
    const int mobMax = mobMapRcp->getMaxGlobalIndex();
    std::vector<int> colMapIndex(mobMax - mobMin + 1);
#pragma omp parallel for
    for (int i = mobMin; i <= mobMax; i++) {
        colMapIndex[i - mobMin] = i;
    }
    // this is the list of the columns that have nnz entries
    // if the column index is out of [mobMinLID, mobMaxLID], add it to the map
    const auto colIndexNum = columnIndices.extent(0);
    if (colIndexNum != colIndexCount) {
        spdlog::critical("colIndexNum error");
        std::exit(1);
    }
    for (size_t i = 0; i < colIndexNum; i++) {
        if (columnIndices[i] < mobMin || columnIndices[i] > mobMax)
            colMapIndex.push_back(columnIndices[i]);
    }

    // sort and unique
    std::sort(colMapIndex.begin(), colMapIndex.end());
    auto ip = std::unique(colMapIndex.begin(), colMapIndex.end());
    colMapIndex.resize(std::distance(colMapIndex.begin(), ip));

    // create colMap
    Teuchos::RCP<TMAP> colMapRcp = Teuchos::rcp(
        new TMAP(Teuchos::OrdinalTraits<int>::invalid(), colMapIndex.data(), colMapIndex.size(), 0, commRcp));

    // convert columnIndices from global column index to local column index according to colMap
    auto &colmap = *colMapRcp;
#pragma omp parallel for
    for (size_t i = 0; i < colIndexNum; i++) {
        columnIndices[i] = colmap.getLocalElement(columnIndices[i]);
    }

    // step 4, allocate the D^Trans matrix
    DMatTransRcp = Teuchos::rcp(new TCMAT(gammaMapRcp, colMapRcp, rowPointers, columnIndices, values));
    DMatTransRcp->fillComplete(mobMapRcp, gammaMapRcp); // domainMap, rangeMap

    // step 5, fill the delta, gammaGuess, invKappa, conFlag vectors
    deltaRcp = Teuchos::rcp(new TV(gammaMapRcp, true));
    invKappaRcp = Teuchos::rcp(new TV(gammaMapRcp, true));
    gammaGuessRcp = Teuchos::rcp(new TV(gammaMapRcp, true));
    auto delta = deltaRcp->getLocalView<Kokkos::HostSpace>();
    auto invKappa = invKappaRcp->getLocalView<Kokkos::HostSpace>();
    auto gammaGuess = gammaGuessRcp->getLocalView<Kokkos::HostSpace>();
    deltaRcp->modify<Kokkos::HostSpace>();
    invKappaRcp->modify<Kokkos::HostSpace>();
    gammaGuessRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(cQueNum)
    for (int que = 0; que < cQueNum; que++) {
        const auto &cQue = cPool[que];
        const int cIndexBase = cBlockQueIndex[que];
        const int queSize = cQue.size();
        for (int j = 0; j < queSize; j++) {
            const auto &block = cQue[j];
            const auto idx = cIndexBase + j;
            delta(idx, 0) = block.delta;
            invKappa(idx, 0) = block.invKappa;
            gammaGuess(idx, 0) = block.gamma;
        }
    }

    return 0;
}

int ConstraintCollector::applyProjectionToDOF(const Teuchos::RCP<TV> &gammaRcp) const {
    // build the index for block queue
    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);

    // multi-thread projecting. nThreads = poolSize, each thread processes a queue
    const auto &cPool = *constraintPoolPtr;
    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    gammaRcp->modify<Kokkos::HostSpace>();

    const int nThreads = cPool.size();
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        // each thread process a queue
        const auto &cQue = cPool[threadId];
        int kk = cQueIndex[threadId];

        const int cNum = cQue.size();
        for (int j = 0; j < cNum; j++) {
            // extract the constrained dof
            const int numDOF = cQue[j]->getNumDOF(); 
            std::vector<double> gammas(numDOF);
            for (int d = 0; d < numDOF; d++) {
                gammas[d] = gammaPtr(kk + d, 0);
            }
            
            // for (auto i: gammas)
            //     std::cout << i << ' ';
            // std::cout << std::endl;

            // apply the projection
            cQue[j]->projectDOF(gammas);

            // for (auto i: gammas)
            //     std::cout << i << ' ';
            // std::cout << std::endl;

            // write back the result
            for (int d = 0; d < numDOF; d++) {
                gammaPtr(kk + d, 0) = gammas[d];
            }

            // increment the index
            kk += numDOF;
        }
    }
    return 0;
}

int ConstraintCollector::applyProjectionToValues(const Teuchos::RCP<const TV> &gammaRcp,
                                                 const Teuchos::RCP<TV> &valuesRcp) const {
    // build the index for block queue
    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);

    // multi-thread projecting. nThreads = poolSize, each thread processes a queue
    const auto &cPool = *constraintPoolPtr;
    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    auto valuesPtr = valuesRcp->getLocalView<Kokkos::HostSpace>();
    valuesRcp->modify<Kokkos::HostSpace>();

    const int nThreads = cPool.size();
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        // each thread process a queue
        const auto &cQue = cPool[threadId];
        int kk = cQueIndex[threadId];

        const int cNum = cQue.size();
        for (int j = 0; j < cNum; j++) {
            // extract the constrained dof and values
            const int numDOF = cQue[j]->getNumDOF(); 
            std::vector<double> gammas(numDOF);
            std::vector<double> values(numDOF);
            for (int d = 0; d < numDOF; d++) {
                gammas[d] = gammaPtr(kk + d, 0);
                values[d] = valuesPtr(kk + d, 0);
            }
            
            // for (auto i: gammas)
            //     std::cout << i << ' ';
            // std::cout << std::endl;

            // for (auto i: values)
            //     std::cout << i * 0.0001 << ' ';
            // std::cout << std::endl;

            // apply the projection
            cQue[j]->projectValues(gammas, values);

            // for (auto i: values)
            //     std::cout << i * 0.0001 << ' ';
            // std::cout << std::endl;

            // write back the result
            for (int d = 0; d < numDOF; d++) {
                valuesPtr(kk + d, 0) = values[d];
            }

            // increment the index
            kk += numDOF;
        }
    }
    return 0;
}


int ConstraintCollector::buildConBlockIndex(std::vector<int> &cBlockQueSize, std::vector<int> &cBlockQueIndex) const {
    const auto &cBlockPool = *constraintBlockPoolPtr;
    const int cBlockQueNum = cBlockPool.size();
    cBlockQueSize.resize(cBlockQueNum);
    cBlockQueIndex.resize(cBlockQueNum + 1);
    for (int i = 0; i < cBlockQueNum; i++) {
        cBlockQueSize[i] = cBlockPool[i].size();
    }
    for (int i = 1; i < cBlockQueNum + 1; i++) {
        cBlockQueIndex[i] = cBlockQueSize[i - 1] + cBlockQueIndex[i - 1];
    }
    return 0;
}

int ConstraintCollector::buildConIndex(std::vector<int> &cQueSize, std::vector<int> &cQueIndex) const {
    const auto &cPool = *constraintPoolPtr;

    // multi-thread filling. nThreads = poolSize, each thread process a queue
    const int nThreads = cPool.size();
    cQueSize.resize(nThreads);
    cQueIndex.resize(nThreads + 1);
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        // each thread process a queue
        const auto &cQue = cPool[threadId];
        const int cNum = cQue.size();

        for (int j = 0; j < cNum; j++) {
            cQueSize[threadId] += cQue[j]->getNumDOF();
        }
    }

    for (int i = 1; i < nThreads + 1; i++) {
        cQueIndex[i] = cQueSize[i - 1] + cQueIndex[i - 1];
    }
    return 0;
}

int ConstraintCollector::writeBackGamma(const Teuchos::RCP<const TV> &gammaRcp) {
    auto &cPool = *constraintBlockPoolPtr; // the constraint pool
    const int cQueNum = cPool.size();

    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);

    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(cQueNum)
    for (int i = 0; i < cQueNum; i++) {
        const int cQueSize = cPool[i].size();
        for (int j = 0; j < cQueSize; j++) {
            cPool[i][j].gamma = gammaPtr(cQueIndex[i] + j, 0);
            for (int k = 0; k < 9; k++) {
                cPool[i][j].stress[k] *= cPool[i][j].gamma;
            }
        }
    }

    return 0;
}

int ConstraintCollector::writeBackDelta(const Teuchos::RCP<const TV> &deltaRcp) {
    auto &cPool = *constraintBlockPoolPtr; // the constraint pool
    const int cQueNum = cPool.size();

    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);

    auto deltaPtr = deltaRcp->getLocalView<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(cQueNum)
    for (int i = 0; i < cQueNum; i++) {
        const int cQueSize = cPool[i].size();
        for (int j = 0; j < cQueSize; j++) {
            cPool[i][j].delta = deltaPtr(cQueIndex[i] + j, 0);
        }
    }

    return 0;
}