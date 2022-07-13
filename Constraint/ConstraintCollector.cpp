#include "ConstraintCollector.hpp"
#include "spdlog/spdlog.h"

#include <cstdlib>

ConstraintCollector::ConstraintCollector() {
    const int totalThreads = omp_get_max_threads();
    constraintPoolPtr = std::make_shared<ConstraintBlockPool>();
    constraintPoolPtr->resize(totalThreads);
    for (auto &queue : *constraintPoolPtr) {
        queue.clear();
    }

    spdlog::debug("ConstraintCollector constructed for {} threads", constraintPoolPtr->size());
}


bool ConstraintCollector::valid() const { return constraintPoolPtr->empty(); }


void ConstraintCollector::clear() {
    assert(constraintPoolPtr);
    for (int i = 0; i < constraintPoolPtr->size(); i++) {
        (*constraintPoolPtr)[i].clear();
    }

    // keep the total number of queues
    const int totalThreads = omp_get_max_threads();
    constraintPoolPtr->resize(totalThreads);
}


int ConstraintCollector::getLocalNumberOfConstraints() {
    int sum = 0;
    for (int i = 0; i < constraintPoolPtr->size(); i++) {
        sum += (*constraintPoolPtr)[i].size();
    }
    return sum;
}


void ConstraintCollector::sumLocalConstraintStress(Emat3 &uniStress, Emat3 &biStress, bool withOneSide) const {
    const auto &cPool = *constraintPoolPtr;
    const int poolSize = cPool.size();

    Emat3 biStressTotal = Emat3::Zero();
    Emat3 uniStressTotal = Emat3::Zero();

#pragma omp parallel for
    for (int que = 0; que < poolSize; que++) {
        // reduction of stress, one que for each thread

        Emat3 uniStressSumQue = Emat3::Zero();
        Emat3 biStressSumQue = Emat3::Zero();
        const auto &conQue = cPool[que];
        for (const auto &cBlock : conQue) {
            if (cBlock.oneSide && !withOneSide) {
                // skip counting oneside collision blocks
                continue;
            } else {
                Emat3 stressBlock;
                cBlock.getStress(stressBlock);
                if (cBlock.bilateral) {
                    biStressSumQue = biStressSumQue + stressBlock;
                } else {
                    uniStressSumQue = uniStressSumQue + stressBlock;
                }
            }
        }
#pragma omp critical
        {
            biStressTotal = biStressTotal + biStressSumQue;
            uniStressTotal = uniStressTotal + uniStressSumQue;
        }
    }
    uniStress = uniStressTotal;
    biStress = biStressTotal;
}


void ConstraintCollector::writePVTP(const std::string &folder, const std::string &prefix, const std::string &postfix,
                                    const int nProcs) const {
    std::vector<std::string> pieceNames;

    std::vector<IOHelper::FieldVTU> pointDataFields;
    pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "gid");
    pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "globalIndex");
    pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "posIJ");
    pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "normIJ");

    std::vector<IOHelper::FieldVTU> cellDataFields;
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "oneSide");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "bilateral");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "value");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "sep");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "gamma");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "diagonal");
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
    const auto &cPool = *constraintPoolPtr;
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
    std::vector<float> posIJ(6 * cBlockNum);
    std::vector<float> normIJ(6 * cBlockNum);

    // point connectivity of line
    std::vector<int32_t> connectivity(2 * cBlockNum);
    std::vector<int32_t> offset(cBlockNum);

    // cell data for ColBlock
    std::vector<int32_t> oneSide(cBlockNum);
    std::vector<int32_t> bilateral(cBlockNum);
    std::vector<float> constraintValue(cBlockNum);
    std::vector<float> sepDist(cBlockNum);
    std::vector<float> gamma(cBlockNum);
    std::vector<float> constraintDiagonal(cBlockNum);
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
            bilateral[cIndex] = block.bilateral ? 1 : 0;
            constraintValue[cIndex] = block.constraintValue;
            sepDist[cIndex] = block.sepDist;
            gamma[cIndex] = block.gamma;
            constraintDiagonal[cIndex] = block.constraintDiagonal;
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
    IOHelper::writeDataArrayBase64(posIJ, "posIJ", 3, file);
    IOHelper::writeDataArrayBase64(normIJ, "normIJ", 3, file);
    file << "</PointData>\n";
    // cell data
    file << "<CellData Scalars=\"scalars\">\n";
    IOHelper::writeDataArrayBase64(oneSide, "oneSide", 1, file);
    IOHelper::writeDataArrayBase64(bilateral, "bilateral", 1, file);
    IOHelper::writeDataArrayBase64(constraintValue, "value", 1, file);
    IOHelper::writeDataArrayBase64(sepDist, "sep", 1, file);
    IOHelper::writeDataArrayBase64(gamma, "gamma", 1, file);
    IOHelper::writeDataArrayBase64(constraintDiagonal, "diagonal", 1, file);
    IOHelper::writeDataArrayBase64(Stress, "Stress", 9, file);
    file << "</CellData>\n";
    // Piece ends
    file << "</Piece>\n";

    IOHelper::writeTailVTP(file);
    file.close();
}


void ConstraintCollector::dumpBlocks() const {
    std::cout << "number of collision queues: " << constraintPoolPtr->size() << std::endl;
    // dump constraint blocks
    for (const auto &blockQue : (*constraintPoolPtr)) {
        std::cout << blockQue.size() << " constraints in this queue" << std::endl;
        for (const auto &block : blockQue) {
            std::cout << block.globalIndexI << " " << block.globalIndexJ << " value:" << block.constraintValue << std::endl;
        }
    }
}


Teuchos::RCP<TCMAT> ConstraintCollector::buildConstraintMatrixVector(const Teuchos::RCP<const TMAP> &mobMapRcp,
                                                                     const Teuchos::RCP<const TMAP> &gammaMapRcp,
                                                                     const Teuchos::RCP<const TV> &gammaRcp) const {
    TEUCHOS_ASSERT(nonnull(mobMapRcp));
    TEUCHOS_ASSERT(nonnull(gammaMapRcp));

    const Teuchos::RCP<const TCOMM> commRcp = mobMapRcp->getComm();

    const auto &cPool = *constraintPoolPtr; // the constraint pool
    const int cQueNum = cPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);
    const int localGammaSize = cQueIndex.back();

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
    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    Kokkos::View<int *> columnIndices("columnIndices", rowPointers[localGammaSize]);
    Kokkos::View<double *> values("values", rowPointers[localGammaSize]);
    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(cQueNum)
    for (int threadId = 0; threadId < cQueNum; threadId++) {
        // each thread process a queue
        const auto &cBlockQue = cPool[threadId];
        const int cBlockNum = cBlockQue.size();
        const int cBlockIndexBase = cQueIndex[threadId];
        int kk = colIndexPool[threadId];

        for (int j = 0; j < cBlockNum; j++) {
            // each 6nnz for an object: gx.ux+gy.uy+gz.uz+(gzpy-gypz)wx+(gxpz-gzpx)wy+(gypx-gxpy)wz
            // 6 nnz for I
            const auto &cBlock = cBlockQue[j];
            const auto idx = cBlockIndexBase + j;

            // TODO: This entire for loop needs modularized. Desperately! 
            // Scale accounts for complementarity constraints 
            double scale;
            if (cBlock.bilateral) {
                // spring constraint
                scale = 1.0;
            } else {
                // collision constraint using Ficher Bermiester function
                // this is the partial derivative of the FB function (sep^k, gamma^k) w.r.t sep^k
                // TODO: what number should be used here?
                if ((std::abs(cBlock.sepDist) < 1e-6) && (std::abs(gammaPtr(idx, 0)) < 1e-6)) {
                    scale = 0.0;
                } else {
                    scale = 1.0 - cBlock.sepDist / std::sqrt(std::pow(cBlock.sepDist , 2) + std::pow(gammaPtr(idx, 0) , 2));
                }
            }

            columnIndices[kk + 0] = 6 * cBlock.globalIndexI;
            columnIndices[kk + 1] = 6 * cBlock.globalIndexI + 1;
            columnIndices[kk + 2] = 6 * cBlock.globalIndexI + 2;
            columnIndices[kk + 3] = 6 * cBlock.globalIndexI + 3;
            columnIndices[kk + 4] = 6 * cBlock.globalIndexI + 4;
            columnIndices[kk + 5] = 6 * cBlock.globalIndexI + 5;
            const double &gx = cBlock.normI[0];
            const double &gy = cBlock.normI[1];
            const double &gz = cBlock.normI[2];
            const double &px = cBlock.posI[0];
            const double &py = cBlock.posI[1];
            const double &pz = cBlock.posI[2];
            values[kk + 0] = scale * gx;
            values[kk + 1] = scale * gy;
            values[kk + 2] = scale * gz;
            values[kk + 3] = scale * (gz * py - gy * pz);
            values[kk + 4] = scale * (gx * pz - gz * px);
            values[kk + 5] = scale * (gy * px - gx * py);
            kk += 6;
            if (!cBlock.oneSide) {
                columnIndices[kk + 0] = 6 * cBlock.globalIndexJ;
                columnIndices[kk + 1] = 6 * cBlock.globalIndexJ + 1;
                columnIndices[kk + 2] = 6 * cBlock.globalIndexJ + 2;
                columnIndices[kk + 3] = 6 * cBlock.globalIndexJ + 3;
                columnIndices[kk + 4] = 6 * cBlock.globalIndexJ + 4;
                columnIndices[kk + 5] = 6 * cBlock.globalIndexJ + 5;
                const double &gx = cBlock.normJ[0];
                const double &gy = cBlock.normJ[1];
                const double &gz = cBlock.normJ[2];
                const double &px = cBlock.posJ[0];
                const double &py = cBlock.posJ[1];
                const double &pz = cBlock.posJ[2];
                values[kk + 0] = scale * gx;
                values[kk + 1] = scale * gy;
                values[kk + 2] = scale * gz;
                values[kk + 3] = scale * (gz * py - gy * pz);
                values[kk + 4] = scale * (gx * pz - gz * px);
                values[kk + 5] = scale * (gy * px - gx * py);
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
    const int colIndexNum = columnIndices.dimension_0();
    if (colIndexNum != colIndexCount) {
        spdlog::critical("colIndexNum error");
        std::exit(1);
    }
    for (int i = 0; i < colIndexNum; i++) {
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
    for (int i = 0; i < colIndexNum; i++) {
        columnIndices[i] = colmap.getLocalElement(columnIndices[i]);
    }

    // printf("local number of cols: %d on rank %d\n", colMapRcp->getNodeNumElements(), commRcp->getRank());
    // printf("local number of rows: %d on rank %d\n", mobMapRcp->getNodeNumElements(), commRcp->getRank());
    // dumpTMAP(colMapRcp,"colMap");
    // dumpTMAP(mobMapRcp,"mobMap");

    // step 4, allocate the D^Trans matrix
    Teuchos::RCP<TCMAT> DMatTransRcp = Teuchos::rcp(new TCMAT(gammaMapRcp, colMapRcp, rowPointers, columnIndices, values));
    DMatTransRcp->fillComplete(mobMapRcp, gammaMapRcp); // domainMap, rangeMap

    return DMatTransRcp;
}


int ConstraintCollector::evalConstraintDiagonal(const Teuchos::RCP<const TV> &gammaRcp,
                                                const Teuchos::RCP<TV> &constraintDiagonalRcp) const {
    TEUCHOS_ASSERT(nonnull(gammaRcp));
    TEUCHOS_ASSERT(nonnull(constraintDiagonalRcp));

    const auto &cPool = *constraintPoolPtr; // the constraint pool
    const int cQueNum = cPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);

    //  fill constraintDiagonal
    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintDiagonalPtr = constraintDiagonalRcp->getLocalView<Kokkos::HostSpace>();
    constraintDiagonalRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(cQueNum)
    for (int threadId = 0; threadId < cQueNum; threadId++) {
        const auto &cBlockQue = cPool[threadId];
        const int cBlockNum = cBlockQue.size();
        const int cBlockIndexBase = cQueIndex[threadId];
        for (int j = 0; j < cBlockNum; j++) {
            const auto &cBlock = cBlockQue[j];
            const auto idx = cBlockIndexBase + j;
            if (cBlock.bilateral) {
                // spring constraint
                // this is the partial derivative of the linear spring constraint w.r.t gamma^k
                constraintDiagonalPtr(idx, 0) = 1.0 / cBlock.kappa;
            } else {
                // collision constraint using Ficher Bermiester function
                // this is the partial derivative of the FB function (sep^k, gamma^k) w.r.t gamma^k
                // TODO: what number should be used here?
                if ((std::abs(cBlock.sepDist) < 1e-6) && (std::abs(gammaPtr(idx, 0)) < 1e-6)) {
                    constraintDiagonalPtr(idx, 0) = 1.0;
                } else {
                    constraintDiagonalPtr(idx, 0) = 1.0 
                        - gammaPtr(idx, 0) / std::sqrt(std::pow(cBlock.sepDist , 2) + std::pow(gammaPtr(idx, 0) , 2));
                }
            }
        }
    }

    return 0;
}


int ConstraintCollector::evalConstraintValues(const Teuchos::RCP<const TV> &gammaRcp,
                                              const Teuchos::RCP<TV> &constraintValueRcp) const {
    TEUCHOS_ASSERT(nonnull(gammaRcp));
    TEUCHOS_ASSERT(nonnull(constraintValueRcp));

    auto &cPool = *constraintPoolPtr; // the constraint pool
    const int cQueNum = cPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);

    //  fill constraintValue
    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintValuePtr = constraintValueRcp->getLocalView<Kokkos::HostSpace>();
    constraintValueRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(cQueNum)
    for (int threadId = 0; threadId < cQueNum; threadId++) {
        auto &cQue = cPool[threadId];
        const int cIndexBase = cQueIndex[threadId];
        const int queSize = cQue.size();
        for (int j = 0; j < queSize; j++) {
            auto &cblock = cQue[j];
            const auto idx = cIndexBase + j;
            if (cblock.bilateral) {
                // spring constraint
                // this is the value of linear spring constraint evaluated at gamma^k, q^k
                constraintValuePtr(idx, 0) = cblock.sepDist + 1.0 / cblock.kappa * gammaPtr(idx, 0);
            } else {
                // collision constraint using Ficher Bermiester function
                // this is the value of the FB function evaluated at gamma^k, sep(q^k)
                constraintValuePtr(idx, 0) = cblock.sepDist + gammaPtr(idx, 0) 
                    - std::sqrt(std::pow(cblock.sepDist , 2) + std::pow(gammaPtr(idx, 0) , 2));
            }
            cblock.constraintValue = constraintValuePtr(idx, 0);
        }
    }

    return 0;
}


int ConstraintCollector::fillConstraintInformation(const Teuchos::RCP<const TCOMM>& commRcp,
                                                   const Teuchos::RCP<TV> &gammaGuessRcp,         
                                                   const Teuchos::RCP<TV> &constraintFlagRcp) const {
    TEUCHOS_ASSERT(nonnull(commRcp));
    TEUCHOS_ASSERT(nonnull(gammaGuessRcp));
    TEUCHOS_ASSERT(nonnull(constraintFlagRcp));

    const auto &cPool = *constraintPoolPtr; // the constraint pool
    const int cQueNum = cPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> cQueSize;
    std::vector<int> cQueIndex;
    buildConIndex(cQueSize, cQueIndex);

    // fill the gammaGuessRcp and constraintFlagRcp vectors
    auto gammaGuessPtr = gammaGuessRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintFlagPtr = constraintFlagRcp->getLocalView<Kokkos::HostSpace>();
    gammaGuessRcp->modify<Kokkos::HostSpace>();
    constraintFlagRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(cQueNum)
    for (int que = 0; que < cQueNum; que++) {
        const auto &cQue = cPool[que];
        const int cIndexBase = cQueIndex[que];
        const int queSize = cQue.size();
        for (int j = 0; j < queSize; j++) {
            const auto &block = cQue[j];
            const auto idx = cIndexBase + j;
            gammaGuessPtr(idx, 0) = block.gammaGuess;
            if (block.bilateral) {
                constraintFlagPtr(idx, 0) = 1;
            }
        }
    }

    return 0;
}


int ConstraintCollector::buildConIndex(std::vector<int> &cQueSize, std::vector<int> &cQueIndex) const {
    const auto &cPool = *constraintPoolPtr;
    const int cQueNum = cPool.size();
    cQueSize.resize(cQueNum);
    cQueIndex.resize(cQueNum + 1);
    for (int i = 0; i < cQueNum; i++) {
        cQueSize[i] = cPool[i].size();
    }
    for (int i = 1; i < cQueNum + 1; i++) {
        cQueIndex[i] = cQueSize[i - 1] + cQueIndex[i - 1];
    }
    return 0;
}


int ConstraintCollector::writeBackGamma(const Teuchos::RCP<const TV> &gammaRcp) {
    TEUCHOS_ASSERT(nonnull(gammaRcp));

    auto &cPool = *constraintPoolPtr; // the constraint pool
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