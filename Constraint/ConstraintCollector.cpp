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

int ConstraintCollector::getLocalNumberOfDOF() {
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);
    return conQueIndex.back();
}

void ConstraintCollector::sumLocalConstraintStress(Emat3 &conStress, bool withOneSide) const {
    // TODO: this needs thread parallelized
    const auto &conPool = *constraintPoolPtr;
    const int conQueNum = conPool.size();

    Emat3 conStressTotal = Emat3::Zero();

#pragma omp parallel for num_threads(conQueNum)
    for (int que = 0; que < conQueNum; que++) {
        // reduction of stress, one que for each thread

        Emat3 conStressSumQue = Emat3::Zero();
        const auto &conQue = conPool[que];
        for (const auto &con : conQue) {
            for (int d = 0; d < con.numDOF; d++) {
                if (con.oneSide && !withOneSide) {
                    // skip counting oneside collision blocks
                    continue;
                } else {
                    Emat3 stressDOF;
                    con.getStress(d, stressDOF);
                    conStressSumQue = conStressSumQue + stressDOF;
                }
            }
        }
#pragma omp critical
        { conStressTotal = conStressTotal + conStressSumQue; }
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
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "id");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "oneSide");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "sep");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "gamma");
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
    const auto &conPool = *constraintPoolPtr;
    const int conQueNum = conPool.size();
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);
    const int conBlockNum = conQueIndex.back();

    // for each block:

    // write VTP for basic data, use float to save some space
    // point and point data
    std::vector<double> pos(6 * conBlockNum); // position always in Float64
    std::vector<int32_t> gid(2 * conBlockNum);
    std::vector<int32_t> globalIndex(2 * conBlockNum);
    std::vector<float> forceComIJ(6 * conBlockNum);
    std::vector<float> torqueComIJ(6 * conBlockNum);

    // point connectivity of line
    std::vector<int32_t> connectivity(2 * conBlockNum);
    std::vector<int32_t> offset(conBlockNum);

    // cell data for ColBlock
    std::vector<int32_t> id(conBlockNum);
    std::vector<int32_t> oneSide(conBlockNum);
    std::vector<float> sep(conBlockNum);
    std::vector<float> gamma(conBlockNum);
    std::vector<float> Stress(9 * conBlockNum);

#pragma omp parallel for
    // connectivity
    for (int i = 0; i < conBlockNum; i++) {
        connectivity[2 * i] = 2 * i;         // index of point 0 in line
        connectivity[2 * i + 1] = 2 * i + 1; // index of point 1 in line
        offset[i] = 2 * i + 2;               // offset is the end of each line. in fortran indexing
    }

#pragma omp parallel for num_threads(conQueNum)
    // data
    for (int q = 0; q < conQueNum; q++) {
        const auto &conQue = conPool[q];
        const int queSize = conQue.size();
        int conIndex = conQueIndex[q];
        for (int c = 0; c < queSize; c++) {
            const auto &con = conQue[c];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                // point position
                double labI[3];
                double labJ[3];
                con.getLabI(d, labI);
                con.getLabJ(d, labJ);

                pos[6 * conIndex + 0] = labI[0];
                pos[6 * conIndex + 1] = labI[1];
                pos[6 * conIndex + 2] = labI[2];
                pos[6 * conIndex + 3] = labJ[0];
                pos[6 * conIndex + 4] = labJ[1];
                pos[6 * conIndex + 5] = labJ[2];

                // point data
                double unscaledForceComI[3];
                double unscaledForceComJ[3];
                double unscaledTorqueComI[3];
                double unscaledTorqueComJ[3];
                const double g = con.getGamma(d);
                con.getUnscaledForceComI(d, unscaledForceComI);
                con.getUnscaledForceComJ(d, unscaledForceComJ);
                con.getUnscaledTorqueComI(d, unscaledTorqueComI);
                con.getUnscaledTorqueComJ(d, unscaledTorqueComJ);

                gid[2 * conIndex + 0] = con.gidI;
                gid[2 * conIndex + 1] = con.gidJ;
                globalIndex[2 * conIndex + 0] = con.globalIndexI;
                globalIndex[2 * conIndex + 1] = con.globalIndexJ;
                forceComIJ[6 * conIndex + 0] = unscaledForceComI[0] * g;
                forceComIJ[6 * conIndex + 1] = unscaledForceComI[1] * g;
                forceComIJ[6 * conIndex + 2] = unscaledForceComI[2] * g;
                forceComIJ[6 * conIndex + 3] = unscaledForceComJ[0] * g;
                forceComIJ[6 * conIndex + 4] = unscaledForceComJ[1] * g;
                forceComIJ[6 * conIndex + 5] = unscaledForceComJ[2] * g;
                torqueComIJ[6 * conIndex + 0] = unscaledTorqueComI[0] * g;
                torqueComIJ[6 * conIndex + 1] = unscaledTorqueComI[1] * g;
                torqueComIJ[6 * conIndex + 2] = unscaledTorqueComI[2] * g;
                torqueComIJ[6 * conIndex + 3] = unscaledTorqueComJ[0] * g;
                torqueComIJ[6 * conIndex + 4] = unscaledTorqueComJ[1] * g;
                torqueComIJ[6 * conIndex + 5] = unscaledTorqueComJ[2] * g;

                // cell data
                double stress[9];
                con.getStress(d, stress);
                id[conIndex] = con.id;
                oneSide[conIndex] = con.oneSide ? 1 : 0;
                sep[conIndex] = con.getSep(d);
                gamma[conIndex] = g;
                for (int kk = 0; kk < 9; kk++) {
                    Stress[9 * conIndex + kk] = stress[kk] * g;
                }
                conIndex += 1;
            }
        }
    }

    std::ofstream file(folder + '/' + prefix + std::string("ConBlock_") + "r" + std::to_string(rank) +
                           std::string("_") + postfix + std::string(".vtp"),
                       std::ios::out);

    IOHelper::writeHeadVTP(file);

    // Piece starts
    file << "<Piece NumberOfPoints=\"" << conBlockNum * 2 << "\" NumberOfLines=\"" << conBlockNum << "\">\n";
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
    IOHelper::writeDataArrayBase64(id, "id", 1, file);
    IOHelper::writeDataArrayBase64(oneSide, "oneSide", 1, file);
    IOHelper::writeDataArrayBase64(sep, "sep", 1, file);
    IOHelper::writeDataArrayBase64(gamma, "gamma", 1, file);
    IOHelper::writeDataArrayBase64(Stress, "Stress", 9, file);
    file << "</CellData>\n";
    // Piece ends
    file << "</Piece>\n";

    IOHelper::writeTailVTP(file);
    file.close();
}

void ConstraintCollector::dumpConstraints() const {
    std::cout << "number of constraint queues: " << constraintPoolPtr->size() << std::endl;
    // dump constraint blocks
    for (const auto &conQue : (*constraintPoolPtr)) {
        std::cout << conQue.size() << " constraints in this queue" << std::endl;
        for (const auto &con : conQue) {
            std::cout << "ID: " << con.id << " | numDOF: " << con.numDOF << std::endl;
        }
    }
}

Teuchos::RCP<TCMAT>
ConstraintCollector::buildConstraintMatrixVector(const Teuchos::RCP<const TMAP> &mobMapRcp,
                                                 const Teuchos::RCP<const TMAP> &gammaMapRcp) const {
    TEUCHOS_ASSERT(nonnull(mobMapRcp));
    TEUCHOS_ASSERT(nonnull(gammaMapRcp));

    const Teuchos::RCP<const TCOMM> commRcp = mobMapRcp->getComm();
    const auto &conPool = *constraintPoolPtr; // the constraint pool
    const int conQueNum = conPool.size();

    // TODO: buildConIndex should store the result and only update if the number of constraints has been updated
    //       Only after switching away from FDPS. FSPS requires this class have low copy overhead.
    //       A shared ptr to a vector would work around this temporarily
    // prepare 1, build the index for block queue
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);
    const int localGammaSize = conQueIndex.back();

    // step 1, count the number of entries to each row
    // each constraint block, correspoding to a gamma, occupies a row
    // 12 entries for two side constraint blocks
    // 6 entries for one side constraint blocks
    Kokkos::View<size_t *> rowPointers("rowPointers", localGammaSize + 1); // last entry is the total nnz in this matrix
    rowPointers[0] = 0;

    std::vector<int> colIndexPool(conQueNum + 1,
                                  0); // The beginning index in crs columnIndices for each constraint queue

    int rowPointerIndex = 0;
    int colIndexCount = 0;
    for (int i = 0; i < conQueNum; i++) {
        const auto &conQue = conPool[i];
        const int conNum = conQue.size();
        for (int j = 0; j < conNum; j++) {
            const auto &con = conQue[j];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                rowPointerIndex++;
                const int cBlockNNZ = (con.oneSide ? 6 : 12);
                rowPointers[rowPointerIndex] = rowPointers[rowPointerIndex - 1] + cBlockNNZ;
                colIndexCount += cBlockNNZ;
            }
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
#pragma omp parallel for num_threads(conQueNum)
    for (int threadId = 0; threadId < conQueNum; threadId++) {
        // each thread process a queue
        const auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        int kk = colIndexPool[threadId];

        for (int j = 0; j < conNum; j++) {
            const auto &con = conQue[j];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                // 6 nnz for I
                double unscaledForceComI[3];
                double unscaledTorqueComI[3];
                con.getUnscaledForceComI(d, unscaledForceComI);
                con.getUnscaledTorqueComI(d, unscaledTorqueComI);

                columnIndices[kk + 0] = 6 * con.globalIndexI;
                columnIndices[kk + 1] = 6 * con.globalIndexI + 1;
                columnIndices[kk + 2] = 6 * con.globalIndexI + 2;
                columnIndices[kk + 3] = 6 * con.globalIndexI + 3;
                columnIndices[kk + 4] = 6 * con.globalIndexI + 4;
                columnIndices[kk + 5] = 6 * con.globalIndexI + 5;
                values[kk + 0] = unscaledForceComI[0];
                values[kk + 1] = unscaledForceComI[1];
                values[kk + 2] = unscaledForceComI[2];
                values[kk + 3] = unscaledTorqueComI[0];
                values[kk + 4] = unscaledTorqueComI[1];
                values[kk + 5] = unscaledTorqueComI[2];

                kk += 6;
                if (!con.oneSide) {
                    // 6 nnz for J
                    double unscaledForceComJ[3];
                    double unscaledTorqueComJ[3];
                    con.getUnscaledForceComJ(d, unscaledForceComJ);
                    con.getUnscaledTorqueComJ(d, unscaledTorqueComJ);

                    columnIndices[kk + 0] = 6 * con.globalIndexJ;
                    columnIndices[kk + 1] = 6 * con.globalIndexJ + 1;
                    columnIndices[kk + 2] = 6 * con.globalIndexJ + 2;
                    columnIndices[kk + 3] = 6 * con.globalIndexJ + 3;
                    columnIndices[kk + 4] = 6 * con.globalIndexJ + 4;
                    columnIndices[kk + 5] = 6 * con.globalIndexJ + 5;
                    values[kk + 0] = unscaledForceComJ[0];
                    values[kk + 1] = unscaledForceComJ[1];
                    values[kk + 2] = unscaledForceComJ[2];
                    values[kk + 3] = unscaledTorqueComJ[0];
                    values[kk + 4] = unscaledTorqueComJ[1];
                    values[kk + 5] = unscaledTorqueComJ[2];
                    kk += 6;
                }
            }
        }
    }

    // step 3 prepare the partitioned column map
    // Each process owns some columns. In the map, processes share entries. (multiply-owned)
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

    // printf("local number of cols: %d on rank %d\n", colMapRcp->getNodeNumElements(), commRcp->getRank());
    // printf("local number of rows: %d on rank %d\n", mobMapRcp->getNodeNumElements(), commRcp->getRank());
    // dumpTMAP(colMapRcp,"colMap");
    // dumpTMAP(mobMapRcp,"mobMap");

    // step 4, allocate the A^Trans matrix, which maps scaled force magnatude to force vector
    Teuchos::RCP<TCMAT> AMatTransRcp =
        Teuchos::rcp(new TCMAT(gammaMapRcp, colMapRcp, rowPointers, columnIndices, values));
    AMatTransRcp->fillComplete(mobMapRcp, gammaMapRcp); // domainMap, rangeMap

    return AMatTransRcp;
}

void ConstraintCollector::updateConstraintMatrixVector(const Teuchos::RCP<TCMAT> &AMatTransRcp) const {
    TEUCHOS_ASSERT(nonnull(AMatTransRcp));

    const auto &conPool = *constraintPoolPtr; // the constraint pool
    const int conQueNum = conPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);
    const int localGammaSize = conQueIndex.back();

    // step 1, fill the values in each row
    const Teuchos::RCP<const TMAP> domainMapRcp = AMatTransRcp->getDomainMap();
    const Teuchos::RCP<const TMAP> rangeMapRcp = AMatTransRcp->getRangeMap();
    AMatTransRcp->resumeFill();

    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(conQueNum)
    for (int threadId = 0; threadId < conQueNum; threadId++) {
        // each thread process a queue
        const auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        int conIndex = conQueIndex[threadId];

        for (int j = 0; j < conNum; j++) {
            const auto &con = conQue[j];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                // 6 nnz for I
                double unscaledForceComI[3];
                double unscaledTorqueComI[3];
                con.getUnscaledForceComI(d, unscaledForceComI);
                con.getUnscaledTorqueComI(d, unscaledTorqueComI);

                const auto nnz = con.oneSide ? 6 : 12;
                GO columnIndices[nnz];
                Scalar values[nnz];

                columnIndices[0] = 6 * con.globalIndexI; // TODO: Should these be gidI or globalIndexI?
                columnIndices[1] = 6 * con.globalIndexI + 1;
                columnIndices[2] = 6 * con.globalIndexI + 2;
                columnIndices[3] = 6 * con.globalIndexI + 3;
                columnIndices[4] = 6 * con.globalIndexI + 4;
                columnIndices[5] = 6 * con.globalIndexI + 5;
                values[0] = unscaledForceComI[0];
                values[1] = unscaledForceComI[1];
                values[2] = unscaledForceComI[2];
                values[3] = unscaledTorqueComI[0];
                values[4] = unscaledTorqueComI[1];
                values[5] = unscaledTorqueComI[2];
                if (!con.oneSide) {
                    // 6 nnz for J
                    double unscaledForceComJ[3];
                    double unscaledTorqueComJ[3];
                    con.getUnscaledForceComJ(d, unscaledForceComJ);
                    con.getUnscaledTorqueComJ(d, unscaledTorqueComJ);

                    columnIndices[6] = 6 * con.globalIndexJ;
                    columnIndices[7] = 6 * con.globalIndexJ + 1;
                    columnIndices[8] = 6 * con.globalIndexJ + 2;
                    columnIndices[9] = 6 * con.globalIndexJ + 3;
                    columnIndices[10] = 6 * con.globalIndexJ + 4;
                    columnIndices[11] = 6 * con.globalIndexJ + 5;
                    values[6] = unscaledForceComJ[0];
                    values[7] = unscaledForceComJ[1];
                    values[8] = unscaledForceComJ[2];
                    values[9] = unscaledTorqueComJ[0];
                    values[10] = unscaledTorqueComJ[1];
                    values[11] = unscaledTorqueComJ[2];
                }

                // update the values using global indices
                // replaceGlobalValues (const GlobalOrdinal globalRow, const LocalOrdinal numEnt, const Scalar vals[],
                // const GlobalOrdinal cols[])
                const auto globalRowIdx = rangeMapRcp->getGlobalElement(conIndex);
                AMatTransRcp->replaceGlobalValues(globalRowIdx, nnz, values, columnIndices);
                conIndex += 1;
            }
        }
    }
    AMatTransRcp->fillComplete(domainMapRcp, rangeMapRcp);
}

int ConstraintCollector::fillFixedConstraintInfo(const Teuchos::RCP<TV> &gammaGuessRcp, const Teuchos::RCP<TV> &biFlagRcp, 
                                                 const Teuchos::RCP<TV> &initialSepRcp, const Teuchos::RCP<TV> &constraintDiagonalRcp) const {
    TEUCHOS_ASSERT(nonnull(gammaGuessRcp));
    TEUCHOS_ASSERT(nonnull(biFlagRcp));
    TEUCHOS_ASSERT(nonnull(initialSepRcp));
    TEUCHOS_ASSERT(nonnull(constraintDiagonalRcp));

    const auto &conPool = *constraintPoolPtr; // the constraint pool
    const int conQueNum = conPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);

    //  fill constraintDiagonal
    auto gammaGuessPtr = gammaGuessRcp->getLocalView<Kokkos::HostSpace>();
    auto biFlagPtr = biFlagRcp->getLocalView<Kokkos::HostSpace>();
    auto initialSepPtr = initialSepRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintDiagonalPtr = constraintDiagonalRcp->getLocalView<Kokkos::HostSpace>();
    gammaGuessRcp->modify<Kokkos::HostSpace>();
    biFlagRcp->modify<Kokkos::HostSpace>();
    initialSepRcp->modify<Kokkos::HostSpace>();
    constraintDiagonalRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(conQueNum)
    for (int threadId = 0; threadId < conQueNum; threadId++) {
        const auto &conQue = conPool[threadId];
        const int queSize = conQue.size();
        int conIndex = conQueIndex[threadId];
        for (int j = 0; j < queSize; j++) {
            const auto &con = conQue[j];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                gammaGuessPtr(conIndex, 0) = con.getGammaGuess(d);
                biFlagPtr(conIndex, 0) = con.bilaterial;
                initialSepPtr(conIndex, 0) = con.getSep(d);
                constraintDiagonalPtr(conIndex, 0) = con.diagonal;
                conIndex += 1;
            }
        }
    }
    return 0;
}

int ConstraintCollector::evalConstraintValues(const Teuchos::RCP<const TV> &gammaRcp,
                                              const Teuchos::RCP<const TV> &scaleRcp,
                                              const Teuchos::RCP<const TV> &constraintSepRcp,
                                              const Teuchos::RCP<TV> &constraintValueRcp,
                                              const Teuchos::RCP<TV> &constraintStatusRcp) const {
    TEUCHOS_ASSERT(nonnull(gammaRcp));
    TEUCHOS_ASSERT(nonnull(scaleRcp));
    TEUCHOS_ASSERT(nonnull(constraintSepRcp));
    TEUCHOS_ASSERT(nonnull(constraintValueRcp));
    TEUCHOS_ASSERT(nonnull(constraintStatusRcp));

    auto &conPool = *constraintPoolPtr; // the constraint pool
    const int conQueNum = conPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);

    //  fill all
    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    auto scalePtr = scaleRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintSepPtr = constraintSepRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintValuePtr = constraintValueRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintStatusPtr = constraintStatusRcp->getLocalView<Kokkos::HostSpace>();
    constraintValueRcp->modify<Kokkos::HostSpace>();
    constraintStatusRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(conQueNum)
    for (int threadId = 0; threadId < conQueNum; threadId++) {
        auto &conQue = conPool[threadId];
        const int queSize = conQue.size();
        int conIndex = conQueIndex[threadId];
        for (int j = 0; j < queSize; j++) {
            auto &con = conQue[j];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                // get the value and status
                constraintValuePtr(conIndex, 0) = con.getValue(con, constraintSepPtr(conIndex, 0), gammaPtr(conIndex, 0));
                constraintStatusPtr(conIndex, 0) =
                    con.isConstrained(con, constraintSepPtr(conIndex, 0), gammaPtr(conIndex, 0));

                // scaling is applied to inactive constraints
                if (constraintStatusPtr(conIndex, 0) < 0.5) {
                    constraintValuePtr(conIndex, 0) = constraintValuePtr(conIndex, 0) * scalePtr(conIndex, 0);
                }

                conIndex += 1;
            }
        }
    }

    return 0;
}

int ConstraintCollector::evalConstraintValues(const Teuchos::RCP<const TV> &gammaRcp,
                                              const Teuchos::RCP<const TV> &constraintSepRcp,
                                              const Teuchos::RCP<TV> &constraintValueRcp) const {
    TEUCHOS_ASSERT(nonnull(gammaRcp));
    TEUCHOS_ASSERT(nonnull(constraintSepRcp));
    TEUCHOS_ASSERT(nonnull(constraintValueRcp));

    auto &conPool = *constraintPoolPtr; // the constraint pool
    const int conQueNum = conPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);

    //  fill all
    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintSepPtr = constraintSepRcp->getLocalView<Kokkos::HostSpace>();
    auto constraintValuePtr = constraintValueRcp->getLocalView<Kokkos::HostSpace>();
    constraintValueRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(conQueNum)
    for (int threadId = 0; threadId < conQueNum; threadId++) {
        auto &conQue = conPool[threadId];
        const int queSize = conQue.size();
        int conIndex = conQueIndex[threadId];
        for (int j = 0; j < queSize; j++) {
            auto &con = conQue[j];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                // get the value only
                constraintValuePtr(conIndex, 0) = con.getValue(con, constraintSepPtr(conIndex, 0), gammaPtr(conIndex, 0));
                conIndex += 1;
            }
        }
    }

    return 0;
}


int ConstraintCollector::writeBackConstraintVariables(const Teuchos::RCP<const TV> &gammaRcp, const Teuchos::RCP<const TV> &sepRcp) {
    TEUCHOS_ASSERT(nonnull(gammaRcp));
    TEUCHOS_ASSERT(nonnull(sepRcp));

    auto &conPool = *constraintPoolPtr;
    const int conQueNum = conPool.size();

    // prepare 1, build the index for block queue
    std::vector<int> conQueSize;
    std::vector<int> conQueIndex;
    buildConIndex(conQueSize, conQueIndex);

    auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();
    auto sepPtr = sepRcp->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for num_threads(conQueNum)
    for (int threadId = 0; threadId < conQueNum; threadId++) {
        // each thread process a queue
        auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        int conIndex = conQueIndex[threadId];

        for (int j = 0; j < conNum; j++) {
            auto &con = conQue[j];
            const int numDOF = con.numDOF;
            for (int d = 0; d < numDOF; d++) {
                // get the current gamma //TODO add in an alpha beta 
                const double gamma = con.getGamma(d);

                // store gamma and sep 
                // TODO: is there a better place to update gammaGuess?
                con.setGammaGuess(d, 0.0);
                con.setGamma(d, gamma + gammaPtr(conIndex, 0));
                con.setSep(d, sepPtr(conIndex, 0));
                conIndex += 1;

            }
        }
    }
    return 0;
}

int ConstraintCollector::buildConIndex(std::vector<int> &conQueSize, std::vector<int> &conQueIndex) const {
    const auto &conPool = *constraintPoolPtr;

    // multi-thread filling. nThreads = poolSize, each thread process a queue
    const int conQueNum = conPool.size();
    conQueSize.resize(conQueNum);
    conQueIndex.resize(conQueNum + 1);
#pragma omp parallel for num_threads(conQueNum)
    for (int threadId = 0; threadId < conQueNum; threadId++) {
        // each thread process a queue
        const auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        for (int j = 0; j < conNum; j++) {
            conQueSize[threadId] += conQue[j].numDOF;
        }
    }

    for (int i = 1; i < conQueNum + 1; i++) {
        conQueIndex[i] = conQueSize[i - 1] + conQueIndex[i - 1];
    }
    return 0;
}
