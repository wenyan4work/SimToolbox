#include "ConstraintCollector.hpp"

ConstraintCollector::ConstraintCollector() {
    const int totalThreads = omp_get_max_threads();
    constraintPoolPtr = std::make_shared<ConstraintBlockPool>();
    constraintPoolPtr->resize(totalThreads);
    for (auto &queue : *constraintPoolPtr) {
        queue.resize(0);
        queue.reserve(50);
    }
    std::cout << "ConstraintCollector constructed for:" << constraintPoolPtr->size() << " threads" << std::endl;
}

void ConstraintCollector::sumLocalConstraintStress(Emat3 &stress, bool withOneSide = false) const {
    const auto &cPool = *constraintPoolPtr;
    const int poolSize = cPool.size();

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

void ConstraintCollector::writeVTP(const std::string &folder, const std::string &prefix, const std::string &postfix,
                                   int rank) const {
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
