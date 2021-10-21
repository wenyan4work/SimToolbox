#include "ConstraintCollector.hpp"
#include "spdlog/spdlog.h"

#include <set>

ConstraintCollector::ConstraintCollector() {
  const int totalThreads = omp_get_max_threads();
  constraintPoolPtr = std::make_shared<ConstraintBlockPool>();
  constraintPoolPtr->resize(totalThreads);
  for (auto &queue : *constraintPoolPtr) {
    queue.clear();
  }

  spdlog::debug("ConstraintCollector constructed for {} threads",
                constraintPoolPtr->size());
}

void ConstraintCollector::clear() {
  assert(constraintPoolPtr);
  for (auto &que : (*constraintPoolPtr)) {
    que.clear();
  }

  // keep the total number of queues
  const auto totalThreads = omp_get_max_threads();
  constraintPoolPtr->resize(totalThreads);
}

int ConstraintCollector::getLocalNumberOfConstraints() {
  int sum = 0;
  for (auto &que : (*constraintPoolPtr)) {
    sum += que.size();
  }
  return sum;
}

void ConstraintCollector::sumLocalConstraintStress(Emat3 &uniStress,
                                                   Emat3 &biStress,
                                                   bool withOneSide) const {
  const auto &cPool = *constraintPoolPtr;
  const auto poolSize = cPool.size();

  Emat3 biStressTotal = Emat3::Zero();
  Emat3 uniStressTotal = Emat3::Zero();

#pragma omp parallel for
  for (long que = 0; que < poolSize; que++) {
    // reduction of stress, one que for each thread

    Emat3 uniStressSumQue = Emat3::Zero();
    Emat3 biStressSumQue = Emat3::Zero();
    const auto &conQue = cPool[que];
    for (const auto &cBlock : conQue) {
      if (cBlock.oneSide && !withOneSide) {
        // skip counting oneside collision blocks
        continue;
      } else {
        Emat3 stressBlock = Eigen::Map<const Emat3>(cBlock.getStress());
        if (cBlock.bilateral) {
          //   biStressSumQue = biStressSumQue + stressBlock;
          biStressSumQue += stressBlock;
        } else {
          //   uniStressSumQue = uniStressSumQue + stressBlock;
          uniStressSumQue += stressBlock;
        }
      }
    }
#pragma omp critical
    {
      //   biStressTotal = biStressTotal + biStressSumQue;
      //   uniStressTotal = uniStressTotal + uniStressSumQue;
      biStressTotal += biStressSumQue;
      uniStressTotal += uniStressSumQue;
    }
  }
  uniStress = uniStressTotal;
  biStress = biStressTotal;
}

void ConstraintCollector::writePVTP(const std::string &folder,
                                    const std::string &prefix,
                                    const std::string &postfix,
                                    const int nProcs) const {
  //   std::vector<std::string> pieceNames;

  //   std::vector<IOHelper::FieldVTU> pointDataFields;
  //   pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int64, "gid");
  //   pointDataFields.emplace_back(1, IOHelper::IOTYPE::Int64, "globalIndex");
  //   pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "posIJ");
  //   pointDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "normIJ");

  //   std::vector<IOHelper::FieldVTU> cellDataFields;
  //   cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "oneSide");
  //   cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "bilateral");
  //   cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "delta0");
  //   cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "gamma");
  //   cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "kappa");
  //   cellDataFields.emplace_back(9, IOHelper::IOTYPE::Float32, "Stress");

  //   for (int i = 0; i < nProcs; i++) {
  //     pieceNames.emplace_back(prefix + std::string("ConBlock_") +
  //                             std::string("r") + std::to_string(i) + "_" +
  //                             postfix + ".vtp");
  //   }

  //   IOHelper::writePVTPFile(folder + "/" + prefix + std::string("ConBlock_")
  //   +
  //                               postfix + ".pvtp",
  //                           pointDataFields, cellDataFields, pieceNames);
}

void ConstraintCollector::writeVTP(const std::string &folder,
                                   const std::string &prefix,
                                   const std::string &postfix, int rank) const {
  //   const auto &cPool = *constraintPoolPtr;
  //   const int cQueNum = cPool.size();
  //   std::vector<int> cQueSize;
  //   std::vector<int> cQueIndex;
  //   buildConIndex(cQueSize, cQueIndex);
  //   const int cBlockNum = cQueIndex.back();

  //   // for each block:

  //   // write VTP for basic data, use float to save some space
  //   // point and point data
  //   std::vector<double> pos(6 * cBlockNum); // position always in Float64
  //   std::vector<long> gid(2 * cBlockNum);
  //   std::vector<long> globalIndex(2 * cBlockNum);
  //   std::vector<float> posIJ(6 * cBlockNum);
  //   std::vector<float> normIJ(6 * cBlockNum);

  //   // point connectivity of line
  //   std::vector<int32_t> connectivity(2 * cBlockNum);
  //   std::vector<int32_t> offset(cBlockNum);

  //   // cell data for ColBlock
  //   std::vector<int32_t> oneSide(cBlockNum);
  //   std::vector<int32_t> bilateral(cBlockNum);
  //   std::vector<float> delta0(cBlockNum);
  //   std::vector<float> gamma(cBlockNum);
  //   std::vector<float> kappa(cBlockNum);
  //   std::vector<float> Stress(9 * cBlockNum);

  // #pragma omp parallel for
  //   // connectivity
  //   for (long i = 0; i < cBlockNum; i++) {
  //     connectivity[2 * i] = 2 * i;         // index of point 0 in line
  //     connectivity[2 * i + 1] = 2 * i + 1; // index of point 1 in line
  //     offset[i] =
  //         2 * i + 2; // offset is the end of each line. in fortran indexing
  //   }

  // #pragma omp parallel for
  //   // data
  //   for (long q = 0; q < cQueNum; q++) {
  //     const int queSize = cQueSize[q];
  //     const int queIndex = cQueIndex[q];
  //     for (long c = 0; c < queSize; c++) {
  //       const int cIndex = queIndex + c;
  //       const auto &block = cPool[q][c];
  //       // point position
  //       pos[6 * cIndex + 0] = block.labI[0];
  //       pos[6 * cIndex + 1] = block.labI[1];
  //       pos[6 * cIndex + 2] = block.labI[2];
  //       pos[6 * cIndex + 3] = block.labJ[0];
  //       pos[6 * cIndex + 4] = block.labJ[1];
  //       pos[6 * cIndex + 5] = block.labJ[2];
  //       // point data
  //       gid[2 * cIndex + 0] = block.gidI;
  //       gid[2 * cIndex + 1] = block.gidJ;
  //       globalIndex[2 * cIndex + 0] = block.globalIndexI;
  //       globalIndex[2 * cIndex + 1] = block.globalIndexJ;
  //       posIJ[6 * cIndex + 0] = block.posI[0];
  //       posIJ[6 * cIndex + 1] = block.posI[1];
  //       posIJ[6 * cIndex + 2] = block.posI[2];
  //       posIJ[6 * cIndex + 3] = block.posJ[0];
  //       posIJ[6 * cIndex + 4] = block.posJ[1];
  //       posIJ[6 * cIndex + 5] = block.posJ[2];
  //       normIJ[6 * cIndex + 0] = block.normI[0];
  //       normIJ[6 * cIndex + 1] = block.normI[1];
  //       normIJ[6 * cIndex + 2] = block.normI[2];
  //       normIJ[6 * cIndex + 3] = block.normJ[0];
  //       normIJ[6 * cIndex + 4] = block.normJ[1];
  //       normIJ[6 * cIndex + 5] = block.normJ[2];
  //       // cell data
  //       oneSide[cIndex] = block.oneSide ? 1 : 0;
  //       bilateral[cIndex] = block.bilateral ? 1 : 0;
  //       delta0[cIndex] = block.delta0;
  //       gamma[cIndex] = block.gamma;
  //       kappa[cIndex] = block.kappa;
  //       for (long kk = 0; kk < 9; kk++) {
  //         Stress[9 * cIndex + kk] = block.stress[kk];
  //       }
  //     }
  //   }

  //   std::ofstream file(folder + '/' + prefix + std::string("ConBlock_") + "r"
  //   +
  //                          std::to_string(rank) + std::string("_") + postfix
  //                          + std::string(".vtp"),
  //                      std::ios::out);

  //   IOHelper::writeHeadVTP(file);

  //   // Piece starts
  //   file << "<Piece NumberOfPoints=\"" << cBlockNum * 2 << "\"
  //   NumberOfLines=\""
  //        << cBlockNum << "\">\n";
  //   // Points
  //   file << "<Points>\n";
  //   IOHelper::writeDataArrayBase64(pos, "position", 3, file);
  //   file << "</Points>\n";
  //   // cell definition
  //   file << "<Lines>\n";
  //   IOHelper::writeDataArrayBase64(connectivity, "connectivity", 1, file);
  //   IOHelper::writeDataArrayBase64(offset, "offsets", 1, file);
  //   file << "</Lines>\n";
  //   // point data
  //   file << "<PointData Scalars=\"scalars\">\n";
  //   IOHelper::writeDataArrayBase64(gid, "gid", 1, file);
  //   IOHelper::writeDataArrayBase64(globalIndex, "globalIndex", 1, file);
  //   IOHelper::writeDataArrayBase64(posIJ, "posIJ", 3, file);
  //   IOHelper::writeDataArrayBase64(normIJ, "normIJ", 3, file);
  //   file << "</PointData>\n";
  //   // cell data
  //   file << "<CellData Scalars=\"scalars\">\n";
  //   IOHelper::writeDataArrayBase64(oneSide, "oneSide", 1, file);
  //   IOHelper::writeDataArrayBase64(bilateral, "bilateral", 1, file);
  //   IOHelper::writeDataArrayBase64(delta0, "delta0", 1, file);
  //   IOHelper::writeDataArrayBase64(gamma, "gamma", 1, file);
  //   IOHelper::writeDataArrayBase64(kappa, "kappa", 1, file);
  //   IOHelper::writeDataArrayBase64(Stress, "Stress", 9, file);
  //   file << "</CellData>\n";
  //   // Piece ends
  //   file << "</Piece>\n";

  //   IOHelper::writeTailVTP(file);
  //   file.close();
}

void ConstraintCollector::dumpBlocks() const {
  std::cout << "number of collision queues: " << constraintPoolPtr->size()
            << std::endl;
  // dump constraint blocks
  for (const auto &blockQue : (*constraintPoolPtr)) {
    std::cout << blockQue.size() << " constraints in this queue" << std::endl;
    for (const auto &block : blockQue) {
      std::cout << block.globalIndexI << " " << block.globalIndexJ
                << "  delta0:" << block.delta0 << std::endl;
    }
  }
}

int ConstraintCollector::buildConstraintMatrixVector(
    const Teuchos::RCP<const TMAP> &mobMapRcp, //
    Teuchos::RCP<TCMAT> &DMatTransRcp,         //
    Teuchos::RCP<TV> &delta0Rcp,               //
    Teuchos::RCP<TV> &invKappaRcp,             //
    Teuchos::RCP<TV> &biFlagRcp,               //
    Teuchos::RCP<TV> &gammaGuessRcp) const {

  Teuchos::RCP<const TCOMM> commRcp = mobMapRcp->getComm();

  const auto &cPool = *constraintPoolPtr; // the constraint pool
  const auto cPoolSize = cPool.size();

  // prepare 1, build the index for block queue
  const auto &cQueOffset = buildConIndex();

  // prepare 2, allocate the map and vectors
  const auto localGammaSize = cQueOffset.back();
  const auto &gammaMapRcp = getTMAPFromLocalSize(localGammaSize, commRcp);

  // prepare 3, build rowmap and colmap
  const auto &rowMapRcp = gammaMapRcp;

  std::set<long> globalParticleIndexOnLocal;
  for (const auto &que : cPool) {
    for (const auto &blk : que) {
      globalParticleIndexOnLocal.insert(blk.globalIndexI);
      if (!blk.oneSide)
        globalParticleIndexOnLocal.insert(blk.globalIndexJ);
    }
  }
  std::vector<TGO> globalIndexOnLocal;
  globalIndexOnLocal.reserve(globalParticleIndexOnLocal.size() * 6);
  for (const auto &k : globalParticleIndexOnLocal) {
    for (int i = 0; i < 6; i++)
      globalIndexOnLocal.push_back(6 * k + i);
  }
  const auto &colMapRcp = getTMAPFromGlobalIndexOnLocal(
      globalIndexOnLocal, mobMapRcp->getGlobalNumElements(), commRcp);

  // step 1, count the number of entries to each row
  // each constraint block, correspoding to a gamma, occupies a row
  // 12 entries for two side constraint blocks
  // 6 entries for one side constraint blocks
  // last entry is the total nnz in this matrix
  Kokkos::View<TLRO *> rowOffset("localRowOffset", localGammaSize + 1);
  rowOffset[0] = 0;

  // pass 1, fill number of nnz for each row in rowOffset
  std::vector<int> colIndexOffset(cPoolSize + 1, 0);
#pragma omp parallel for num_threads(cPoolSize)
  for (int i = 0; i < cPoolSize; i++) {
    const auto &que = cPool[i];
    const auto jsize = que.size();
    const auto jbase = cQueOffset[i];
    int nnz = 0;
    for (int j = 0; j < jsize; j++) {
      const int cBlockNNZ = (que[j].oneSide ? 6 : 12);
      rowOffset[1 + jbase + j] = cBlockNNZ;
      nnz += cBlockNNZ;
    }
    colIndexOffset[i + 1] = nnz;
  }

  // get colIndexOffset for each que, [0, nnz1, nnz1+nnz2, ....]
  std::partial_sum(colIndexOffset.begin(), colIndexOffset.end(),
                   colIndexOffset.begin());

  // pass 2, inclusive prefix sum to get rowOffset
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,
                                                         rowOffset.extent(0)),
      KOKKOS_LAMBDA(const int i, TLRO &update_value, const bool final) {
        const auto val_i = rowOffset(i);
        update_value += val_i;
        if (final)
          rowOffset(i) = update_value;
      });

  // step 2, fill the values to each row
  Kokkos::View<TLO *> colIndices("localColumnIndices", colIndexOffset.back());
  Kokkos::View<double *> values("values", colIndexOffset.back());

  // multi-thread filling. nThreads = poolSize, each thread process a queue
  const auto &colMap = *colMapRcp;
#pragma omp parallel for num_threads(cPoolSize)
  for (int i = 0; i < cPoolSize; i++) {
    // each thread process a queue
    const auto &cQue = cPool[i];
    int kk = colIndexOffset[i];
    for (const auto &cBlk : cQue) {
      // each 6nnz for an object:
      // gx.ux+gy.uy+gz.uz+(gzpy-gypz)wx+(gxpz-gzpx)wy+(gypx-gxpy)wz 6 nnz for
      const TGO glbIdxBase = 6 * cBlk.globalIndexI;
      for (int l = 0; l < 6; l++) {
        colIndices[kk + l] = colMap.getLocalElement(glbIdxBase + l);
      }
      const double &gx = cBlk.normI[0];
      const double &gy = cBlk.normI[1];
      const double &gz = cBlk.normI[2];
      const double &px = cBlk.posI[0];
      const double &py = cBlk.posI[1];
      const double &pz = cBlk.posI[2];
      values[kk + 0] = gx;
      values[kk + 1] = gy;
      values[kk + 2] = gz;
      values[kk + 3] = (gz * py - gy * pz);
      values[kk + 4] = (gx * pz - gz * px);
      values[kk + 5] = (gy * px - gx * py);
      kk += 6;
      if (!cBlk.oneSide) {
        const TGO glbIdxBase = 6 * cBlk.globalIndexJ;
        for (int l = 0; l < 6; l++) {
          colIndices[kk + l] = colMap.getLocalElement(glbIdxBase + l);
        }
        const double &gx = cBlk.normJ[0];
        const double &gy = cBlk.normJ[1];
        const double &gz = cBlk.normJ[2];
        const double &px = cBlk.posJ[0];
        const double &py = cBlk.posJ[1];
        const double &pz = cBlk.posJ[2];
        values[kk + 0] = gx;
        values[kk + 1] = gy;
        values[kk + 2] = gz;
        values[kk + 3] = (gz * py - gy * pz);
        values[kk + 4] = (gx * pz - gz * px);
        values[kk + 5] = (gy * px - gx * py);
        kk += 6;
      }
    }
    if (kk != colIndexOffset[i + 1]) {
      spdlog::error("kk mismatch {}", i);
      std::exit(1);
    }
  }

  // step 4, allocate the D^Trans matrix
  DMatTransRcp = Teuchos::rcp(
      new TCMAT(rowMapRcp, colMapRcp, rowOffset, colIndices, values));
  DMatTransRcp->fillComplete(mobMapRcp, gammaMapRcp); // domainMap, rangeMap

  // step 5, fill the delta0, gammaGuess, invKappa, conFlag vectors
  delta0Rcp = Teuchos::rcp(new TV(gammaMapRcp, true));
  invKappaRcp = Teuchos::rcp(new TV(gammaMapRcp, true));
  biFlagRcp = Teuchos::rcp(new TV(gammaMapRcp, true));
  gammaGuessRcp = Teuchos::rcp(new TV(gammaMapRcp, true));
  auto delta0 = delta0Rcp->getLocalView<Kokkos::HostSpace>();
  auto gammaGuess = gammaGuessRcp->getLocalView<Kokkos::HostSpace>();
  auto invKappa = invKappaRcp->getLocalView<Kokkos::HostSpace>();
  auto biFlag = biFlagRcp->getLocalView<Kokkos::HostSpace>();
  delta0Rcp->modify<Kokkos::HostSpace>();
  gammaGuessRcp->modify<Kokkos::HostSpace>();
  invKappaRcp->modify<Kokkos::HostSpace>();
  biFlagRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for num_threads(cPoolSize)
  for (int i = 0; i < cPoolSize; i++) {
    const auto &cQue = cPool[i];
    const int cIndexBase = cQueOffset[i];
    const int queSize = cQue.size();
    for (int j = 0; j < queSize; j++) {
      const auto &block = cQue[j];
      const auto idx = cIndexBase + j;
      delta0(idx, 0) = block.delta0;
      gammaGuess(idx, 0) = block.gamma;
      if (block.bilateral) {
        invKappa(idx, 0) = block.kappa > 0 ? 1 / block.kappa : 0;
        biFlag(idx, 0) = 1;
      }
    }
  }

  return 0;
}

std::vector<int> ConstraintCollector::buildConQueOffset() const {

  const auto &cPool = *constraintPoolPtr;

  std::vector<int> cQueOffset(cPool.size() + 1, 0);
  for (int i = 1; i < cPool.size() + 1; ++i) {
    cQueOffset[i] = cPool[i - 1].size();
  }
  std::partial_sum(cQueOffset.begin(), cQueOffset.end(), cQueOffset.begin());

  return cQueOffset;
}

int ConstraintCollector::writeBackGamma(
    const Teuchos::RCP<const TV> &gammaRcp) {

  auto gammaPtr = gammaRcp->getLocalView<Kokkos::HostSpace>();

  const auto &cQueOffset = buildConIndex();
  TEUCHOS_ASSERT(cQueOffset.back() == gammaPtr.extent(0));

  auto &cPool = *constraintPoolPtr; // the constraint pool
  const int cQueNum = cPool.size();
#pragma omp parallel for num_threads(cQueNum)
  for (int i = 0; i < cQueNum; i++) {
    const int cQueSize = cPool[i].size();
    for (int j = 0; j < cQueSize; j++) {
      cPool[i][j].gamma = gammaPtr(cQueOffset[i] + j, 0);
      for (int k = 0; k < 9; k++) {
        cPool[i][j].stress[k] *= cPool[i][j].gamma;
      }
    }
  }

  return 0;
}