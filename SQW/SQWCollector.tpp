#include "Trilinos/TpetraUtil.hpp"
#include "Util/SpecialQuadWeights.hpp"

#include <cstdlib>

template <int N>
SQWCollector<N>::SQWCollector(const double sqwbuf_) : sqwbuf(sqwbuf_) {
    const int totalThreads = omp_get_max_threads();
    sqwPoolPtr = std::make_shared<SQWBlockPool<N>>();
    weightPoolPtr = std::make_shared<WeightBlockPool>();
    sqwPoolPtr->resize(totalThreads);
    weightPoolPtr->resize(totalThreads);
    for (auto &queue : *sqwPoolPtr) {
        queue.resize(0);
        queue.reserve(50);
    }
    for (auto &queue : *weightPoolPtr) {
        queue.resize(0);
        queue.reserve(50);
    }
    std::cout << "SQWCollector constructed for:" << sqwPoolPtr->size() << " threads" << std::endl;
}

template <int N>
bool SQWCollector<N>::valid() const {
    return sqwPoolPtr->empty();
}

template <int N>
void SQWCollector<N>::clear() {
    assert(sqwPoolPtr);
    for (int i = 0; i < sqwPoolPtr->size(); i++) {
        (*sqwPoolPtr)[i].clear();
    }

    // keep the total number of queues
    const int totalThreads = omp_get_max_threads();
    sqwPoolPtr->resize(totalThreads);
}

template <int N>
int SQWCollector<N>::getLocalOverallSQWQueSize() {
    int sum = 0;
    for (int i = 0; i < sqwPoolPtr->size(); i++) {
        sum += (*sqwPoolPtr)[i].size();
    }
    return sum;
}

template <int N>
int SQWCollector<N>::getLocalOverallWeightQueSize() {
    int sum = 0;
    for (int i = 0; i < weightPoolPtr->size(); i++) {
        sum += (*weightPoolPtr)[i].size();
    }
    return sum;
}

template <int N>
void SQWCollector<N>::dumpSQWBlocks() const {
    std::cout << "number of sqw queues: " << sqwPoolPtr->size() << std::endl;
    // dump sqw blocks
    for (const auto &blockQue : (*sqwPoolPtr)) {
        std::cout << blockQue.size() << " rods in this queue" << std::endl;
        for (const auto &block : blockQue) {
            std::cout << block.globalIndexI << " " << block.globalIndexJ << std::endl;
        }
    }
}

template <int N>
void SQWCollector<N>::dumpWeightData() const {
    std::cout << "number of weight queues: " << weightPoolPtr->size() << std::endl;
    // dump weight blocks
    for (const auto &weightQue : (*weightPoolPtr)) {
        std::cout << weightQue.size() << " rods in this queue" << std::endl;
        for (const auto &block : weightQue) {
            std::cout << block.lidQuadPtI << " " << block.lidQuadPtJ << std::endl;
        }
    }
    std::cout << "----------------------------" << std::endl;
}

template <int N>
void SQWCollector<N>::buildWeightPool(const Teuchos::RCP<TMAP> &rodMapRcp, const std::vector<int> &rodPtsIndex,
                                      const Config &cellConfig) {
    const int nPtsLocal = rodPtsIndex.back(); // number of quadrature points on local
    nSourcePerTarget.resize(6 * nPtsLocal);

    const auto &sqwPool = *sqwPoolPtr; // the special quadrature weight pool
    const int sQueNum = sqwPool.size();

    // multi-thread filling. nThreads = poolSize, each thread processes a queue
#pragma omp parallel for num_threads(sQueNum)
    for (int threadId = 0; threadId < sQueNum; threadId++) {
        // each thread processes a queue
        const auto &sqwBlockQue = sqwPool[threadId];
        const int sqwBlockNum = sqwBlockQue.size();
        auto &weightQue = (*weightPoolPtr)[threadId];

        for (int sqwBlockIdx = 0; sqwBlockIdx < sqwBlockNum; sqwBlockIdx++) {
            // extract the necessary information
            const int gidI = sqwBlockQue[sqwBlockIdx].gidI;
            const int gidJ = sqwBlockQue[sqwBlockIdx].gidJ;
            const double lineHalfLengthI = sqwBlockQue[sqwBlockIdx].lengthI / 2;
            const double lineHalfLengthJ = sqwBlockQue[sqwBlockIdx].lengthI / 2;
            const int speciesIDI = sqwBlockQue[sqwBlockIdx].speciesIDI;
            const int speciesIDJ = sqwBlockQue[sqwBlockIdx].speciesIDJ;
            const Evec3 posI = ECmap3(sqwBlockQue[sqwBlockIdx].posI);
            const Evec3 posJ = ECmap3(sqwBlockQue[sqwBlockIdx].posJ);
            const Evec3 dirI = ECmap3(sqwBlockQue[sqwBlockIdx].directionI);
            const Evec3 dirJ = ECmap3(sqwBlockQue[sqwBlockIdx].directionJ);
            const int numQuadPtI = sqwBlockQue[sqwBlockIdx].quadPtrI->getSize();
            const int numQuadPtJ = sqwBlockQue[sqwBlockIdx].quadPtrJ->getSize();
            const auto &sQuadPtI = sqwBlockQue[sqwBlockIdx].quadPtrI->getPoints();
            const auto &sQuadPtJ = sqwBlockQue[sqwBlockIdx].quadPtrJ->getPoints();

            // setup data for quadrature points according to cell species
            std::vector<double> radiusPt(numQuadPtI, 0);
            const Hydro *pHydro = &(cellConfig.cellHydro.at(speciesIDI));
            const auto &table = pHydro->cellHydroTable;
            table.getValueRadius(numQuadPtI, sqwBlockQue[sqwBlockIdx].quadPtrI->getPoints(), radiusPt.data());

            // loop over all combinations of source and target quadrature point
            SpecialQuadWeights<32> sqw(numQuadPtI); // recommended max num of quad pts is 32
            for (int iQuadPtJ = 0; iQuadPtJ < numQuadPtJ; iQuadPtJ++) {
                Evec3 locQuadPtJ = posJ + (lineHalfLengthJ * sQuadPtJ[iQuadPtJ]) * dirJ;

                // check if special quad is unnecessary for each target quadrature point
                double rcheck = (locQuadPtJ - posI).norm();
                if (rcheck > 2 * sqwbuf * lineHalfLengthI) {
                    continue;
                }
                // calculate the SQW and GL weights only if necessary
                sqw.calcWeights(lineHalfLengthI, posI.data(), locQuadPtJ.data(), dirI.data());
                const double *w1_all = sqw.getWeights1(); // integral from [-1,1]
                const double *w3_all = sqw.getWeights3();
                const double *w5_all = sqw.getWeights5();
                const auto &wGL_all = sqwBlockQue[sqwBlockIdx].quadPtrJ->getWeights();

                // extract the target ID and increment the target count by 3*numQuadPtI
                int lidJ = rodMapRcp->getLocalElement(gidJ);
                int lidQuadPtJ = rodPtsIndex[lidJ] + iQuadPtJ;
                nSourcePerTarget[6 * lidQuadPtJ] += 3 * numQuadPtI;
                nSourcePerTarget[6 * lidQuadPtJ + 1] += 3 * numQuadPtI;
                nSourcePerTarget[6 * lidQuadPtJ + 2] += 3 * numQuadPtI;
                nSourcePerTarget[6 * lidQuadPtJ + 3] += 3 * numQuadPtI;
                nSourcePerTarget[6 * lidQuadPtJ + 4] += 3 * numQuadPtI;
                nSourcePerTarget[6 * lidQuadPtJ + 5] += 3 * numQuadPtI;

                // store the weights for each source-target point pair
                for (int iQuadPtI = 0; iQuadPtI < numQuadPtI; iQuadPtI++) {
                    // extract the source and target local ID
                    int lidI = rodMapRcp->getLocalElement(gidI);
                    int lidQuadPtI = rodPtsIndex[lidI] + iQuadPtI;

                    // calculate and store the weights for each source-target pair
                    const Evec3 locQuadPtI = posI + (lineHalfLengthI * sQuadPtI[iQuadPtI]) * dirI;
                    const Evec3 rIJ = locQuadPtJ - locQuadPtI;

                    weightQue.emplace_back(lidQuadPtI, lidQuadPtJ, radiusPt[iQuadPtI], lineHalfLengthI,
                                           w1_all[iQuadPtI], w3_all[iQuadPtI], w5_all[iQuadPtI], wGL_all[iQuadPtI],
                                           rIJ.data());
                }
            }
        }
    }

    printf("WeightPool constructed\n");
}

template <int N>
void SQWCollector<N>::buildSQWMobilityMatrix(const Teuchos::RCP<TMAP> &pointValuesMapRcp,
                                             const Teuchos::RCP<TMAP> &targetValuesMapRcp,
                                             Teuchos::RCP<TCMAT> &sqwMobilityMatrixRcp) {
    // sparce near rod hydro mobility with precomputed RPY values
    // domain space: 3*nSQWGlobal
    // range space: 6*nSQWGlobal
    // currently, each rank occupies sequential rows and owns the entire column
    const Teuchos::ArrayView<const size_t> numEntPerRowToAlloc(nSourcePerTarget);

    // SwqMobMat has contiguous row partitioning and no column partitioning
    // This constructor does allow a column map to be provided. Use this to optimize memory management
    sqwMobilityMatrixRcp =
        Teuchos::rcp(new TCMAT(targetValuesMapRcp, pointValuesMapRcp, numEntPerRowToAlloc, Tpetra::StaticProfile));

    const auto &wPool = *weightPoolPtr; // the weight pool with near rod info
    const int wQueNum = wPool.size();

    // multi-thread filling. nThreads = poolSize, each thread processes a queue
#pragma omp parallel for num_threads(wQueNum)
    for (int threadId = 0; threadId < wQueNum; threadId++) {
        // each thread process a queue
        const auto &wBlockQue = wPool[threadId];
        const int wBlockNum = wBlockQue.size();

        // the 3x3 u and laplacian of u mobility blocks
        Emat3 MobU;
        Emat3 MobLapU;

        constexpr double Pi = M_PI;
        const double FACV = 1.0 / (8 * Pi);

        for (int wBlockIdx = 0; wBlockIdx < wBlockNum; wBlockIdx++) {
            // extract the necessary information from the weight block
            const int lidQuadPtI = wBlockQue[wBlockIdx].lidQuadPtI;
            const int lidQuadPtJ = wBlockQue[wBlockIdx].lidQuadPtJ;
            const double radiusQuadPtI = wBlockQue[wBlockIdx].radiusQuadPtI;
            const double lineHalfLengthI = wBlockQue[wBlockIdx].lineHalfLengthI;
            const double sqw1 = wBlockQue[wBlockIdx].sqw1 * lineHalfLengthI;
            const double sqw3 = wBlockQue[wBlockIdx].sqw1 * lineHalfLengthI;
            const double sqw5 = wBlockQue[wBlockIdx].sqw1 * lineHalfLengthI;
            const double wGL = wBlockQue[wBlockIdx].wGL * lineHalfLengthI;
            const Evec3 R = ECmap3(wBlockQue[wBlockIdx].rIJ);

            // compute the source-target mobility matrix
            const Emat3 RR = R * R.transpose();
            const double w1Corr = sqw1 - wGL;
            const double w3Corr = sqw3 - wGL;
            const double w5Corr = sqw5 - wGL;

            const double a2_over_three = (1.0 / 3.0) * radiusQuadPtI * radiusQuadPtI;
            const double rinv = 1.0 / R.norm();
            const double rinv3 = rinv * rinv * rinv;
            const double rinv5 = rinv * rinv * rinv3;
            const double w1Corr_rinv = w1Corr * rinv;
            const double w3Corr_rinv3 = w3Corr * rinv3;
            const double w5Corr_rinv5 = w5Corr * rinv5;

            const double c1 = w1Corr_rinv + a2_over_three * w3Corr_rinv3;
            const double c2 = w3Corr_rinv3 - radiusQuadPtI * radiusQuadPtI * w5Corr_rinv5;
            MobU = c1 * Emat3::Identity() + c2 * RR;
            MobLapU = (2 * w3Corr_rinv3) * Emat3::Identity() - (6 * w5Corr_rinv5) * RR;

            // fill the sparce matrix one row block at a time
            int localRow; // this row *must* be locally owned
            int cols[3];
            double vals[3];
            const int numEnt = 3;
            for (int r = 0; r < 3; r++) { // first 3 rows are MobU
                cols[0] = 3 * lidQuadPtI;
                cols[1] = 3 * lidQuadPtI + 1;
                cols[2] = 3 * lidQuadPtI + 2;
                vals[0] = FACV * MobU(r, 0);
                vals[1] = FACV * MobU(r, 1);
                vals[2] = FACV * MobU(r, 2);
                localRow = 6 * lidQuadPtJ + r;

                // insertLocalValues is not thread safe
                sqwMobilityMatrixRcp->replaceLocalValues(localRow, numEnt, vals, cols);
            }

            for (int r = 0; r < 3; r++) { // final 3 rows are MobLapU
                cols[0] = 3 * lidQuadPtI;
                cols[1] = 3 * lidQuadPtI + 1;
                cols[2] = 3 * lidQuadPtI + 2;
                vals[0] = FACV * MobLapU(r, 0);
                vals[1] = FACV * MobLapU(r, 1);
                vals[2] = FACV * MobLapU(r, 2);
                localRow = 6 * lidQuadPtJ + r + 3;
                sqwMobilityMatrixRcp->replaceLocalValues(localRow, numEnt, vals, cols);
            }
        }
    }
    // sqwMobilityMatrix has domain: 3*nTrgGlobal and range: 3*nTrgGlobal
    sqwMobilityMatrixRcp->fillComplete(pointValuesMapRcp, targetValuesMapRcp); // domainMap, rangeMap
}