/**
 * @file SQWCollector.hpp
 * @author Bryce Palmer (palme200@msu.edu)
 * @brief Collect all special quadrature corrections
 * @version 0.1
 * @date 2021-03-01
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SQWCOLLECTOR_HPP_
#define SQWCOLLECTOR_HPP_

#include "SQWBlock.hpp"

#include "Trilinos/TpetraUtil.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"
#include "Util/QuadInt.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <omp.h>

/**
 * @brief special quadrature correction information block
 *
 * This struct contains everything needed to compute the sqw mobility matrix
 *
 */
struct weightData {
    const int lidQuadPtI = GEO_INVALID_INDEX;
    const int lidQuadPtJ = GEO_INVALID_INDEX;
    const double radiusQuadPtI = 0;
    const double lineHalfLengthI = 0;
    // Weights are original [-1,1] data, not scaled by length
    const double sqw1 = 0;
    const double sqw3 = 0;
    const double sqw5 = 0;
    const double wGL = 0;
    double rIJ[3] = {0, 0, 0};

    /**
     * @brief Construct a new empty weight block
     *
     */
    weightData() = default;

    /**
     * @brief Construct a new weightData object
     *
     * @param lidQuadPtI_ local source point ID
     * @param lidQuadPtJ_ local target point ID
     * @param radiusQuadPtI_ source point radius
     * @param lineHalfLengthI_ source rod halflength
     * @param w1_ special quadrature weight for 1/r terms
     * @param w3_ special quadrature weight for 1/r^3 terms
     * @param w5_ special quadrature weight for 1/r^5 terms
     * @param wGL_ uncorrected Gauss Legendre quadrature weight
     * @param rIJ_ distance from source to target point
     */
    weightData(const int lidQuadPtI_, const int lidQuadPtJ_, const double radiusQuadPtI_, const double lineHalfLengthI_,
               const double w1_, const double w3_, const double w5_, const double wGL_, const double *rIJ_)
        : radiusQuadPtI(radiusQuadPtI_), lidQuadPtJ(lidQuadPtJ_), lidQuadPtI(lidQuadPtI_),
          lineHalfLengthI(lineHalfLengthI_), sqw1(w1_), sqw3(w3_), sqw5(w5_), wGL(wGL_) {
        for (int i = 0; i < 3; i++) {
            rIJ[i] = rIJ_[i];
        }
    }
};

/**
 * @brief collecter of special quadrature weight blocks
 *
 * This class must have very low copy overhead due to the design of FDPS
 *
 */
template <int N>
class SQWCollector {
  public:
    std::shared_ptr<SQWBlockPool<N>> sqwPoolPtr;         ///< pointer to SQWBlockPool
    using weightBlockQue = std::vector<weightData>;      ///< a queue containing sqw weights collected by one thread
    using WeightBlockPool = std::vector<weightBlockQue>; ///< a pool containing queues on different threads
    std::shared_ptr<WeightBlockPool> weightPoolPtr;      ///< pointer to WeightBlockPool
    std::vector<size_t> nSourcePerTarget;                ///< number of source points that contribute to each target

    ///< all copies of collector share a pointer to sqw pool
    ///< this is required by FDPS

    SQWCollector();

    /**
     * @brief Default copy
     *
     * @param obj
     */
    SQWCollector(const SQWCollector &obj) = default;
    SQWCollector(SQWCollector &&obj) = default;
    SQWCollector &operator=(const SQWCollector &obj) = default;
    SQWCollector &operator=(SQWCollector &&obj) = default;

    /**
     * @brief Destroy the SQWCollector object
     *
     */
    ~SQWCollector() = default;

    /**
     * @brief if the shared pointer to sqw pool is not allocated
     *
     * After the constructor this should always be false
     * @return true
     * @return false
     */
    bool valid() const;

    /**
     * @brief clear the blocks and get an empty sqw pool
     *
     * The sqw pool still contains (the number of openmp threads) queues
     *
     */
    void clear();

    /**
     * @brief get the number of sqwBlocks queued on the local node
     *
     * @return int
     */
    int getLocalOverallSQWQueSize();

    /**
     * @brief get the number of weightBlocks queued on the local node
     *
     * @return int
     */
    int getLocalOverallWeightQueSize();

    /**
     * @brief dump the sqwblocks to screen for debugging
     *
     */
    void dumpSQWBlocks() const;

    /**
     * @brief dump the weightData to screen for debugging
     *
     */
    void dumpWeightData() const;

    /**
     * @brief build the weight pool from the near rods
     *
     * @param rodMapRcp Teuchos rod map, 1 dof per rod
     * @param rodPtsIndex beginning of quadrature point local ID per rod
     * @param cellConfig hydro user input
     */
    void buildWeightPool(const Teuchos::RCP<TMAP> &rodMapRcp, const std::vector<int> &rodPtsIndex,
                         const Config &cellConfig);

    /**
     * @brief build the mobility matrix for applying the SQW correction
     *
     * @param pointValuesMapRcp domain map 3Q dof per rod
     * @param targetValuesMapRcp range map 6Q dof per rod
     * @param sqwMobilityMatrixRcp default initialized CrsMatrix to be constructed
     */
    void buildSQWMobilityMatrix(const Teuchos::RCP<TMAP> &pointValuesMapRcp,
                                const Teuchos::RCP<TMAP> &targetValuesMapRcp,
                                Teuchos::RCP<TCMAT> &sqwMobilityMatrixRcp);
};

// Include the SQWCollector implimentation
#include "SQWCollector.tpp"
#endif