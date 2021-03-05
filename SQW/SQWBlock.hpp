/**
 * @file SQWBlock.hpp
 * @author Bryce Palmer (palme200@msu.edu)
 * @brief
 * @version 0.1
 * @date 2021-03-01
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef SQWBLOCK_HPP_
#define SQWBLOCK_HPP_

#include "Util/EigenDef.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"
#include "Util/QuadInt.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

/**
 * @brief special quadrature near rod information block
 *
 * Each block stores the information for one rod-rod special quadrature correction.
 * The blocks are collected by SQWCollector and then used to construct the weightPool of point-point corrections
 */
template <int N>
struct SQWBlock {
  public:
    int gidI = GEO_INVALID_INDEX;         ///< unique global ID of particle I
    int gidJ = GEO_INVALID_INDEX;         ///< unique global ID of particle J
    int globalIndexI = GEO_INVALID_INDEX; ///< global index of particle I
    int globalIndexJ = GEO_INVALID_INDEX; ///< global index of particle J
    int speciesIDI = 0;
    int speciesIDJ = 0;
    double lengthI = 0;
    double lengthJ = 0;
    double posI[3] = {0, 0, 0};
    double posJ[3] = {0, 0, 0};
    double directionI[3] = {0, 0, 0};
    double directionJ[3] = {0, 0, 0};
    const QuadInt<N> *quadPtrI; 
    const QuadInt<N> *quadPtrJ; 

    /**
     * @brief Construct a new empty sqw block
     *
     */
    SQWBlock() = default;

    /**
     * @brief Construct a new SQWBlock object
     *
     * @param gidI_
     * @param gidJ_
     * @param globalIndexI_
     * @param globalIndexJ_
     * @param speciesIDI_
     * @param speciesIDJ_
     * @param lengthI_
     * @param lengthJ_
     * @param posI_
     * @param posJ_
     * @param directionI_
     * @param directionJ_
     * @param quadPtrI_
     * @param quadPtrJ_
     */
    SQWBlock(int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_, int speciesIDI_, int speciesIDJ_, 
             const double lengthI_, const double lengthJ_, const double posI_[3], const double posJ_[3], 
             const double directionI_[3], const double directionJ_[3], 
             const QuadInt<N> *quadPtrI_, const QuadInt<N> *quadPtrJ_)
        : gidI(gidI_), gidJ(gidJ_), globalIndexI(globalIndexI_), globalIndexJ(globalIndexJ_), 
          lengthI(lengthI_), lengthJ(lengthJ_), quadPtrI(quadPtrI_), quadPtrJ(quadPtrJ_),
          speciesIDI(speciesIDI_), speciesIDJ(speciesIDJ_) {
        for (int d = 0; d < 3; d++) {
            posI[d] = posI_[d];
            posJ[d] = posJ_[d];
            directionI[d] = directionI_[d];
            directionJ[d] = directionJ_[d];
        }
    }
};

static_assert(std::is_trivially_copyable<SQWBlock<10>>::value, "");
static_assert(std::is_default_constructible<SQWBlock<10>>::value, "");

template <int N>
using SQWBlockQue = std::vector<SQWBlock<N>>;     ///< a queue contains blocks collected by one thread
template <int N>
using SQWBlockPool = std::vector<SQWBlockQue<N>>; ///< a pool contains queues on different threads

#endif