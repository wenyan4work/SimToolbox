/**
 * @file PointDistribution.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief helper functions for generate and distribute points
 * @version 0.1
 * @date 2019-04-10
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef POINTDISTRIBUTION_HPP_
#define POINTDISTRIBUTION_HPP_

#include <vector>
#include <string>

/**
 * @brief generate fixed 1, 2, or 4 points
 *
 * @param nPts
 * @param box
 * @param shift
 * @param srcCoord
 */
void fixedPoints(int nPts, double box, double shift, std::vector<double> &srcCoord);

/**
 * @brief generate random points
 *
 * @param nPts
 * @param box
 * @param shift
 * @param ptsCoord
 */
void randomPoints(int nPts, double box, double shift, std::vector<double> &ptsCoord);

/**
 * @brief shift and scale points in x,y,z directions.
 *
 * @param ptsCoord
 * @param shift
 * @param scale
 */
void shiftAndScalePoints(std::vector<double> &ptsCoord, double shift[3], double scale);

/**
 * @brief fill the vector with U[low, high)
 *
 * @param vec
 * @param low
 * @param high
 * @param seed
 */
void randomUniformFill(std::vector<double> &vec, double low, double high, int seed = 0);

/**
 * @brief fill the vector with logNormal(a,b)
 * 
 * @param vec 
 * @param a 
 * @param b 
 * @param seed 
 */
void randomLogNormalFill(std::vector<double> &vec, double a, double b, int seed = 0);

/**
 * @brief write points and values to a file
 * 
 * @param filename 
 * @param coord always 3D
 * @param value 
 * @param valueDimension dimension of value per point 
 */
void dumpPoints(const std::string &filename, std::vector<double> &coord, std::vector<double> &value,
                const int valueDimension);

/**
 * @brief check if error between value and valueTrue
 * 
 * @param value 
 * @param valueTrue 
 * @param bar if not zero, compares max error with bar and print error information
 */
void checkError(const std::vector<double> &value, const std::vector<double> &valueTrue, const double bar = 0);

/**
 * @brief distribute points to MPI ranks, maintain dimension per point
 * 
 * @param pts 
 * @param dimension 
 */
void distributePts(std::vector<double> &pts, int dimension);

/**
 * @brief concatenate pts from all ranks to rank0
 * 
 * @param pts 
 */
void collectPts(std::vector<double> &pts);

/**
 * @brief concatenate pts from all ranks to all ranks
 * 
 * @param pts 
 */
void collectPtsAll(std::vector<double> &pts);

#endif