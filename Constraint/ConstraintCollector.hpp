/**
 * @file ConstraintCollector.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Collect all constraints
 * @version 0.1
 * @date 2018-12-14
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef CONSTRAINTCOLLECTOR_HPP_
#define CONSTRAINTCOLLECTOR_HPP_

#include "ConstraintBlock.hpp"

#include "Trilinos/TpetraUtil.hpp"
#include "Util/IOHelper.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <omp.h>

/**
 * @brief Collecter of constraint blocks
 *
 */
class ConstraintCollector {
  public:
    std::shared_ptr<ConstraintBlockPool> constraintPoolPtr;

    ConstraintCollector();

    /**
     * @brief Destroy the ConstraintCollector object
     *
     */
    ~ConstraintCollector() = default;

    /**
     * @brief Forbid copying the ConstraintCollector object
     *
     */
    ConstraintCollector(const ConstraintCollector &) = delete;
    ConstraintCollector &operator=(const ConstraintCollector &) = delete;

    /**
     * @brief if the shared pointer to constraint pool is not allocated
     *
     * After the constructor this should always be false
     * @return true
     * @return false
     */
    bool valid() const;

    /**
     * @brief clear the blocks and get an empty constraint pool
     *
     * The constraint pool still contains (the number of openmp threads) queues
     *
     */
    void clear();

    /**
     * @brief get the number of constraints on the local node
     *
     * @return int
     */
    int getLocalNumberOfConstraints();

    /**
     * @brief compute the total constraint stress of all constraints (blocks)
     *
     * @param stress the sum of all stress blocks for all threads on the local rank
     * @param withOneSide include the stress (without proper definition) of one side constraints
     */
    void sumLocalConstraintStress(Emat3 &uniStress, Emat3 &biStress, bool withOneSide = false) const;

    /**
     * @brief write VTK XML PVTP Header file from rank 0
     *
     * the files will be written as folder/prefixConBlock_rX_postfix.vtp
     * @param folder
     * @param prefix
     * @param postfix
     * @param nProcs
     */
    void writePVTP(const std::string &folder, const std::string &prefix, const std::string &postfix,
                   const int nProcs) const;

    /**
     * @brief write VTK XML binary base64 VTP data file from every MPI rank
     *
     * files are written as
     * folder + '/' + prefix + ("ColBlock_") + "r" + (rank) + ("_") + postfix + (".vtp")
     *
     * Procedure for dumping constraint blocks in the system:
     * Each block writes a polyline with two (connected) points.
     * Points are labeled with float -1 and 1
     * Constraint data fields are written as point and cell data
     * Rank 0 writes the parallel header , then each rank write its own serial vtp file
     *
     * @param folder
     * @param prefix
     * @param postfix
     * @param rank
     */
    void writeVTP(const std::string &folder, const std::string &prefix, const std::string &postfix, int rank) const;

    /**
     * @brief dump the blocks to screen for debugging
     *
     */
    void dumpBlocks() const;

    /**
     * @brief build the matrix and vectors used in constraint solver
     *
     * @param [in] mobMapRcp  mobility map
     * @param DTransRcp D^Trans matrix
     * @param invKappaRcp K^{-1} vector
     * @param biFlagRcp 1 for bilateral, 1 for unilateral
     * @param gammaGuessRcp initial guess of gamma
     * @return DMatTransRcp D^Trans matrix
     */
    Teuchos::RCP<TCMAT> buildConstraintMatrixVector(const Teuchos::RCP<const TMAP> &mobMapRcp,
                                                    const Teuchos::RCP<const TMAP> &gammaMapRcp,
                                                    const Teuchos::RCP<const TV> &gammaRcp) const;

    int fillConstraintInformation(const Teuchos::RCP<const TCOMM>& commRcp,
                                  const Teuchos::RCP<TV> &gammaGuessRcp,         
                                  const Teuchos::RCP<TV> &constraintFlagRcp) const;
                                  
    int evalConstraintDiagonal(const Teuchos::RCP<const TV> &gammaRcp, 
                               const Teuchos::RCP<TV> &constraintDiagonalRcp) const;

    int evalConstraintValues(const Teuchos::RCP<const TV> &gammaRcp, 
                             const Teuchos::RCP<TV> &constraintValueRcp) const;


    // /**
    //  * @brief build the K^{-1} diagonal matrix
    //  *
    //  * @param invKappa
    //  * @return int  error code (future)
    //  */
    // int buildInvKappa(std::vector<double> &invKappa) const;

    /**
     * @brief build the index of constraints in the ConstraintPool
     *
     * @param cQueSize size of each queue
     * @param cQueIndex index of queue
     * @return int error code (future)
     */
    int buildConIndex(std::vector<int> &cQueSize, std::vector<int> &cQueIndex) const;

    /**
     * @brief write back the solution gamma to the blocks
     *
     * @param gammaRcp solution
     * @return int error code (future)
     */
    int writeBackGamma(const Teuchos::RCP<const TV> &gammaRcp);
};

#endif