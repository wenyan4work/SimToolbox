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
#include "Util/EigenDef.hpp"
#include "Util/IOHelper.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <omp.h>

/**
 * @brief collecter of collision blocks
 *
 * This class must have very low copy overhead
 *
 */
class ConstraintCollector {
public:
  // all copy of collector share a pointer to collision pool
  std::shared_ptr<ConstraintBlockPool> constraintPoolPtr;

  ConstraintCollector();
  ~ConstraintCollector() = default;

  ConstraintCollector(const ConstraintCollector &obj) = default;
  ConstraintCollector(ConstraintCollector &&obj) = default;
  ConstraintCollector &operator=(const ConstraintCollector &obj) = default;
  ConstraintCollector &operator=(ConstraintCollector &&obj) = default;

  /**
   * @brief clear the blocks and get an empty collision pool
   *
   * The collision pool still contains (the number of openmp threads) queues
   *
   */
  void clear();

  /**
   * @brief get the number of collision constraints on the local node
   *
   * @return int
   */
  int getLocalNumberOfConstraints();

  /**
   * @brief compute the total collision stress of all constraints (blocks)
   *
   * @param stress the sum of all stress blocks for all threads on the local
   * rank
   * @param withOneSide include the stress (without proper definition) of one
   * side collisions
   */
  void sumLocalConstraintStress(Emat3 &uniStress, Emat3 &biStress,
                                bool withOneSide = false) const;

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
   * @param delta0Rcp delta_0 vector
   * @param invKappaRcp K^{-1} vector
   * @param biFlagRcp 1 for bilateral, 1 for unilateral
   * @param gammaGuessRcp initial guess of gamma
   * @return int error code (TODO:)
   */
  int buildConstraintMatrixVector(const Teuchos::RCP<const TMAP> &mobMapRcp, //
                                  Teuchos::RCP<TCMAT> &DMatTransRcp,         //
                                  Teuchos::RCP<TV> &delta0Rcp,               //
                                  Teuchos::RCP<TV> &invKappaRcp,             //
                                  Teuchos::RCP<TV> &biFlagRcp,               //
                                  Teuchos::RCP<TV> &gammaGuessRcp) const;

  /**
   * @brief build the offset for each ConstarintBlockQue
   *
   * @return std::vector<int>
   */
  std::vector<int> buildConQueOffset() const;

  /**
   * @brief write back the solution gamma to the blocks
   *
   * @param gammaRcp solution
   * @return int error code (TODO:)
   */
  int writeBackGamma(const Teuchos::RCP<const TV> &gammaRcp);

  /**
   * @brief write data in msgpack format
   *
   * @param filename
   * @param overwrite
   */
  void writeConstraintBlockPool(const std::string &filename,
                                bool overwrite = false) const;
};

#endif