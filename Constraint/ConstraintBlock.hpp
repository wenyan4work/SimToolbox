/**
 * @file ConstraintBlock.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 1.0
 * @date 2021-10-21
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef CONSTRAINTBLOCK_HPP_
#define CONSTRAINTBLOCK_HPP_

#include "Util/IOHelper.hpp"

#include <msgpack.hpp>

#include <deque>
#include <string>
#include <vector>

constexpr long GEO_INVALID_INDEX = -1;
constexpr double GEO_DEFAULT_COLBUF = 0.5;
constexpr double GEO_DEFAULT_RADIUS = 1.2;
constexpr double GEO_DEFAULT_NEIGHBOR = 2.0;

/**
 * @brief collision constraint information block
 *
 * Each block stores the information for one collision constraint.
 * The blocks are collected by ConstraintCollector and then used to construct
 * the sparse fcTrans matrix
 */
struct ConstraintBlock {
public:
  double delta0 = 0;      ///< current value of the constraint function
  double gamma = 0;       ///< force magnitude, could be an initial guess
  double gammaLB = 0;     ///< lower bound of gamma in solver
  double kappa = 0;       ///< spring constant. <=0 means no spring
  bool oneSide = false;   ///< true means J appears in mobility matrix
  bool bilateral = false; ///< true means a bilateral constraint

  long gidI = GEO_INVALID_INDEX;         ///< unique global ID of I
  long gidJ = GEO_INVALID_INDEX;         ///< unique global ID of J
  long globalIndexI = GEO_INVALID_INDEX; ///< global index of I in mobility
  long globalIndexJ = GEO_INVALID_INDEX; ///< global index of J in mobility

  double normI[3] = {0, 0, 0}; ///< F on I = gamma * normI
  double normJ[3] = {0, 0, 0}; ///< F on J = gamma * normJ
  double posI[3] = {0, 0, 0};  ///< F position relative to center of mass of I
  double posJ[3] = {0, 0, 0};  ///< F position relative to center of mass of J
  double labI[3] = {0, 0, 0};  ///< F position in lab frame on I
  double labJ[3] = {0, 0, 0};  ///< F position in lab frame on J

  double stress[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}; ///< row-major stress tensor

  /**
   * @brief Construct a new empty collision block
   *
   */
  ConstraintBlock() = default;
  ~ConstraintBlock() = default;

  ConstraintBlock(const ConstraintBlock &other) = default;
  ConstraintBlock(ConstraintBlock &&other) = default;
  ConstraintBlock &operator=(const ConstraintBlock &other) = default;
  ConstraintBlock &operator=(ConstraintBlock &&other) = default;

  /**
   * @brief Construct a new Constraint Block object
   *
   * @param delta0_
   * @param gamma_
   * @param gidI_
   * @param gidJ_
   * @param globalIndexI_
   * @param globalIndexJ_
   * @param normI_
   * @param normJ_
   * @param posI_
   * @param posJ_
   * @param labI_
   * @param labJ_
   * @param oneSide_
   * @param bilateral_
   * @param kappa_
   * @param gammaLB_
   */
  ConstraintBlock(double delta0_, double gamma_,          //
                  long gidI_, long gidJ_,                 //
                  long globalIndexI_, long globalIndexJ_, //
                  const double normI_[3], const double normJ_[3],
                  const double posI_[3], const double posJ_[3],
                  const double labI_[3], const double labJ_[3], //
                  bool oneSide_, bool bilateral_, double kappa_ = 0,
                  double gammaLB_ = 0)
      : delta0(delta0_), gamma(gamma_), gidI(gidI_), gidJ(gidJ_),
        globalIndexI(globalIndexI_), globalIndexJ(globalIndexJ_),
        oneSide(oneSide_), bilateral(bilateral_), kappa(kappa_),
        gammaLB(gammaLB_) {

    for (int d = 0; d < 3; d++) {
      normI[d] = normI_[d];
      normJ[d] = normJ_[d];
      posI[d] = posI_[d];
      posJ[d] = posJ_[d];
      labI[d] = labI_[d];
      labJ[d] = labJ_[d];
    }

    std::fill(stress, stress + 9, 0.0);
  }

  void setStress(const double *stress_) {
    std::copy(stress_, stress_ + 9, stress);
  }

  const double *getStress() const { return stress; }

  void reverseIJ() {
    std::swap(gidI, gidJ);
    std::swap(globalIndexI, globalIndexJ);
    for (int k = 0; k < 3; k++) {
      std::swap(normI[k], normJ[k]);
      std::swap(posI[k], posJ[k]);
      std::swap(labI[k], labJ[k]);
    }
  }
};

static_assert(std::is_trivially_copyable<ConstraintBlock>::value, "");
static_assert(std::is_default_constructible<ConstraintBlock>::value, "");

/**
 * @brief a queue contains blocks collected by one thread
 *
 */
using ConstraintBlockQue = std::deque<ConstraintBlock>;

/**
 * @brief  a pool contains queues on different threads
 *
 */
using ConstraintBlockPool = std::deque<ConstraintBlockQue>;

namespace msgpack {
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
  namespace adaptor {

  /**
   * @brief full specialization of msgpack::pack
   *
   * @tparam
   */
  template <>
  struct pack<ConstraintBlockPool> {
    /**
     * @brief
     *
     * @tparam Stream
     * @param o destination of data to be saved
     * @param cPool data to be saved
     * @return packer<Stream>&
     */
    template <typename Stream>
    packer<Stream> &operator()(msgpack::packer<Stream> &o,
                               const ConstraintBlockPool &cPool) const {
      int blkN = 0;
      for (auto &que : cPool) {
        blkN += que.size();
      }

      o.pack_map(17); // pack to a map with 17 arrays

      /**
       * @brief pack every data member as a contiguous array in the map
       *
       */
      auto pack_variable = [&](const std::string &m_name,
                               const auto ConstraintBlock::*m_ptr) {
        o.pack(m_name);
        o.pack_array(blkN);
        for (const auto &que : cPool) {
          for (const auto &blk : que) {
            o.pack(blk.*m_ptr);
          }
        }
      };

      pack_variable("delta0", &ConstraintBlock::delta0);
      pack_variable("gamma", &ConstraintBlock::gamma);
      pack_variable("gammaLB", &ConstraintBlock::gammaLB);
      pack_variable("oneSide", &ConstraintBlock::oneSide);
      pack_variable("bilateral", &ConstraintBlock::bilateral);
      pack_variable("kappa", &ConstraintBlock::kappa);
      pack_variable("gidI", &ConstraintBlock::gidI);
      pack_variable("gidJ", &ConstraintBlock::gidJ);
      pack_variable("globalIndexI", &ConstraintBlock::globalIndexI);
      pack_variable("globalIndexJ", &ConstraintBlock::globalIndexJ);
      pack_variable("normI", &ConstraintBlock::normI);
      pack_variable("normJ", &ConstraintBlock::normJ);
      pack_variable("posI", &ConstraintBlock::posI);
      pack_variable("posJ", &ConstraintBlock::posJ);
      pack_variable("labI", &ConstraintBlock::labI);
      pack_variable("labJ", &ConstraintBlock::labJ);
      pack_variable("stress", &ConstraintBlock::stress);

      return o;
    }
  };

  } // namespace adaptor
} // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
} // namespace msgpack

#endif