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

#include <deque>
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
  double delta0 = 0;
  double gamma = 0;
  double gammaLB = 0;

  long gidI = GEO_INVALID_INDEX;         ///< unique global ID of particle I
  long gidJ = GEO_INVALID_INDEX;         ///< unique global ID of particle J
  long globalIndexI = GEO_INVALID_INDEX; ///< global index of particle I
  long globalIndexJ = GEO_INVALID_INDEX; ///< global index of particle J

  bool oneSide = false;
  bool bilateral = false;

  double kappa = 0; ///< spring constant. =0 means no spring

  double normI[3] = {0, 0, 0};
  double normJ[3] = {0, 0, 0};
  double posI[3] = {0, 0, 0};
  double posJ[3] = {0, 0, 0};
  double labI[3] = {0, 0, 0};
  double labJ[3] = {0, 0, 0};

  double stress[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  ///< stress 3x3 matrix (row-major) for unit constraint force gamma

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
   * @brief Construct a new ConstraintBlock object
   *
   * @param delta0_ current value of the constraint function
   * @param gamma_ force magnitude, could be an initial guess
   * @param gidI_
   * @param gidJ_
   * @param globalIndexI_
   * @param globalIndexJ_
   * @param normI_ surface norm vector at the location of constraints.
   * @param normJ_
   * @param posI_ the constraint position on bodies I and J relative to CoM
   * @param posJ_
   * @param labI_ the labframe location of collision points
   * @param labJ_
   * @param oneSide_ flag for one side constarint, body J does not appear in
   * mobility matrix
   * @param bilateral_ flag for bilateral constraint
   * @param kappa_ flag for kappa of bilateral constraint
   * @param gammaLB_ lower bound of gamma for unilateral constraints
   */
  ConstraintBlock(double delta0_, double gamma_,          //
                  long gidI_, long gidJ_,                 //
                  long globalIndexI_, long globalIndexJ_, //
                  const double normI_[3], const double normJ_[3],
                  const double posI_[3], const double posJ_[3],
                  const double labI_[3], const double labJ_[3], //
                  bool oneSide_, bool bilateral_, double kappa_,
                  double gammaLB_)
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

///< a queue contains blocks collected by one thread
using ConstraintBlockQue = std::deque<ConstraintBlock>;
///< a pool contains queues on different threads
using ConstraintBlockPool = std::deque<ConstraintBlockQue>;

#endif