/**
 * @file Particle.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2021-10-28
 *
 * @copyright Copyright (c) 2021
 *
 */
#ifndef PARTICLE_HPP_
#define PARTICLE_HPP_

#include "Util/EigenDef.hpp"

#include <type_traits>
#include <utility>

/**
 * @brief specify the link of sylinders
 *
 */
struct Link {
  long prev = -1; ///< previous link in the link group
  long next = -1; ///< next link in the link group
};

/**
 * @brief base class for all particle types
 *
 * vel    = velConU   + velConB   + velNonCon     + vel Brown.
 * force  = forceConU + forceConB + forceNonCon.
 * there is no Brownian force.
 */
struct Particle {
  long gid = -1;         ///< unique global id
  long globalIndex = -1; ///< unique and sequentially ordered from rank 0
  long group = -1;       ///< a 'marker'

  int rank = -1;            ///< mpi rank
  bool isImmovable = false; ///< if true, mobmat 6x6 = 0

  double pos[3] = {0, 0, 0}; ///< position

  /**
   * @brief orientation quaternion
   *
   * direction norm vector = orientation * (0,0,1)
   * orientation = {x,y,z,w}, {x,y,z} vector and w scalar
   * This order of storage is used by Eigen::Quaternion
   * See Util/EquatnHelper.hpp
   */
  double orientation[4] = {0, 0, 0, 1};

  double vel[6] = {0, 0, 0, 0, 0, 0};       ///< vel [6] = vel [3] + omega [3]
  double velConU[6] = {0, 0, 0, 0, 0, 0};   ///< unilateral constraint
  double velConB[6] = {0, 0, 0, 0, 0, 0};   ///< bilateral constraint
  double velNonCon[6] = {0, 0, 0, 0, 0, 0}; ///<  sum of non-Brown and non-Con

  double velBrown[6] = {0, 0, 0, 0, 0, 0}; ///< Brownian velocity

  double force[6] = {0, 0, 0, 0, 0, 0}; ///< force [6] = force[3] + torque [3]
  double forceConU[6] = {0, 0, 0, 0, 0, 0};   ///< unilateral constraint
  double forceConB[6] = {0, 0, 0, 0, 0, 0};   ///< bilateral constraint
  double forceNonCon[6] = {0, 0, 0, 0, 0, 0}; ///<  sum of non-Brown and non-Con

  Particle() = default;
  virtual ~Particle() = default;

  Particle(const Particle &) = default;
  Particle(Particle &&) = default;
  Particle &operator=(Const Particle &) = default;
  Particle &operator=(Particle &&) = default;

  /**
   * @brief Get position quaternion
   *
   * @return const double*
   */
  inline const double *getPos() const { return pos; };

  /**
   * @brief Get orientation quaternion
   *
   * @return const double*
   */
  inline const double *getOrientation() const { return orientation; }

  /**
   * @brief calculate mobility matrix
   *
   * @return EMat6 mobility matrix 6x6
   */
  virtual EMat6 calcMobMat() const = 0;

  /**
   * @brief Get AABB for neighbor search
   *
   * @return std::pair<double *, double *> double boxLow[3], double boxHigh[3]
   */
  virtual std::pair<double *, double *> getBox() const = 0;
};

static_assert(std::is_trivially_copyable<Particle>::value, "");
static_assert(std::is_default_constructible<Particle>::value, "");

#endif