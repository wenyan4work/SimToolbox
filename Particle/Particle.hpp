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

#include <msgpack.hpp>

#include <array>
#include <utility>

constexpr double Pi = 3.14159265358979323846;

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
template <class ShapeData>
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

  ShapeData data;

  /**
   * @brief define msgpack serialization format
   *
   */
  MSGPACK_DEFINE(gid, globalIndex, group, rank, isImmovable, //
                 pos, orientation,                           //
                 vel, velConU, velConB, velNonCon, velBrown, //
                 force, forceConU, forceConB, forceNonCon, data);

  Particle() = default;
  ~Particle() = default;

  Particle(const Particle &) = default;
  Particle(Particle &&) = default;
  Particle &operator=(const Particle &) = default;
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
  Emat6 getMobMat() const {
    return isImmovable ? std::move(Emat6::Zero())
                       : std::move(data.getMobMat(orientation));
  };

  /**
   * @brief Get AABB for neighbor search
   *
   * @return std::pair<std::array<double,3>, std::array<double,3>>
   * boxLow,boxHigh
   */
  std::pair<std::array<double, 3>, std::array<double, 3>> getBox() const {
    return std::move(data.getBox(pos, orientation));
  };

  /**
   * @brief display
   *
   */
  void echo() const {
    printf("-----------------------------------------------\n");
    printf("gid %d, globalIndex %d, pos %g, %g, %g\n", //
           gid, globalIndex,                           //
           pos[0], pos[1], pos[2]);
    printf("orient %g, %g, %g, %g\n", //
           orientation[0], orientation[1], orientation[2], orientation[3]);
    printf("vel %g, %g, %g; omega %g, %g, %g\n", //
           vel[0], vel[1], vel[2],               //
           vel[3], vel[4], vel[5]);
    data.echo();
  }
};

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type //
for_each(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
{}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if < I<sizeof...(Tp), void>::type //
                                     for_each(std::tuple<Tp...> &t, FuncT f) {
  f(std::get<I>(t));
  for_each<I + 1, FuncT, Tp...>(t, f);
}

/**
 * @brief Container for multiple types of particles
 *
 * @tparam ParType
 */
template <class... ParType>
struct MultiTypeContainer {
  std::tuple<std::vector<ParType>...> particles;

  std::vector<int> buildOffset() {
    std::vector<int> offset(1, 0);
    // iterate over particle types

    auto getsize = [&](const auto &container) {
      offset.push_back(container.size() + offset.back());
    };

    for_each(particles, getsize);
    return offset;
  }
};

#endif