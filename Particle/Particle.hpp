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

#include "Constraint/ConstraintBlock.hpp"
#include "Util/EigenDef.hpp"
#include "Util/EquatnHelper.hpp"

#include <msgpack.hpp>

#include <array>
#include <sstream>
#include <utility>

constexpr double Pi = 3.14159265358979323846;

/**
 * @brief an empty user data class
 *
 */
struct EmptyData {
  int data;
  void echo() const { return; }
  void parse(const std::string &line) { return; }
  MSGPACK_DEFINE_ARRAY(data);
};

/**
 * @brief base class for all particle types
 *
 * vel    = velConU   + velConB   + velNonCon     + vel Brown.
 * force  = forceConU + forceConB + forceNonCon.
 * there is no Brownian force.
 */
template <class Shape, class Data = EmptyData>
struct Particle {
  int rank = -1; ///< mpi rank

  long gid = -1;         ///< unique global id
  long globalIndex = -1; ///< unique and sequentially ordered from rank 0
  long group = -1;       ///< a 'marker'

  bool immovable = false; ///< if true, mobmat 6x6 = 0

  double buffer = 0; ///< shape buffer on AABB

  std::array<double, 3> pos = {0, 0, 0}; ///< position

  /**
   * @brief orientation quaternion
   *
   * direction norm vector = quaternion * (0,0,1)
   * quaternion = {x,y,z,w}, {x,y,z} vector and w scalar
   * This order of storage is used by Eigen::Quaternion
   * See Util/EquatnHelper.hpp
   */
  std::array<double, 4> quaternion = {0, 0, 0, 1};

  // vel [6] = vel [3] + omega [3]
  std::array<double, 6> vel = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> velConU = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> velConB = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> velNonCon = {0, 0, 0, 0, 0, 0};

  std::array<double, 6> velBrown = {0, 0, 0, 0, 0, 0};

  // force [6] = force[3] + torque [3]
  std::array<double, 6> force = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> forceConU = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> forceConB = {0, 0, 0, 0, 0, 0};
  std::array<double, 6> forceNonCon = {0, 0, 0, 0, 0, 0};

  Shape shape; ///< shape variables and methods
  Data data;   ///< other user-defined variables and methods

  /**
   * @brief define msgpack serialization format
   *
   */
  MSGPACK_DEFINE_ARRAY(gid, globalIndex, group, rank, immovable, //
                       buffer, pos, quaternion,                  //
                       vel, velConU, velConB, velNonCon,         //
                       velBrown,                                 //
                       force, forceConU, forceConB, forceNonCon, //
                       shape, data);

  Particle() = default;
  ~Particle() = default;

  Particle(const Particle &) = default;
  Particle(Particle &&) = default;
  Particle &operator=(const Particle &) = default;
  Particle &operator=(Particle &&) = default;

  /**
   * @brief Construct a new Particle object
   *
   * @param line a line of string
   */
  Particle(std::stringstream &line) {
    // shape parse line, set immovable, pos and quaternion
    // dataline contains the rest of line
    line = shape.parse(line, immovable, pos, quaternion);
    line = data.parse(line);
    // std::cout << "unused data: " << line << std::endl;
  }

  /**
   * @brief clear velocity
   *
   */
  void clear() {
    vel.fill(0);
    velConB.fill(0);
    velConU.fill(0);
    velNonCon.fill(0);
    velBrown.fill(0);
    force.fill(0);
    forceConB.fill(0);
    forceConU.fill(0);
    forceNonCon.fill(0);
  }

  /**
   * @brief Get position quaternion
   *
   * @return const double*
   */
  auto getPos() const { return pos; };

  /**
   * @brief Get quaternion
   *
   * @return const double*
   */
  auto getQuaternion() const { return quaternion; }

  /**
   * @brief calculate mobility matrix
   *
   * @return EMat6 mobility matrix 6x6
   */
  Emat6 getMobMat(const double mu) const {
    return immovable ? Emat6::Zero() : shape.getMobMat(quaternion, mu);
  };

  Evec6 getVelBrown(const std::array<double, 12> &rngN01s, const double dt,
                    const double kbt, const double mu) const {
    return immovable ? Emat6::Zero()
                     : shape.getVelBrown(quaternion, rngN01s, dt, mu);
  }

  /**
   * @brief Get AABB for neighbor search
   *
   * @return std::pair<std::array<double,3>, std::array<double,3>>
   * boxLow,boxHigh
   */
  std::pair<std::array<double, 3>, std::array<double, 3>> getBox() const {
    auto box = shape.getBox(pos, quaternion);
    for (auto &v : box.first) {
      v -= buffer;
    }
    for (auto &v : box.second) {
      v += buffer;
    }
    return box;
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
           quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    printf("vel %g, %g, %g; omega %g, %g, %g\n", //
           vel[0], vel[1], vel[2],               //
           vel[3], vel[4], vel[5]);
    shape.echo();
    data.echo();
  }

  double getVolume() const { return shape.getVolume(); }
};

#endif