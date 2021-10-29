/**
 * @file Sylinder.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Sphero-cylinder type
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef SYLINDER_HPP_
#define SYLINDER_HPP_

#include "Particle/Particle.hpp"

#include "Util/EigenDef.hpp"
#include "Util/EquatnHelper.hpp"
#include "Util/IOHelper.hpp"

#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

/**
 * @brief Sphero-cylinder class
 *
 */
class Sylinder : public Particle {
public:
  double radius;          ///< radius
  double radiusCollision; ///< radius for collision resolution
  double length;          ///< length
  double lengthCollision; ///< length for collision resolution
  double radiusSearch;    ///< radiusSearch for short range interactions
  double sepmin; ///< minimal separation with its neighbors within radiusSearch
  double colBuf; ///< collision buffer

  /**
   * @brief Construct a new Sylinder object
   *
   */
  Sylinder() = default;

  /**
   * @brief Destroy the Sylinder object
   *
   */
  ~Sylinder() = default;

  /**
   * @brief Construct a new Sylinder object
   *
   * @param gid_
   * @param radius_
   * @param radiusCollision_
   * @param length_
   * @param lengthCollision_
   * @param pos_ if not specified position is set as [0,0,0]
   * @param orientation_ if not specied orientation is set as identity
   */
  Sylinder(const int &gid_, const double &radius_,
           const double &radiusCollision_, const double &length_,
           const double &lengthCollision_, const double pos_[3] = nullptr,
           const double orientation_[4] = nullptr);

  /**
   * @brief Copy constructor
   *
   */
  Sylinder(const Sylinder &) = default;
  Sylinder(Sylinder &&) = default;
  Sylinder &operator=(const Sylinder &) = default;
  Sylinder &operator=(Sylinder &&) = default;

  /**
   * @brief display the data fields for this sylinder
   *
   */
  void dumpSylinder() const;

  /**
   * @brief set the velocity data fields to zero
   *
   */
  void clear();

  /**
   * @brief return if this sylinder is treated as a sphere
   *
   * @param collision
   * @return true
   * @return false
   */
  bool isSphere(bool collision = false) const;

  /**
   * @brief calculate the three drag coefficient
   *
   * @param viscosity
   * @param dragPara
   * @param dragPerp
   * @param dragRot
   */
  void calcDragCoeff(const double viscosity, double &dragPara, double &dragPerp,
                     double &dragRot) const;

  /**
   * @brief update the position and orientation with internal velocity data
   * fields and given dt
   *
   * @param dt
   */
  void stepEuler(double dt);

  /**
   * @brief return position
   *
   * necessary interface for InteractionManager.hpp
   * @return const double*
   */
  const double *Coord() const { return pos; }

  /**
   * @brief return search radius
   *
   * necessary interface for InteractionManager.hpp
   * @return double
   */
  double Rad() const { return radiusCollision * 4 + lengthCollision; }

  /**
   * @brief Get the Gid
   *
   * necessary interface for FDPS FullParticle class
   * @return int
   */
  int getGid() const { return gid; }
};

#endif
