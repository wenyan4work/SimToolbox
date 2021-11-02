#include "Constraint/ConstraintBlock.hpp"

#include "Sphere.hpp"
#include "Sylinder.hpp"

/**
 * @brief generic interface
 *
 * @tparam A
 * @tparam B
 * @param a
 * @param b
 * @return ConstraintBlock
 */
template <class A, class B>
ConstraintBlock collide(const A &a, const B &b) {

  ConstraintBlock cBlk;
  return cBlk;
}

/**
 * @brief Sylinder-Sylinder collision
 *
 * @tparam
 * @param a
 * @param b
 * @return ConstraintBlock
 */
template <>
ConstraintBlock collide(const Sylinder &a, const Sylinder &b) {

  ConstraintBlock cBlk;
  return cBlk;
}

/**
 * @brief Sylinder-Sphere collision
 *
 * @tparam
 * @param a
 * @param b
 * @return ConstraintBlock
 */
template <>
ConstraintBlock collide(const Sylinder &a, const Sphere &b) {

  ConstraintBlock cBlk;
  return cBlk;
}

/**
 * @brief Sphere-Sphere collision
 *
 * @tparam
 * @param a
 * @param b
 * @return ConstraintBlock
 */
template <>
ConstraintBlock collide(const Sphere &a, const Sphere &b) {

  ConstraintBlock cBlk;
  return cBlk;
}