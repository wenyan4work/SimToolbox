#ifndef SPHERE_HPP_
#define SPHERE_HPP_

#include "Particle.hpp"

struct SphereShape {
  double radius = 1;

  MSGPACK_DEFINE_ARRAY(radius);

  /**
   * @brief
   *
   */
  void echo() const { printf("radius %g\n", radius); }

  Emat6 getMobMat(const std::array<double, 4> &quaternion) const {
    return Emat6::Identity() / (6 * Pi * radius);
  };

  /**
   * @brief Get AABB for neighbor search
   *
   * @return std::pair<std::array<double,3>, std::array<double,3>>
   * boxLow,boxHigh
   */
  std::pair<std::array<double, 3>, std::array<double, 3>>
  getBox(const std::array<double, 3> &pos,
         const std::array<double, 4> &quaternion) const {
    using Point = std::array<double, 3>;
    return std::make_pair<Point, Point>(
        Point{pos[0] - radius, pos[1] - radius, pos[2] - radius}, //
        Point{pos[0] + radius, pos[1] + radius, pos[2] + radius});
  };

  double getVolume() const { return 4 * Pi * radius * radius * radius / 3.0; }

  std::stringstream &parse(std::stringstream &line, //
                           long &gid, bool &immovable,
                           std::array<double, 3> &pos,
                           std::array<double, 4> &quaternion) {
    // S immovable gid radius cx cy cz data

    // required data
    char type, shape;
    double cx, cy, cz;
    line >> shape >> type >> gid >> radius >> cx >> cy >> cz;
    assert(shape == 'S');
    immovable = type == 'T' ? true : false;

    Evec3 center(cx, cy, cz);
    Emap3(pos.data()) = center;
    Emap3(quaternion.data()).setIdentity();

    return line;
  }
};

using Sphere = Particle<SphereShape>;

static_assert(std::is_trivially_copyable<Sphere>::value, "");
static_assert(std::is_default_constructible<Sphere>::value, "");

#endif