#include "Particle.hpp"

struct SphereShape {
  double radius = 1;
  MSGPACK_DEFINE(radius);

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
};

using Sphere = Particle<SphereShape>;

static_assert(std::is_trivially_copyable<Sphere>::value, "");
static_assert(std::is_default_constructible<Sphere>::value, "");