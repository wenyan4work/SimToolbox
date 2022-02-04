#include "Particle.hpp"

/**
 * @brief concrete particle class of spherosylinder
 *
 */
struct SpherocylinderShape {
  double radius = 1.0; ///< radius
  double length = 1.0; ///< length

  MSGPACK_DEFINE(radius, length)

  /**
   * @brief
   *
   */
  void echo() const { printf("R %g, L %g\n", radius, length); }

  /**
   * @brief Get the Mob Mat object
   *
   * @param quaternion
   * @return Emat6
   */
  Emat6 getMobMat(const std::array<double, 4> &quaternion) const {
    double dragPara, dragPerp, dragRot;
    const double b = -(1 + 2 * log(radius / (length)));
    dragPara = 8 * Pi * length / (2 * b);
    dragPerp = 8 * Pi * length / (b + 2);
    dragRot = 2 * Pi * length * length * length / (3 * (b + 2));

    Emat6 mobmat = Emat6::Zero();
    Emat3 qq;
    Emat3 Imqq;
    Evec3 q = ECmapq(quaternion.data()) * Evec3(0, 0, 1);
    qq = q * q.transpose();
    Imqq = Emat3::Identity() - qq;

    const double dragParaInv = 1 / dragPara;
    const double dragPerpInv = 1 / dragPerp;
    const double dragRotInv = 1 / dragRot;

    mobmat.block<3, 3>(0, 0) = dragParaInv * qq + dragPerpInv * Imqq;
    mobmat.block<3, 3>(3, 3) = dragRotInv * Emat3::Identity();
    return mobmat;
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
    Evec3 q = ECmapq(quaternion.data()) * Evec3(0, 0, 1);

    return std::make_pair<Point, Point>(
        Point{pos[0] - radius, pos[1] - radius, pos[2] - radius}, //
        Point{pos[0] + radius, pos[1] + radius, pos[2] + radius});
  };

  double getVolume() const {
    return 4 * Pi * radius * radius * radius / 3.0 +
           Pi * radius * radius * length / 4.0;
  }

  std::stringstream &parse(std::stringstream &line, //
                           long &gid, bool &immovable,
                           std::array<double, 3> &pos,
                           std::array<double, 4> &quaternion) {
    // C immovable gid radius mx my mz px py pz data

    // required data
    char type, shape;
    double mx, my, mz;
    double px, py, pz;
    line >> shape >> type >> gid >> radius >> mx >> my >> mz >> px >> py >> pz;
    assert(shape == 'C');
    immovable = type == 'T' ? true : false;

    Evec3 minus(mx, my, mz);
    Evec3 plus(px, py, pz);
    Emap3(pos.data()) = 0.5 * (minus + plus);
    Evec3 direction = plus - minus;
    length = direction.norm();
    Emapq(quaternion.data()) =
        Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);

    return line;
  }
};

using Sylinder = Particle<SpherocylinderShape>;

static_assert(std::is_trivially_copyable<Sylinder>::value, "");
static_assert(std::is_default_constructible<Sylinder>::value, "");