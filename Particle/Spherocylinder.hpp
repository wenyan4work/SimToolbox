#ifndef SPHEROCYLINDER_HPP_
#define SPHEROCYLINDER_HPP_

#include "Particle.hpp"

/**
 * @brief concrete particle class of spherosylinder
 *
 */
struct SpherocylinderShape {
  double radius = 1.0; ///< radius
  double length = 1.0; ///< length

  MSGPACK_DEFINE(radius, length);

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
  Emat6 getMobMat(const std::array<double, 4> &quaternion,
                  const double mu = 1) const {
    const double b = -(1 + 2 * log(radius / (length)));
    const double dragPara = 8 * Pi * mu * length / (2 * b);
    const double dragPerp = 8 * Pi * mu * length / (b + 2);
    const double dragRot =
        2 * Pi * mu * length * length * length / (3 * (b + 2));
    const double dragParaInv = 1 / dragPara;
    const double dragPerpInv = 1 / dragPerp;
    const double dragRotInv = 1 / dragRot;

    Emat6 mobmat = Emat6::Zero();
    Emat3 qq;
    Emat3 Imqq;
    Evec3 q = ECmapq(quaternion.data()) * Evec3(0, 0, 1);
    qq = q * q.transpose();
    Imqq = Emat3::Identity() - qq;

    mobmat.block<3, 3>(0, 0) = dragParaInv * qq + dragPerpInv * Imqq;
    mobmat.block<3, 3>(3, 3) = dragRotInv * Emat3::Identity();
    return mobmat;
  };

  Evec6 getVelBrown(const std::array<double, 4> &quaternion,
                    const std::array<double, 12> &rngN01s, const double dt,
                    const double kBT, const double mu = 1) const {
    const double delta = dt * 0.1; // a small parameter used in RFD algorithm
    const double kBTfactor = sqrt(2 * kBT / dt);
    const double b = -(1 + 2 * log(radius / (length)));
    const double dragPara = 8 * Pi * mu * length / (2 * b);
    const double dragPerp = 8 * Pi * mu * length / (b + 2);
    const double dragRot =
        2 * Pi * mu * length * length * length / (3 * (b + 2));
    const double dragParaInv = 1 / dragPara;
    const double dragPerpInv = 1 / dragPerp;
    const double dragRotInv = 1 / dragRot;

    Evec3 direction = ECmapq(quaternion.data()) * Evec3(0, 0, 1);

    // RFD from Delong, JCP, 2015
    // slender fiber has 0 rot drag, regularize with identity rot mobility
    // trans mobility is this
    Evec3 q = direction;
    Emat3 Nmat = (dragParaInv - dragPerpInv) * (q * q.transpose()) +
                 (dragPerpInv)*Emat3::Identity();
    Emat3 Nmatsqrt = Nmat.llt().matrixL();

    Evec3 Wrot(rngN01s[0], rngN01s[1], rngN01s[2]);
    Evec3 Wpos(rngN01s[3], rngN01s[4], rngN01s[5]);
    Evec3 Wrfdrot(rngN01s[6], rngN01s[7], rngN01s[8]);
    Evec3 Wrfdpos(rngN01s[9], rngN01s[10], rngN01s[11]);

    Equatn orientRFD = ECmapq(quaternion.data());
    EquatnHelper::rotateEquatn(orientRFD, Wrfdrot, delta);
    q = orientRFD * Evec3(0, 0, 1);
    Emat3 Nmatrfd = (dragParaInv - dragPerpInv) * (q * q.transpose()) +
                    (dragPerpInv)*Emat3::Identity();
    Evec6 vel;
    // Gaussian noise
    vel.segment<3>(0) = kBTfactor * (Nmatsqrt * Wpos);
    // rfd drift. seems no effect in this case
    vel.segment<3>(0) += (kBT / delta) * ((Nmatrfd - Nmat) * Wrfdpos);
    // regularized identity rotation drag
    vel.segment<3>(3) = sqrt(dragRotInv) * kBTfactor * Wrot;
    return vel;
  }

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

  /**
   * @brief Get volume of spherocylinder
   *
   * @return double
   */
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

#endif