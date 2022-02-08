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

  Emat6 getMobMat(const std::array<double, 4> &quaternion,
                  const double mu = 1) const {
    const double drag = 6 * Pi * mu * radius;
    const double dragRot = 8 * Pi * mu * radius * radius * radius;

    Emat6 mobmat = Emat6::Identity();
    for (int k = 0; k < 3; k++) {
      mobmat(k, k) = 1 / drag;
      mobmat(3 + k, 3 + k) = 1 / dragRot;
    }

    return mobmat;
  };

  Evec6 getVelBrown(const std::array<double, 4> &quaternion,
                    const std::array<double, 12> &rngN01s, const double dt,
                    const double kBT, const double mu = 1) const {
    const double delta = dt * 0.1; // a small parameter used in RFD algorithm
    const double kBTfactor = sqrt(2 * kBT / dt);
    const double drag = 6 * Pi * mu * radius;
    const double dragRot = 8 * Pi * mu * radius * radius * radius;
    const double dragInv = 1 / drag;
    const double dragRotInv = 1 / dragRot;

    Evec3 direction = ECmapq(quaternion.data()) * Evec3(0, 0, 1);

    // RFD from Delong, JCP, 2015
    // slender fiber has 0 rot drag, regularize with identity rot mobility
    // trans mobility is this
    Evec3 q = direction;
    Emat3 Nmat = dragInv * Emat3::Identity();
    Emat3 Nmatsqrt = Nmat.llt().matrixL();

    Evec3 Wrot(rngN01s[0], rngN01s[1], rngN01s[2]);
    Evec3 Wpos(rngN01s[3], rngN01s[4], rngN01s[5]);
    Evec3 Wrfdrot(rngN01s[6], rngN01s[7], rngN01s[8]);
    Evec3 Wrfdpos(rngN01s[9], rngN01s[10], rngN01s[11]);

    Equatn orientRFD = ECmapq(quaternion.data());
    EquatnHelper::rotateEquatn(orientRFD, Wrfdrot, delta);
    q = orientRFD * Evec3(0, 0, 1);
    Emat3 Nmatrfd = dragInv * Emat3::Identity();
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