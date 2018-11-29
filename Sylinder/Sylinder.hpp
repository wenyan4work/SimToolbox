#ifndef SYLINDER_HPP_
#define SYLINDER_HPP_

#include "MPI/FDPS/particle_simulator.hpp"
#include "Util/Buffer.hpp"
#include "Util/EigenDef.hpp"
#include "Util/EquatnHelper.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"

#include <cstdio>
#include <vector>

#include <type_traits>
#include <unordered_map>
#include <vector>

class Sylinder {
  public:
    int gid = GEO_INVALID_INDEX;
    int globalIndex = GEO_INVALID_INDEX;
    double radius;
    double radiusCollision;
    double length;
    double lengthCollision;
    double radiusSearch;

    double pos[3];
    double vel[3];
    double omega[3];
    double orientation[4];

    // these are not packed and transferred
    double velCol[3];
    double omegaCol[3];
    double velBrown[3];
    double omegaBrown[3];
    double velNonB[3]; // all non-Brownian deterministic motion beforce collision
    double omegaNonB[3];

    Sylinder() = default;
    ~Sylinder() = default;

    Sylinder(const int &gid_, const double &radius_, const double &radiusCollision_, const double &length_,
             const double &lengthCollision_, const double pos_[3] = nullptr, const double orientation_[4] = nullptr);

    Sylinder(const Sylinder &) = default;
    Sylinder &operator=(const Sylinder &) = default;

    Sylinder(Sylinder &&) = default;
    Sylinder &operator=(Sylinder &&) = default;

    void dumpSylinder() const;

    void clear();

    // motion
    void stepEuler(double dt);

    // necessary interface for InteractionManager.hpp
    const double *Coord() const { return pos; }

    double Rad() const { return radiusCollision * 4; }

    void Pack(std::vector<char> &buff) const;

    void Unpack(const std::vector<char> &buff);

    // necessary interface for FDPS FullParticle class
    int getGid() const { return gid; }
    PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }

    // Output to VTK
    static void writeVTP(const std::vector<Sylinder> &sylinder, const std::string &prefix, const std::string &postfix,
                         int rank);
    static void writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs);
};

static_assert(std::is_trivially_copyable<Sylinder>::value, "");
static_assert(std::is_default_constructible<Sylinder>::value, "");

#endif