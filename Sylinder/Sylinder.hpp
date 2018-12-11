#ifndef SYLINDER_HPP_
#define SYLINDER_HPP_

#include "FDPS/particle_simulator.hpp"
#include "Util/Buffer.hpp"
#include "Util/EigenDef.hpp"
#include "Util/EquatnHelper.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"

#include <cstdio>
#include <cstdlib>
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
    double sepmin;

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
    // FDPS IO interface
    void writeAscii(FILE *fptr) const;

    // Output to VTK
    // PVTP Header file from rank 0
    static void writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs);

    // PVTP data file from every MPI rank
    template <class Container>
    static void writeVTP(const Container &sylinder, const int sylinderNumber, const std::string &prefix,
                         const std::string &postfix, int rank) {
        // for each sylinder:
        /*
         Procedure for dumping sylinders in the system:
        each sylinder writes a polyline with two (connected) points. points are labeled with float -1 and 1
        sylinder data written as cell data
        1 basic data (x,y,z), r, rcol, v, omega, etc output as VTK POLY_DATA file (*.vtp), each partiel is a VTK_VERTEX
        For each dataset, rank 0 writes the parallel header , then each rank write its own serial vtp/vtu file
        */

        // write VTP for basic data
        //  use float to save some space
        // point and point data
        std::vector<double> pos(6 * sylinderNumber); // position always in Float64
        std::vector<float> label(2 * sylinderNumber);

        // point connectivity of line
        std::vector<int32_t> connectivity(2 * sylinderNumber);
        std::vector<int32_t> offset(sylinderNumber);

        // sylinder data
        std::vector<int> gid(sylinderNumber);
        std::vector<float> radius(sylinderNumber);
        std::vector<float> radiusCollision(sylinderNumber);
        std::vector<float> length(sylinderNumber);
        std::vector<float> lengthCollision(sylinderNumber);
        std::vector<float> vel(3 * sylinderNumber);
        std::vector<float> omega(3 * sylinderNumber);
        std::vector<float> velCol(3 * sylinderNumber);
        std::vector<float> omegaCol(3 * sylinderNumber);
        std::vector<float> velNonB(3 * sylinderNumber);
        std::vector<float> omegaNonB(3 * sylinderNumber);
        std::vector<float> velBrown(3 * sylinderNumber);
        std::vector<float> omegaBrown(3 * sylinderNumber);
        std::vector<float> xnorm(3 * sylinderNumber);
        std::vector<float> znorm(3 * sylinderNumber);

#pragma omp parallel for
        for (int i = 0; i < sylinderNumber; i++) {
            const auto &sy = sylinder[i];
            // point and point data
            Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
            Evec3 end0 = ECmap3(sy.pos) - direction * (sy.length * 0.5);
            Evec3 end1 = ECmap3(sy.pos) + direction * (sy.length * 0.5);
            pos[6 * i] = end0[0];
            pos[6 * i + 1] = end0[1];
            pos[6 * i + 2] = end0[2];
            pos[6 * i + 3] = end1[0];
            pos[6 * i + 4] = end1[1];
            pos[6 * i + 5] = end1[2];
            label[2 * i] = -1;
            label[2 * i + 1] = 1;

            // connectivity
            connectivity[2 * i] = 2 * i;         // index of point 0 in line
            connectivity[2 * i + 1] = 2 * i + 1; // index of point 1 in line
            offset[i] = 2 * i + 2;               // offset is the end of each line. in fortran indexing

            // sylinder data
            gid[i] = sy.gid;
            radius[i] = sy.radius;
            radiusCollision[i] = sy.radiusCollision;
            length[i] = sy.length;
            lengthCollision[i] = sy.lengthCollision;

            Evec3 nx = ECmapq(sy.orientation) * Evec3(1, 0, 0);
            Evec3 nz = ECmapq(sy.orientation) * Evec3(0, 0, 1);
            for (int j = 0; j < 3; j++) {
                vel[3 * i + j] = sy.vel[j];
                omega[3 * i + j] = sy.omega[j];
                velBrown[3 * i + j] = sy.velBrown[j];
                omegaBrown[3 * i + j] = sy.omegaBrown[j];
                velCol[3 * i + j] = sy.velCol[j];
                omegaCol[3 * i + j] = sy.omegaCol[j];
                velNonB[3 * i + j] = sy.velNonB[j];
                omegaNonB[3 * i + j] = sy.omegaNonB[j];

                xnorm[3 * i + j] = nx[j];
                znorm[3 * i + j] = nz[j];
            }
        }

        std::ofstream file(prefix + std::string("Sylinder_") + "r" + std::to_string(rank) + std::string("_") + postfix +
                               std::string(".vtp"),
                           std::ios::out);

        IOHelper::writeHeadVTP(file);

        file << "<Piece NumberOfPoints=\"" << sylinderNumber * 2 << "\" NumberOfLines=\"" << sylinderNumber << "\">\n";
        // Points
        file << "<Points>\n";
        IOHelper::writeDataArrayBase64(pos, "position", 3, file);
        file << "</Points>\n";
        // cell definition
        file << "<Lines>\n";
        IOHelper::writeDataArrayBase64(connectivity, "connectivity", 1, file);
        IOHelper::writeDataArrayBase64(offset, "offsets", 1, file);
        file << "</Lines>\n";
        // point data
        file << "<PointData Scalars=\"scalars\">\n";
        IOHelper::writeDataArrayBase64(label, "endLabel", 1, file);
        file << "</PointData>\n";
        // cell data
        file << "<CellData Scalars=\"scalars\">\n";
        IOHelper::writeDataArrayBase64(gid, "gid", 1, file);
        IOHelper::writeDataArrayBase64(radius, "radius", 1, file);
        IOHelper::writeDataArrayBase64(radiusCollision, "radiusCollision", 1, file);
        IOHelper::writeDataArrayBase64(length, "length", 1, file);
        IOHelper::writeDataArrayBase64(lengthCollision, "lengthCollision", 1, file);
        IOHelper::writeDataArrayBase64(vel, "velocity", 3, file);
        IOHelper::writeDataArrayBase64(omega, "omega", 3, file);
        IOHelper::writeDataArrayBase64(velBrown, "velocityBrown", 3, file);
        IOHelper::writeDataArrayBase64(omegaBrown, "omegaBrown", 3, file);
        IOHelper::writeDataArrayBase64(velCol, "velocityCollision", 3, file);
        IOHelper::writeDataArrayBase64(omegaCol, "omegaCollision", 3, file);
        IOHelper::writeDataArrayBase64(velNonB, "velocityNonB", 3, file);
        IOHelper::writeDataArrayBase64(omegaNonB, "omegaNonB", 3, file);
        IOHelper::writeDataArrayBase64(xnorm, "xnorm", 3, file);
        IOHelper::writeDataArrayBase64(znorm, "znorm", 3, file);
        file << "</CellData>\n";
        file << "</Piece>\n";

        IOHelper::writeTailVTP(file);
        file.close();
    }
};

// for FDPS writeAscii file header
class SylinderAsciiHeader {
  public:
    int nparticle;
    double time;
    void writeAscii(FILE *fp) const { fprintf(fp, "%d \n %lf\n", nparticle, time); }
};

static_assert(std::is_trivially_copyable<Sylinder>::value, "");
static_assert(std::is_default_constructible<Sylinder>::value, "");

#endif