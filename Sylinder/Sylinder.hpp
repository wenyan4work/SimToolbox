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

#include "FDPS/particle_simulator.hpp"
#include "Util/EigenDef.hpp"
#include "Util/EquatnHelper.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"
#include "Util/QuadInt.hpp"

#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <vector>

/**
 * @brief specify the link of sylinders
 *
 */
struct Link {
    int group = GEO_INVALID_INDEX; ///< group id of sylinder links
    int prev = GEO_INVALID_INDEX;  ///< previous link in the link group
    int next = GEO_INVALID_INDEX;  ///< next link in the link group
};

/**
 * @brief Sphero-cylinder class
 *
 */
template <int N>
class Sylinder {
  public:
    int gid = GEO_INVALID_INDEX;         ///< unique global id
    int globalIndex = GEO_INVALID_INDEX; ///< unique global index sequentially ordered
    int rank = -1;                       ///< mpi rank

    double radius;          ///< radius
    double radiusCollision; ///< radius for collision resolution
    double length;          ///< length
    double lengthCollision; ///< length for collision resolution
    double radiusSearch;    ///< radiusSearch for short range interactions
    double sepmin;          ///< minimal separation with its neighbors within radiusSearch

    double tg;
    double t;
    double L0;

    Link link; ///< link of this sylinder

    double pos[3];         ///< position
    double orientation[4]; ///< orientation quaternion. direction norm vector = orientation * (0,0,1)

    // vel = velBrown + velCol + velBi + velNonB
    // force =          forceCol + forceBi + forceNonB
    // there is no Brownian force

    // velocity
    double vel[3];        ///< velocity
    double omega[3];      ///< angular velocity
    double velCol[3];     ///< collision velocity
    double omegaCol[3];   ///< collision angular velocity
    double velBi[3];      ///< bilateral constraint velocity
    double omegaBi[3];    ///< bilateral constraint angular velocity
    double velNonB[3];    ///< all non-Brownian deterministic velocity before constraint resolution
    double omegaNonB[3];  ///< all non-Brownian deterministic angular velocity before constraint resolution
    double velHydro[3];   ///< hydrodynamic velocity
    double omegaHydro[3]; ///< hydrodynamic angular velocity

    // force
    double force[3];      ///< force
    double torque[3];     ///< torque
    double forceCol[3];   ///< collision force
    double torqueCol[3];  ///< collision torque
    double forceBi[3];    ///< bilateral constraint force
    double torqueBi[3];   ///< bilateral constraint torque
    double forceNonB[3];  ///< all non-Brownian deterministic force before constraint resolution
    double torqueNonB[3]; ///< all non-Brownian deterministic torque before constraint resolution

    // Brownian displacement
    double velBrown[3];   ///< Brownian velocity
    double omegaBrown[3]; ///< Brownian angular velocity

    // Hydrodynamic quadrature point data
    int numQuadPt;            ///< number of quadrature points
    QuadInt<N> *quadPtr;      ///< pointer to QuadInt object
    double forceHydro[3 * N]; ///< hydrodynamic force at each quadrature point
    double uinfHydro[3 * N];  ///< background flow at each quadrature point (imposed plus those due to rod-rod
                              ///< hydrodynamic interactions)

    // Sylinder Species
    int speciesID = 0;  ///< species id (defaults to uniform species type)

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
    Sylinder(const int &gid_, const double &radius_, const double &radiusCollision_, const double &length_,
             const double &lengthCollision_, const double pos_[3] = nullptr, const double orientation_[4] = nullptr);

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
     * @brief set the hydro data fields to zero
     *
     */
    void clearHydro();

    /**
     * @brief update the position and orientation with internal velocity data fields and given dt
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

    /**
     * @brief Get position as a PS::F64vec3 object
     *
     * necessary interface for FDPS FullParticle class
     * @return PS::F64vec3
     */
    PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }

    /**
     * @brief Set position with given PS::F64vec3 object
     *
     * necessary interface for FDPS FullParticle class
     * @param newPos
     */
    void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }

    /**
     * @brief write to a file*
     *
     * FDPS IO interface
     * @param fptr
     */
    void writeAscii(FILE *fptr) const;

    /**
     * @brief write VTK XML PVTP Header file from rank 0 for single-cell polylines
     *
     * @param prefix
     * @param postfix
     * @param nProcs
     */
    static void writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs);

    /**
     * @brief write VTK XML PVTP Header file from rank 0 for multi-cell polylines
     *
     * @param prefix
     * @param postfix
     * @param nProcs
     */
    static void writePVTPdist(const std::string &prefix, const std::string &postfix, const int nProcs);

    /**
     * @brief write VTK XML binary base64 VTP data file containing single-cell polylines from every MPI rank
     *
     * Procedure for dumping sylinders in the system:
     * Each sylinder writes a polyline with two (connected) points.
     * Points are labeled with float -1 and 1
     * Sylinder data fields are written as cell data
     * Rank 0 writes the parallel header , then each rank write its own serial vtp/vtu file
     *
     * @tparam Container container for local sylinders which supports [] operator
     * @param sylinder
     * @param sylinderNumber
     * @param prefix
     * @param postfix
     * @param rank
     */
    template <class Container>
    static void writeVTP(const Container &sylinder, const int sylinderNumber, const std::string &prefix,
                         const std::string &postfix, int rank) {
        // for each sylinder:

        // write VTP for basic data
        // use float to save some space
        // point and point data
        std::vector<double> pos(6 * sylinderNumber); // position always in Float64
        std::vector<float> label(2 * sylinderNumber);

        // point connectivity of line
        std::vector<int32_t> connectivity(2 * sylinderNumber);
        std::vector<int32_t> offset(sylinderNumber);

        // sylinder data
        std::vector<int> gid(sylinderNumber);
        std::vector<int> numQuadPt(sylinderNumber);
        std::vector<float> radius(sylinderNumber);
        std::vector<float> radiusCollision(sylinderNumber);
        std::vector<float> length(sylinderNumber);
        std::vector<float> lengthCollision(sylinderNumber);
        std::vector<int> group(sylinderNumber);
        std::vector<int> speciesID(sylinderNumber);

        // vel
        std::vector<float> vel(3 * sylinderNumber);
        std::vector<float> omega(3 * sylinderNumber);
        std::vector<float> velCol(3 * sylinderNumber);
        std::vector<float> omegaCol(3 * sylinderNumber);
        std::vector<float> velBi(3 * sylinderNumber);
        std::vector<float> omegaBi(3 * sylinderNumber);
        std::vector<float> velNonB(3 * sylinderNumber);
        std::vector<float> omegaNonB(3 * sylinderNumber);

        // force
        std::vector<float> force(3 * sylinderNumber);
        std::vector<float> torque(3 * sylinderNumber);
        std::vector<float> forceCol(3 * sylinderNumber);
        std::vector<float> torqueCol(3 * sylinderNumber);
        std::vector<float> forceBi(3 * sylinderNumber);
        std::vector<float> torqueBi(3 * sylinderNumber);
        std::vector<float> forceNonB(3 * sylinderNumber);
        std::vector<float> torqueNonB(3 * sylinderNumber);

        // Brownian motion
        std::vector<float> velBrown(3 * sylinderNumber);
        std::vector<float> omegaBrown(3 * sylinderNumber);

        // rigid body orientation
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
            group[i] = sy.link.group;
            speciesID[i] = sy.speciesID;
            numQuadPt[i] = sy.numQuadPt;
            radius[i] = sy.radius;
            radiusCollision[i] = sy.radiusCollision;
            length[i] = sy.length;
            lengthCollision[i] = sy.lengthCollision;

            Evec3 nx = ECmapq(sy.orientation) * Evec3(1, 0, 0);
            Evec3 nz = ECmapq(sy.orientation) * Evec3(0, 0, 1);
            for (int j = 0; j < 3; j++) {
                vel[3 * i + j] = sy.vel[j];
                omega[3 * i + j] = sy.omega[j];
                velCol[3 * i + j] = sy.velCol[j];
                omegaCol[3 * i + j] = sy.omegaCol[j];
                velBi[3 * i + j] = sy.velBi[j];
                omegaBi[3 * i + j] = sy.omegaBi[j];
                velNonB[3 * i + j] = sy.velNonB[j];
                omegaNonB[3 * i + j] = sy.omegaNonB[j];

                force[3 * i + j] = sy.force[j];
                torque[3 * i + j] = sy.torque[j];
                forceCol[3 * i + j] = sy.forceCol[j];
                torqueCol[3 * i + j] = sy.torqueCol[j];
                forceBi[3 * i + j] = sy.forceBi[j];
                torqueBi[3 * i + j] = sy.torqueBi[j];
                forceNonB[3 * i + j] = sy.forceNonB[j];
                torqueNonB[3 * i + j] = sy.torqueNonB[j];

                velBrown[3 * i + j] = sy.velBrown[j];
                omegaBrown[3 * i + j] = sy.omegaBrown[j];

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
        IOHelper::writeDataArrayBase64(group, "group", 1, file);
        IOHelper::writeDataArrayBase64(speciesID, "speciesID", 1, file);        
        IOHelper::writeDataArrayBase64(numQuadPt, "numQuadPt", 1, file);
        IOHelper::writeDataArrayBase64(radius, "radius", 1, file);
        IOHelper::writeDataArrayBase64(radiusCollision, "radiusCollision", 1, file);
        IOHelper::writeDataArrayBase64(length, "length", 1, file);
        IOHelper::writeDataArrayBase64(lengthCollision, "lengthCollision", 1, file);

        IOHelper::writeDataArrayBase64(vel, "vel", 3, file);
        IOHelper::writeDataArrayBase64(omega, "omega", 3, file);
        IOHelper::writeDataArrayBase64(velCol, "velCollision", 3, file);
        IOHelper::writeDataArrayBase64(omegaCol, "omegaCollision", 3, file);
        IOHelper::writeDataArrayBase64(velBi, "velBilateral", 3, file);
        IOHelper::writeDataArrayBase64(omegaBi, "omegaBilateral", 3, file);
        IOHelper::writeDataArrayBase64(velNonB, "velNonBrown", 3, file);
        IOHelper::writeDataArrayBase64(omegaNonB, "omegaNonBrown", 3, file);

        IOHelper::writeDataArrayBase64(force, "force", 3, file);
        IOHelper::writeDataArrayBase64(torque, "torque", 3, file);
        IOHelper::writeDataArrayBase64(forceCol, "forceCollision", 3, file);
        IOHelper::writeDataArrayBase64(torqueCol, "torqueCollision", 3, file);
        IOHelper::writeDataArrayBase64(forceBi, "forceBilateral", 3, file);
        IOHelper::writeDataArrayBase64(torqueBi, "torqueBilateral", 3, file);
        IOHelper::writeDataArrayBase64(forceNonB, "forceNonBrown", 3, file);
        IOHelper::writeDataArrayBase64(torqueNonB, "torqueNonBrown", 3, file);

        IOHelper::writeDataArrayBase64(velBrown, "velBrown", 3, file);
        IOHelper::writeDataArrayBase64(omegaBrown, "omegaBrown", 3, file);

        IOHelper::writeDataArrayBase64(xnorm, "xnorm", 3, file);
        IOHelper::writeDataArrayBase64(znorm, "znorm", 3, file);
        file << "</CellData>\n";
        file << "</Piece>\n";

        IOHelper::writeTailVTP(file);
        file.close();
    }

    /**
     * @brief write VTK XML binary base64 VTP data file containing muli-cell polylines from every MPI rank
     *
     * Procedure for dumping sylinders in the system:
     * Each sylinder writes a polyline with N (connected) points.
     * Points are labeled with evenly spaded float between -0.5 and 0.5
     * Sylinder data fields are written as cell data for each segment
     * Rank 0 writes the parallel header , then each rank write its own serial vtp/vtu file
     *
     * @tparam Container container for local sylinders which supports [] operator
     * @param sylinder
     * @param sylinderNumber
     * @param prefix
     * @param postfix
     * @param rank
     */
    template <class Container>
    static void writeVTPdist(const Container &sylinder, const int sylinderNumber, const std::string &prefix,
                             const std::string &postfix, int rank) {

        // build index
        std::vector<int> rodPts(sylinderNumber, 0);
        std::vector<int> rodPtsIndex(sylinderNumber + 1, 0);
        for (int i = 0; i < sylinderNumber; i++) {
            const auto &sy = sylinder[i];
            rodPts[i] = sy.numQuadPt;
            rodPtsIndex[i + 1] = rodPts[i] + rodPtsIndex[i];
            assert(sy.numQuadPt == sy.quadPtr->getSize());
        }
        const int nPtsLocal = rodPtsIndex.back();

        // for each sylinder:

        // write VTP for data distributed along the sylinder
        // use float to save some space
        // point and point data
        std::vector<double> pos(3 * nPtsLocal); // position always in Float64
        std::vector<float> forceHydro(3 * nPtsLocal);
        std::vector<float> uinfHydro(3 * nPtsLocal);
        std::vector<float> sQuad(nPtsLocal);
        std::vector<float> weightQuad(nPtsLocal);

        // point connectivity of line
        std::vector<int32_t> connectivity(nPtsLocal, 0);
        std::vector<int32_t> offset(sylinderNumber);

#pragma omp parallel for
        for (int i = 0; i < sylinderNumber; i++) {
            const auto &sy = sylinder[i];
            if (!sy.quadPtr) {
                std::cout << "writeVTPdist failed sy.quadPtr undefined" << std::endl;
            }
            Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
            auto quadPtr = sy.quadPtr;
            const auto &sQuadPt = quadPtr->getPoints();
            const auto &wQuadPt = quadPtr->getWeights();
            const int idx = rodPtsIndex[i];
            offset[i] = rodPtsIndex[i + 1];
            // Loop over each point:
            for (int iPoint = 0; iPoint < sy.numQuadPt; iPoint++) {
                connectivity[idx + iPoint] = idx + iPoint;

                // position data
                Evec3 loc = ECmap3(sy.pos) + (sy.length * 0.5 * sQuadPt[iPoint]) * direction;
                pos[(iPoint + idx) * 3 + 0] = loc[0];
                pos[(iPoint + idx) * 3 + 1] = loc[1];
                pos[(iPoint + idx) * 3 + 2] = loc[2];
                // quadrature data
                weightQuad[idx + iPoint] = wQuadPt[iPoint];
                sQuad[idx + iPoint] = sQuadPt[iPoint];

                // sylinder data
                for (int j = 0; j < 3; j++) {
                    forceHydro[3 * (idx + iPoint) + j] = sy.forceHydro[iPoint * 3 + j];
                    uinfHydro[3 * (idx + iPoint) + j] = sy.uinfHydro[iPoint * 3 + j];
                }
            }
        }

        std::ofstream file(prefix + std::string("SylinderDist_") + "r" + std::to_string(rank) + std::string("_") +
                               postfix + std::string(".vtp"),
                           std::ios::out);

        IOHelper::writeHeadVTP(file);

        file << "<Piece NumberOfPoints=\"" << nPtsLocal << "\" NumberOfLines=\"" << sylinderNumber << "\">\n";
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
        IOHelper::writeDataArrayBase64(sQuad, "sQuad", 1, file);
        IOHelper::writeDataArrayBase64(weightQuad, "weightQuad", 1, file);
        IOHelper::writeDataArrayBase64(forceHydro, "forceHydro", 3, file);
        IOHelper::writeDataArrayBase64(uinfHydro, "uinfHydro", 3, file);
        file << "</PointData>\n";
        // cell data
        file << "<CellData Scalars=\"scalars\">\n";
        file << "</CellData>\n";
        file << "</Piece>\n";

        IOHelper::writeTailVTP(file);
        file.close();
    }
};

/**
 * @brief FDPS writeAscii file header
 */
class SylinderAsciiHeader {
  public:
    int nparticle;
    double time;
    void writeAscii(FILE *fp) const { fprintf(fp, "%d \n %lf\n", nparticle, time); }
};

static_assert(std::is_trivially_copyable<Sylinder<10>>::value, "");
static_assert(std::is_default_constructible<Sylinder<10>>::value, "");

// Include the Sylinder implimentation
#include "Sylinder.tpp"
#endif