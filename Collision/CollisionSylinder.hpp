#ifndef COLLISIONSYLINDER_HPP_
#define COLLISIONSYLINDER_HPP_

#include "CollisionCollector.hpp"
#include "DCPQuery.hpp"
#include "Sylinder/Sylinder.hpp"
#include "Util/Buffer.hpp"
#include "Util/EigenDef.hpp"
#include "Util/GeoCommon.h"

class CollisionSylinder {
    // this is a POD type, used to generate collision blocks
  public:
    int gid = GEO_INVALID_INDEX;
    int globalIndex = GEO_INVALID_INDEX;
    double radiusCollision;
    double lengthCollision;
    double radiusSearch;
    Evec3 pos;
    Equatn orientation;
    // Evec3 direction;

    DCPQuery<3, double, EAvec3> DistSegSeg3; // not transferred over mpi

    void CopyFromFull(const Sylinder &s) {
        gid = s.gid;
        globalIndex = s.globalIndex;
        radiusCollision = s.radiusCollision;
        lengthCollision = s.lengthCollision;
        radiusSearch = s.radiusSearch;
        pos = s.pos;
        orientation = s.orientation;
    }

    // necessary interface for Near Interaction
    const double *Coord() const { return pos.data(); }

    double Rad() const { return radiusSearch; }

    void Pack(std::vector<char> &buff) const {
        Buffer mybuff(buff);
        mybuff.pack(gid);
        mybuff.pack(globalIndex);
        mybuff.pack(radiusCollision);
        mybuff.pack(lengthCollision);
        mybuff.pack(radiusSearch);
        mybuff.pack(pos[0]);
        mybuff.pack(pos[1]);
        mybuff.pack(pos[2]);
        mybuff.pack(orientation.w());
        mybuff.pack(orientation.x());
        mybuff.pack(orientation.y());
        mybuff.pack(orientation.z());
    }

    void Unpack(const std::vector<char> &buff) {
        Buffer mybuff;
        mybuff.unpack(gid, buff);
        mybuff.unpack(globalIndex, buff);
        mybuff.unpack(radiusCollision, buff);
        mybuff.unpack(lengthCollision, buff);
        mybuff.unpack(radiusSearch, buff);
        mybuff.unpack(pos[0], buff);
        mybuff.unpack(pos[1], buff);
        mybuff.unpack(pos[2], buff);
        mybuff.unpack(orientation.w(), buff);
        mybuff.unpack(orientation.x(), buff);
        mybuff.unpack(orientation.y(), buff);
        mybuff.unpack(orientation.z(), buff);
    }

    inline bool collide(const CollisionSylinder &sJ, CollisionBlock &block,
                        const std::array<double, 3> &srcShift = std::array<double, 3>{0.0, 0.0, 0.0}) {
        if (gid >= sJ.gid) {
            // no self collisio
            // do not record gid > J.gidn
            return false;
        }
        const auto &sI = *this;
        EAvec3 posI = sI.pos;
        EAvec3 posJ = sJ.pos;
        posJ[0] += srcShift[0];
        posJ[1] += srcShift[1];
        posJ[2] += srcShift[2];
        EAvec3 dirI = orientation * EAvec3(0, 0, 1);
        EAvec3 dirJ = sJ.orientation * EAvec3(0, 0, 1);

        const EAvec3 Pm = posI - (0.5 * sI.lengthCollision) * dirI;
        const EAvec3 Pp = posI + (0.5 * sI.lengthCollision) * dirI;

        const EAvec3 Qm = posJ - (0.5 * sJ.lengthCollision) * dirJ;
        const EAvec3 Qp = posJ + (0.5 * sJ.lengthCollision) * dirJ;

        // location
        EAvec3 Ploc = Evec3::Zero(), Qloc = Evec3::Zero();
        double s = 0, t = 0;

        // check collision for line segment I and J
        const double distMin = DistSegSeg3(Pm, Pp, Qm, Qp, Ploc, Qloc, s, t);
        const double sep = distMin - (sI.radiusCollision + sJ.radiusCollision);

        // save collision block
        // save only block gidI < gidJ
        if (sep < GEO_DEFAULT_COLBUF * (radiusCollision + sJ.radiusCollision)) {
            // collision
            block.normI = (Ploc - Qloc).normalized();
            block.normJ = -block.normI;
            block.phi0 = sep;
            block.gidI = gid;
            block.gidJ = sJ.gid;
            block.globalIndexI = globalIndex;
            block.globalIndexJ = sJ.globalIndex;
            block.posI = Ploc - posI; // collision location relative to geometric center
            block.posJ = Qloc - posJ;
            block.gamma = sep < 0 ? -sep : 0; // a crude initial guess

            collideStress(orientation, sJ.orientation, posI, posJ, lengthCollision, sJ.lengthCollision, radiusCollision,
                          sJ.radiusCollision, 1.0, Ploc, Qloc, block.stress);

            return true;
        } else {
            // no collision
            return false;
        }
    }

    void InitializeSyN(Emat3 &NI, double r, double h, double rho) {
        NI.setZero();
        double beta = h / 2.0 / r;
        NI(0, 0) = 1.0 / 30.0 * (15.0 * beta + 8);
        NI(1, 1) = NI(0, 0);
        NI(2, 2) = 1.0 / 15.0 * (10.0 * beta * beta * beta + 20.0 * beta * beta + 15.0 * beta + 4.0);
        NI = NI * rho * r * r * r * r * r * M_PI;
    }

    void InitializeSyGA(Emat3 &GAMMAI, double r, double h, double rho) {
        GAMMAI.setZero();
        double beta = h / 2.0 / r;
        GAMMAI(0, 0) = 1.0 / 30.0 * (20.0 * beta * beta * beta + 40.0 * beta * beta + 45.0 * beta + 16.0);
        GAMMAI(1, 1) = GAMMAI(0, 0);
        GAMMAI(2, 2) = 1.0 / 15.0 * (15 * beta + 8);
        GAMMAI = GAMMAI * M_PI * r * r * r * r * r * rho;
    }

    void collideStress(const Equatn &qI, const Equatn &qJ, const Evec3 &posI, const Evec3 &posJ, double hI, double hJ,
                       const double rI, const double rJ, const double rho, const Evec3 &Ploc, const Evec3 &Qloc,
                       Emat3 &StressIJ) {

        Emat3 NI, GAMMAI, NJ, GAMMAJ, InvGAMMAI, InvGAMMAJ;
        InitializeSyN(NI, rI, hI, rho);
        InitializeSyGA(GAMMAI, rI, hI, rho);
        InitializeSyN(NJ, rJ, hJ, rho);
        InitializeSyGA(GAMMAJ, rJ, hJ, rho);

        Emat3 strIJ;

        Emat3 RI = qI.toRotationMatrix();
        Emat3 RJ = qJ.toRotationMatrix();
        Evec3 dirI = qI * Evec3(0, 0, 1);
        Evec3 dirJ = qJ * Evec3(0, 0, 1);

        // compute xsurP and xsurQ

        Evec3 PlocQloc = (Qloc - Ploc).normalized();
        double Ploc2surf = dirI.dot(PlocQloc);
        Evec3 xsurP = Ploc + PlocQloc * rI / sqrt(1 - Ploc2surf * Ploc2surf);

        double Qloc2surf = dirJ.dot(PlocQloc);
        Evec3 xsurQ = Qloc - PlocQloc * rJ / sqrt(1 - Qloc2surf * Qloc2surf);

        Evec3 xcI = xsurP - posI;
        Evec3 xcJ = xsurQ - posJ;

        Evec3 F1 = (xsurP - xsurQ).normalized();

        double aI = NI(0, 0), bI = NI(2, 2), aJ = NJ(0, 0), bJ = NJ(2, 2);

        NI = aI * Emat3::Identity() + (bI - aI) * (dirI * dirI.transpose());
        NJ = aJ * Emat3::Identity() + (bJ - aJ) * (dirJ * dirJ.transpose());

        aI = 1.0 / GAMMAI(0, 0);
        bI = 1.0 / GAMMAI(2, 2);
        aJ = 1.0 / GAMMAJ(0, 0);
        bJ = 1.0 / GAMMAJ(2, 2);

        InvGAMMAI = aI * Emat3::Identity() + (bI - aI) * (dirI * dirI.transpose());
        InvGAMMAJ = aJ * Emat3::Identity() + (bJ - aJ) * (dirJ * dirJ.transpose());

        auto rIf = (posI) * (F1.transpose());
        auto rJf = -(posJ) * (F1.transpose());
        auto xICf = (xcI).cross(F1);
        auto xJCf = (F1).cross(xcJ);

        // Levi-Civita symbol epsilon

        double epsilon[3][3][3];

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    epsilon[i][j][k] = 0.0;
                }
            }
        }

        epsilon[0][1][2] = 1.0;
        epsilon[1][2][0] = 1.0;
        epsilon[2][0][1] = 1.0;
        epsilon[1][0][2] = -1.0;
        epsilon[2][1][0] = -1.0;
        epsilon[0][2][1] = -1.0;

        Emat3 SGI, SGJ;
        SGI.setZero();
        SGJ.setZero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    for (int l = 0; l < 3; l++) {
                        for (int r = 0; r < 3; r++) {
                            SGI(i, j) += NI(i, l) * epsilon[j][k][l] * InvGAMMAI(k, r) * xICf(r);
                            SGJ(i, j) += NJ(i, l) * epsilon[j][k][l] * InvGAMMAJ(k, r) * xJCf(r);
                        }
                    }
                }
            }
        }

        StressIJ = rIf + rJf + SGI + SGJ;
    }
};

#endif