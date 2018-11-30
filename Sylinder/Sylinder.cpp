#include "Sylinder.hpp"
#include "Util/Base64.hpp"

/*****************************************************
 *  Sphero-cylinder
 ******************************************************/

Sylinder::Sylinder(const int &gid_, const double &radius_, const double &radiusCollision_, const double &length_,
                   const double &lengthCollision_, const double pos_[3], const double orientation_[4]) {
    gid = gid_;
    radius = radius_;
    radiusCollision = radiusCollision_;
    length = length_;
    lengthCollision = lengthCollision_;
    if (pos_ == nullptr) {
        Emap3(pos).setZero();
    } else {
        for (int i = 0; i < 3; i++) {
            pos[i] = pos_[i];
        }
    }
    if (orientation_ == nullptr) {
        Emapq(orientation).setIdentity();
    } else {
        for (int i = 0; i < 4; i++) {
            orientation[i] = orientation_[i];
        }
    }

    clear();
    return;
}

void Sylinder::clear() {
    Emap3(vel).setZero();
    Emap3(omega).setZero();
    Emap3(velCol).setZero();
    Emap3(omegaCol).setZero();
    Emap3(velBrown).setZero();
    Emap3(omegaBrown).setZero();
    Emap3(velNonB).setZero();
    Emap3(omegaNonB).setZero();
}

void Sylinder::dumpSylinder() const {
    printf("gid %8d, r %8f, rCol %8f, l %8f, lCol %8f, pos %8f, %8f, %8f\n", gid, radius, radiusCollision, length,
           lengthCollision, pos[0], pos[1], pos[2]);
    printf("vel %8f, %8f, %8f; omega %8f, %8f, %8f\n", vel[0], vel[1], vel[2], omega[0], omega[1], omega[2]);
    printf("orient %8f, %8f, %8f, %8f\n", orientation[0], orientation[1], orientation[2], orientation[3]);
}

void Sylinder::Pack(std::vector<char> &buff) const {
    Buffer buffer(buff);
    // head
    buffer.pack(std::string("SYLINDER"));
    // fixed size data
    buffer.pack(gid);                                                 // int gid = INVALID;
    buffer.pack(globalIndex);                                         // int gid = INVALID;
    buffer.pack(radius);                                              // double radius;
    buffer.pack(radiusCollision);                                     // double radiusCollision;
    buffer.pack(length);                                              // double ;
    buffer.pack(lengthCollision);                                     // double ;
    buffer.pack(radiusSearch);                                        // double
    buffer.pack(std::array<double, 3>{pos[0], pos[1], pos[2]});       // Evec3 pos;
    buffer.pack(std::array<double, 3>{vel[0], vel[1], vel[2]});       // Evec3 vel;
    buffer.pack(std::array<double, 3>{omega[0], omega[1], omega[2]}); // Evec3 omega;
    buffer.pack(std::array<double, 4>{orientation[0], orientation[1], orientation[2], orientation[3]});
}

void Sylinder::Unpack(const std::vector<char> &buff) {
    Buffer buffer;
    // head
    std::string strbuf;
    buffer.unpack(strbuf, buff);
    assert(strbuf == std::string("SYLINDER"));
    // fixed size data
    buffer.unpack(gid, buff);             // int gid = INVALID;
    buffer.unpack(globalIndex, buff);     // int gid = INVALID;
    buffer.unpack(radius, buff);          // double radius;
    buffer.unpack(radiusCollision, buff); // double radiusCollision;
    buffer.unpack(length, buff);          // double
    buffer.unpack(lengthCollision, buff); // double
    buffer.unpack(radiusSearch, buff);    // double
    std::array<double, 3> array3;
    buffer.unpack(array3, buff); // Evec3 pos;
    pos[0] = array3[0];
    pos[1] = array3[1];
    pos[2] = array3[2];
    buffer.unpack(array3, buff); // Evec3 vel;
    vel[0] = array3[0];
    vel[1] = array3[1];
    vel[2] = array3[2];
    buffer.unpack(array3, buff); // Evec3 omega;
    omega[0] = array3[0];
    omega[1] = array3[1];
    omega[2] = array3[2];
    std::array<double, 4> array4;
    buffer.unpack(array4, buff); // Equatn orientation;
    orientation[0] = array4[0];
    orientation[1] = array4[1];
    orientation[2] = array4[2];
    orientation[3] = array4[3];
}

void Sylinder::writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs) {
    std::vector<std::string> pieceNames;

    std::vector<IOHelper::FieldVTU> pointDataFields;
    pointDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "endLabel");

    std::vector<IOHelper::FieldVTU> cellDataFields;
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Int32, "gid");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "radius");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "radiusCollision");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "length");
    cellDataFields.emplace_back(1, IOHelper::IOTYPE::Float32, "lengthCollision");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "velocity");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "omega");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "velocityBrown");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "omegaBrown");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "velocityCollision");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "omegaCollision");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "velocityHydro");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "omegaHydro");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "xnorm");
    cellDataFields.emplace_back(3, IOHelper::IOTYPE::Float32, "znorm");

    for (int i = 0; i < nProcs; i++) {
        pieceNames.emplace_back(std::string("Sylinder_") + std::string("r") + std::to_string(i) + "_" + postfix +
                                ".vtp");
    }

    IOHelper::writePVTPFile(prefix + "Sylinder_" + postfix + ".pvtp", pointDataFields, cellDataFields, pieceNames);
}

void Sylinder::stepEuler(double dt) {
    Emap3(pos) += Emap3(vel) * dt;
    Equatn currOrient = Emapq(orientation);
    EquatnHelper::rotateEquatn(currOrient, Emap3(omega), dt);
    Emapq(orientation).x() = currOrient.x();
    Emapq(orientation).y() = currOrient.y();
    Emapq(orientation).z() = currOrient.z();
    Emapq(orientation).w() = currOrient.w();
}