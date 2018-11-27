#include "MPI/MixPairInteraction.hpp"
#include "Util/TRngPool.hpp"
#include <type_traits>

struct Tubule {
    int gid;
    // int species;
    double RSearch = 1;
    double pos[3] = {1, 1, 1};

    inline int getGid() const { return gid; }
    inline double getRSearch() const { return RSearch; }
    inline const double *getPos() const { return pos; }
};

struct Motor {
    int gid;
    // int species;
    double RSearch;
    double pos[3];

    inline int getGid() const { return gid; }
    inline double getRSearch() const { return RSearch; }
    inline const double *getPos() const { return pos; }
};

void initTubule(PS::ParticleSystem<Tubule> &tubule) {
    const int nT = tubule.getNumberOfParticleLocal();

    TRngPool rngPool(nT);

#pragma omp parallel for
    for (int i = 0; i < nT; i++) {
        tubule[i].gid = i;

        tubule[i].RSearch = rngPool.getU01() / 20;
        tubule[i].pos[0] = rngPool.getU01();
        tubule[i].pos[1] = rngPool.getU01();
        tubule[i].pos[2] = rngPool.getU01();
    }
}

void initMotor(PS::ParticleSystem<Motor> &motor) {
    const int nM = motor.getNumberOfParticleLocal();

    TRngPool rngPool(nM);

#pragma omp parallel for
    for (int i = 0; i < nM; i++) {
        motor[i].gid = i;

        motor[i].RSearch = rngPool.getU01() / 20;
        motor[i].pos[0] = rngPool.getU01();
        motor[i].pos[1] = rngPool.getU01();
        motor[i].pos[2] = rngPool.getU01();
    }
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    PS::Initialize(argc, argv);

    PS::ParticleSystem<Tubule> sysTubule;
    PS::ParticleSystem<Motor> sysMotor;
    initTubule(sysTubule);
    initMotor(sysMotor);

    PS::DomainInfo dinfo;

    dinfo.initialize();
    dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XYZ);
    // rootdomain must be specified after PBC
    dinfo.setPosRootDomain(PS::F64vec3(0, 0, 0), PS::F64vec3(1, 1, 1));

    PS::Finalize();
    MPI_Finalize();
    return 0;
}