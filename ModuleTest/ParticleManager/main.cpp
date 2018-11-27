#include "MPI/ParticleManager.hpp"
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

void initTubule(std::vector<Tubule> &tubule) {
    const int nT = tubule.size();

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

void initMotor(std::vector<Motor> &motor) {
    const int nM = motor.size();

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
    {
        constexpr int npar = 10;
        ParticleManager<Tubule, Motor> parManager(argc, argv);

        int flag;
        MPI_Initialized(&flag);
        std::cout << "MPI Initialized? " << flag << std::endl;

        std::vector<Tubule> tubuleVec(npar);
        std::vector<Motor> motorVec(npar);

        parManager.initSpecies<0>(&tubuleVec);
        parManager.initSpecies<1>(&motorVec);

        initTubule(tubuleVec);
        initMotor(motorVec);

        parManager.showSpecies<0>();
        parManager.showSpecies<1>();

        parManager.clearIndex();
        parManager.addSpeciesToIndex<0>();
        parManager.addSpeciesToIndex<1>();

        bool pbc[3] = {true, true, true};
        double boxLow[3] = {0, 0, 0};
        double boxHigh[3] = {10, 20, 30};
        parManager.setBox(pbc, boxLow, boxHigh);

        parManager.dumpSystem();
    }
    MPI_Finalize();
    return 0;
}