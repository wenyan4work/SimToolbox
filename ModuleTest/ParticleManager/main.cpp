#include "MPI/ParticleManager.hpp"

struct Tubule {
    int gid;
    double RSearch;
    double pos[3];

    inline int getGid() { return gid; }
    inline double getRSearch() { return RSearch; }
};

struct Motor {
    int gid;
    double RSearch;
    double pos[3];

    inline int getGid() { return gid; }
    inline double getRSearch() { return RSearch; }
};

int main() {
    constexpr int npar = 10;
    ParticleManager<Tubule, Motor> parManager;
    std::vector<Tubule> tubuleVec(npar);
    std::vector<Motor> motorVec(npar);

    parManager.initKind<0>(&tubuleVec);
    parManager.initKind<1>(&motorVec);

    parManager.showKind<0>();
    parManager.showKind<1>();

    return 0;
}