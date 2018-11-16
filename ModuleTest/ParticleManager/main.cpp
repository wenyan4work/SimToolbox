#include "MPI/ParticleManager.hpp"

struct Tubule {
    int gid;
    int kind;
    double RSearch = 1;
    double pos[3] = {1, 1, 1};

    inline int getGid() { return gid; }
    inline int getKind() { return kind; }
    inline double getRSearch() { return RSearch; }
    inline double *getPos() { return pos; }
};

struct Motor {
    int gid;
    int kind;
    double RSearch;
    double pos[3];

    inline int getGid() { return gid; }
    inline int getKind() { return kind; }
    inline double getRSearch() { return RSearch; }
    inline double *getPos() { return pos; }
};

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

        parManager.initKind<0>(&tubuleVec);
        parManager.initKind<1>(&motorVec);

        parManager.showKind<0>();
        parManager.showKind<1>();
    }
    MPI_Finalize();
    return 0;
}