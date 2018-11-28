#include "MPI/MixPairInteraction.hpp"
#include "Util/TRngPool.hpp"
#include <type_traits>

struct Tubule {
    int gid;
    double RSearch = 1;
    double pos[3];

    // inline int getGid() const { return gid; }
    inline double getRSearch() const { return RSearch; }
    inline const PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    inline void copyFromFP(const Tubule &other) { *this = other; }
};

struct Motor {
    int gid;
    double RSearch;
    double pos[3];

    // inline int getGid() const { return gid; }
    inline double getRSearch() const { return RSearch; }
    inline const PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    inline void copyFromFP(const Motor &other) { *this = other; }
};

struct ForceTest {
    int nbCount = 0;
    void clear() { nbCount = 0; }
};

template <class EPT, class EPS>
class CountMixNeighbor {

  public:
    void operator()(const MixEPI<EPT> *const trgPtr, const PS::S32 nTrg, const MixEPJ<EPS> *const srcPtr,
                    const PS::S32 nSrc, ForceTest *const mixForcePtr) {
        for (int t = 0; t < nTrg; t++) {
            auto &trg = trgPtr[t];
            auto &force = mixForcePtr[t];
            force.clear();
            if (!trg.trgFlag) {
                continue;
            }
            auto RSearchTrg = trg.getRSearch();
            const auto &trgPos = trg.getPos();
            for (int s = 0; s < nSrc; s++) {
                auto &src = srcPtr[s];
                if (!src.srcFlag) {
                    continue;
                }
                // actual interaction
                const auto &srcPos = src.getPos();
                auto RSearchSrc = src.getRSearch();
                const PS::F64vec3 &vecTS = srcPos - trgPos;
                double r2 = trgPos.getDistanceSQ(srcPos);
                if (r2 < pow(RSearchSrc + RSearchTrg, 2)) {
                    force.nbCount++;
                }
            }
        }
    }
};

void initTubule(PS::ParticleSystem<Tubule> &tubule) {
    const int nT = 10;
    tubule.initialize();
    tubule.setNumberOfParticleLocal(nT);

    TRngPool rngPool(nT);

#pragma omp parallel for
    for (int i = 0; i < nT; i++) {
        tubule[i].gid = i;

        tubule[i].RSearch = 0.1;
        tubule[i].pos[0] = rngPool.getU01();
        tubule[i].pos[1] = rngPool.getU01();
        tubule[i].pos[2] = rngPool.getU01();
    }
}

void initMotor(PS::ParticleSystem<Motor> &motor) {
    const int nM = 10000;
    motor.initialize();
    motor.setNumberOfParticleLocal(nM);

    TRngPool rngPool(nM);

#pragma omp parallel for
    for (int i = 0; i < nM; i++) {
        motor[i].gid = i;

        motor[i].RSearch = 0.1;
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
    dinfo.setPosRootDomain(PS::F64vec3(0, 0, 0), PS::F64vec3(1, 1, 1)); // rootdomain must be specified after PBC

    dinfo.decomposeDomainAll(sysTubule);
    sysTubule.exchangeParticle(dinfo);
    sysMotor.exchangeParticle(dinfo);

    MixPairInteraction<Tubule, Motor, Tubule, Motor, ForceTest> mixSystem(sysTubule, sysMotor, dinfo);

    mixSystem.updateSystem();
    mixSystem.updateTree();

    CountMixNeighbor<Tubule, Motor> countMixNbFtr;
    mixSystem.computeForce<CountMixNeighbor<Tubule, Motor>>(countMixNbFtr);

    const auto &forceResult = mixSystem.getForceResult();

    for (auto &f : forceResult) {
        std::cout << f.nbCount << std::endl;
    }

    PS::Finalize();
    MPI_Finalize();
    return 0;
}