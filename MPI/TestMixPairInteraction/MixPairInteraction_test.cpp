#include "MixPairInteraction.hpp"

#include <type_traits>

struct Point {
    int gid;
    double RSearch;
    double pos[3];

    inline int getGid() const { return gid; }
    inline double getRSearch() const { return RSearch; }
    inline const PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    inline void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }
    inline void copyFromFP(const Point &other) { *this = other; }
};

struct Query {
    int gid;
    double RSearch;
    double pos[3];

    inline double getRSearch() const { return RSearch; }
    inline const PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    inline void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }
    inline void copyFromFP(const Query &other) { *this = other; }
};

struct Count {
    int nbCount = 0;
    void clear() { nbCount = 0; }
};

template <class EPT, class EPS>
class FindPair {

  public:
    void operator()(const MixEPI<EPT> *const trgPtr, const PS::S32 nTrg, const MixEPJ<EPS> *const srcPtr,
                    const PS::S32 nSrc, Count *const mixForcePtr) {
        for (int t = 0; t < nTrg; t++) {
            auto &trg = trgPtr[t];
            auto &force = mixForcePtr[t];
            force.clear();

            auto RSearchTrg = trg.epTrg.getRSearch();
            const auto &trgPos = trg.getPos();
            if (!trg.trgFlag) {
                continue;
            }

            for (int s = 0; s < nSrc; s++) {
                auto &src = srcPtr[s];
                if (!src.srcFlag) {
                    continue;
                }
                // actual interaction
                const auto &srcPos = src.getPos();
                auto RSearchSrc = src.epSrc.getRSearch();
                const PS::F64vec3 &vecTS = srcPos - trgPos;
                double r2 = trgPos.getDistanceSQ(srcPos);
                if (r2 < pow(RSearchSrc, 2) || r2 < pow(RSearchTrg, 2)) {
                    // printf("trg %d,%lf,%lf,%lf\t;", trg.epTrg.getGid(), trgPos.x, trgPos.y, trgPos.z);
                    // printf("src %d,%lf,%lf,%lf\n", src.epSrc.getGid(), srcPos.x, srcPos.y, srcPos.z);
                    force.nbCount++;
                }
            }
        }
    }
};

template <class Particle>
void initFromCSV(PS::ParticleSystem<Particle> &sys, const std::string &csvName) {
    sys.initialize();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank > 0) {
        sys.setNumberOfParticleLocal(0);
    } else {
        std::vector<Particle> data;
        std::ifstream myfile(csvName);
        std::string line;
        std::getline(myfile, line); // read a header line

        while (std::getline(myfile, line)) {
            std::istringstream liness(line);
            double px, py, pz;
            double radius;
            liness >> px >> py >> pz >> radius;
            Particle newPar;
            newPar.gid = 0;
            newPar.RSearch = radius;
            newPar.pos[0] = px;
            newPar.pos[1] = py;
            newPar.pos[2] = pz;
            data.push_back(newPar);
        }

        const int nPar = data.size();
        sys.setNumberOfParticleLocal(nPar);
        for (int i = 0; i < nPar; i++) {
            sys[i] = data[i];
            sys[i].gid = i;
        }
    }

    return;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    {

        PS::Initialize(argc, argv);

        PS::ParticleSystem<Point> sysPoint;
        PS::ParticleSystem<Query> sysQuery;
        initFromCSV(sysPoint, "Pts.txt");
        initFromCSV(sysQuery, "Query.txt");
        PS::Comm::barrier();
        printf("initialized\n");

        PS::DomainInfo dinfo;

        dinfo.initialize();
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XYZ);
        dinfo.setPosRootDomain(PS::F64vec3(0, 0, 0), PS::F64vec3(10, 10, 10)); // rootdomain must be specified after PBC

        dinfo.decomposeDomainAll(sysPoint);
        printf("decomposed\n");
        sysPoint.exchangeParticle(dinfo);
        sysPoint.exchangeParticle(dinfo);
        printf("exchanged\n");

        MixPairInteraction<Point, Query, Point, Query, Count> mixSystem;
        mixSystem.initialize();

        mixSystem.updateSystem(sysPoint, sysQuery, dinfo);
        printf("mixSystemUpdated\n");
        mixSystem.updateTree();
        printf("mixTreeUpdated\n");

        using Interactor = FindPair<Point, Query>;
        Interactor countFtr;
        mixSystem.computeForce<Interactor>(countFtr, dinfo);
        printf("forceComputed\n");

        const auto &forceResult = mixSystem.getForceResult();

        for (auto &f : forceResult) {
            std::cout << f.nbCount << std::endl;
        }

        PS::Finalize();
    }
    MPI_Finalize();
    return 0;
}