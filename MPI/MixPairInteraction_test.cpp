#include "CommMPI.hpp"
#include "MixPairInteraction.hpp"

#include <algorithm>
#include <deque>
#include <memory>

struct Point {
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

template <class Particle>
void initFromFile(PS::ParticleSystem<Particle> &sys, const std::string &csvName) {
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

    MPI_Barrier(MPI_COMM_WORLD);

    return;
}

struct Count {
    int nbCount = 0;
    void clear() { nbCount = 0; }
};

struct Pair {
    int a, b;
    double r;
    double xa, ya, za, xb, yb, zb;
    Pair() = default;
    Pair(int a_, int b_, double r_, double xa_, double ya_, double za_, double xb_, double yb_, double zb_) {
        a = a_;
        b = b_;
        r = r_;
        xa = xa_;
        ya = ya_;
        za = za_;
        xb = xb_;
        yb = yb_;
        zb = zb_;
    }
};

using PairList = std::vector<Pair>;

template <class EPT, class EPS>
class FindPair {

  public:
    std::shared_ptr<PairList> pairsPtr;

    void operator()(const MixEPI<EPT> *const trgPtr, const PS::S32 nTrg, //
                    const MixEPJ<EPS> *const srcPtr, const PS::S32 nSrc, Count *const mixForcePtr) {
        for (int t = 0; t < nTrg; t++) {
            auto &trg = trgPtr[t];
            auto &force = mixForcePtr[t];
            force.clear();

            const double RSearchTrg = trg.getRSearch();
            const auto &trgPos = trg.getPos();
            if (!trg.trgFlag) {
                continue;
            }

            for (int s = 0; s < nSrc; s++) {
                auto &src = srcPtr[s];
                if (!src.srcFlag) {
                    continue;
                }
                const auto &srcPos = src.getPos();
                const double RSearchSrc = src.getRSearch();
                double r2 = trgPos.getDistanceSQ(srcPos);
                if (r2 < pow(RSearchSrc, 2) || r2 < pow(RSearchTrg, 2)) {
                    force.nbCount++;
#pragma omp critical
                    {
                        pairsPtr->emplace_back(trg.epTrg.gid, src.epSrc.gid, sqrt(r2), trgPos.x, trgPos.y, trgPos.z,
                                               srcPos.x, srcPos.y, srcPos.z);
                    }
                }
            }
        }
    }
};

void printRank0(const std::string &message, int rank) {
#ifdef DEBUG
    if (rank == 0) {
        std::cout << message << std::endl;
    }
#endif
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int rank, nProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    {
        PS::Initialize(argc, argv);

        PS::ParticleSystem<Point> sysPoint;
        PS::ParticleSystem<Query> sysQuery;
        initFromFile(sysPoint, "Pts.txt");
        initFromFile(sysQuery, "Query.txt");
        PS::Comm::barrier();
        printRank0("initialized", rank);

        PS::DomainInfo dinfo;

        dinfo.initialize();
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XYZ);
        dinfo.setPosRootDomain(PS::F64vec3(0, 0, 0), PS::F64vec3(10, 10, 10)); // rootdomain must be specified after PBC

        dinfo.decomposeDomainAll(sysPoint);
        printRank0("decomposed", rank);
        sysPoint.exchangeParticle(dinfo);
        sysQuery.exchangeParticle(dinfo);
        printRank0("exchanged", rank);

        MixPairInteraction<Query, Point, Query, Point, Count> mixSystem;
        mixSystem.initialize();
        mixSystem.updateSystem(sysQuery, sysPoint, dinfo);
        printRank0("mixSystemUpdated", rank);

        mixSystem.updateTree();
        printRank0("mixTreeUpdated", rank);

        using Interactor = FindPair<Query, Point>;
        Interactor countFtr;
        countFtr.pairsPtr = std::make_shared<PairList>();

        mixSystem.computeForce<Interactor>(countFtr, dinfo);
        printRank0("forceComputed", rank);

        // gather all pairs to rank 0 and print
        const auto &pairs = *(countFtr.pairsPtr);
        // printf("%d,%d\n", rank, pairs.size());

        const int nPairsLocal = pairs.size();
        std::vector<int> nPairsRank(nProcs);
        MPI_Gather(&nPairsLocal, 1, MPI_INT, nPairsRank.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displ(nProcs + 1);
        std::fill(displ.begin(), displ.end(), 0);
        for (int i = 1; i <= nProcs; i++) {
            displ[i] = displ[i - 1] + nPairsRank[i - 1];
        }
        const int nPairsGlobal = displ.back();
        std::vector<Pair> pairsGlobal(nPairsGlobal);
        auto Data = createMPIStructType<Pair>();
        MPI_Gatherv(pairs.data(), pairs.size(), Data,                          //
                    pairsGlobal.data(), nPairsRank.data(), displ.data(), Data, //
                    0, MPI_COMM_WORLD);

        std::sort(pairsGlobal.begin(), pairsGlobal.end(),
                  [](const Pair &p1, const Pair &p2) { return p1.a == p2.a ? p1.b < p2.b : p1.a < p2.a; });

        if (rank == 0) {
            for (auto &pair : pairsGlobal) {
                std::cout << pair.a << " " << pair.b << " " << pair.r << " " << pair.xa << " " << pair.ya << " "
                          << pair.za << " " << pair.xb << " " << pair.yb << " " << pair.zb << std::endl;
            }
        }

        PS::Finalize();
    }
    MPI_Finalize();
    return 0;
}