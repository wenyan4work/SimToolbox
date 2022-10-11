#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

#include <limits>

void dumpTOP(const Teuchos::RCP<const TOP> &A, const std::string &filename) {
    const std::string filename_mod = filename + std::string("_TOP.mtx");
    spdlog::info("dumping " + filename_mod);

    Tpetra::MatrixMarket::Writer<TCMAT> topDumper;
    topDumper.writeOperator(filename_mod, *A);
}

void dumpTCMAT(const Teuchos::RCP<const TCMAT> &A, const std::string &filename) {
    const std::string filename_mod = filename + std::string("_TCMAT.mtx");
    spdlog::info("dumping " + filename_mod);

    Tpetra::MatrixMarket::Writer<TCMAT> matDumper;
    matDumper.writeSparseFile(filename_mod, A, filename_mod, filename_mod, true);
}

void dumpTV(const Teuchos::RCP<const TV> &A, std::string filename) {
    filename = filename + std::string("_TV.mtx");
    spdlog::info("dumping " + filename);

    const auto &fromMap = A->getMap();
    const auto &toMap =
        Teuchos::rcp(new TMAP(fromMap->getGlobalNumElements(), 0, fromMap->getComm(), Tpetra::GloballyDistributed));
    Tpetra::Import<TV::local_ordinal_type, TV::global_ordinal_type, TV::node_type> importer(fromMap, toMap);
    Teuchos::RCP<TV> B = Teuchos::rcp(new TV(toMap, true));
    B->doImport(*A, importer, Tpetra::CombineMode::REPLACE);

    Tpetra::MatrixMarket::Writer<TV> matDumper;
    matDumper.writeDenseFile(filename, B, filename, filename);
}

void dumpTMAP(const Teuchos::RCP<const TMAP> &map, std::string filename) {
    filename = filename + std::string("_TMAP.mtx");
    spdlog::info("dumping " + filename);

    Tpetra::MatrixMarket::Writer<TV> writer;
    writer.writeMapFile(filename, *map);
}

Teuchos::RCP<const TCOMM> getMPIWORLDTCOMM() { return Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD)); }

Teuchos::RCP<TMAP> getTMAPFromLocalSize(const size_t localSize, const Teuchos::RCP<const TCOMM> &commRcp) {
    return Teuchos::rcp(new TMAP(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), localSize, 0LL, commRcp));
}

Teuchos::RCP<TMAP> getTMAPFromGlobalIndexOnLocal(const std::vector<GO> &gidOnLocal, const GO globalSize,
                                                 const Teuchos::RCP<const TCOMM> &commRcp) {
    return Teuchos::rcp(new TMAP(globalSize, gidOnLocal.data(), gidOnLocal.size(), 0, commRcp));
}

Teuchos::RCP<TMAP> getTMAPFromTwoBlockTMAP(const Teuchos::RCP<const TMAP> &map1, const Teuchos::RCP<const TMAP> &map2) {
    // assumption: both map1 and map2 are contiguous and start from 0-indexbase
    TEUCHOS_TEST_FOR_EXCEPTION(!map1->isContiguous(), std::invalid_argument, "map1 must be contiguous");
    TEUCHOS_TEST_FOR_EXCEPTION(!map2->isContiguous(), std::invalid_argument, "map2 must be contiguous");

    auto gid1 = map1->getMyGlobalIndices();
    auto gid2 = map2->getMyGlobalIndices();
    const auto localSize1 = map1->getNodeNumElements();
    const auto localSize2 = map2->getNodeNumElements();
    const auto globalSize1 = map1->getGlobalNumElements();
    const auto globalSize2 = map2->getGlobalNumElements();

    std::vector<GO> gidOnLocal(localSize1 + localSize2, 0);
    for (size_t i = 0; i < localSize1; i++) {
        gidOnLocal[i] = gid1[i];
    }
    for (size_t i = 0; i < localSize2; i++) {
        gidOnLocal[i + localSize1] = gid2[i] + globalSize1;
    }

    auto commRcp = map1->getComm();
    auto map = Teuchos::rcp(new TMAP(globalSize1 + globalSize2, gidOnLocal.data(), gidOnLocal.size(), 0, commRcp));
    return map;
}

Teuchos::RCP<TV> getTVFromTwoBlockTV(const Teuchos::RCP<const TV> &vec1, const Teuchos::RCP<const TV> &vec2) {
    const auto &map1 = vec1->getMap();
    const auto &map2 = vec2->getMap();
    Teuchos::RCP<TMAP> map = getTMAPFromTwoBlockTMAP(map1, map2);
    Teuchos::RCP<TV> vec = Teuchos::rcp(new TV(map, true));
    Teuchos::RCP<TV> vecSub1 = vec->offsetViewNonConst(map1, 0);
    vecSub1->update(1.0, *vec1, 0.0);
    Teuchos::RCP<TV> vecSub2 = vec->offsetViewNonConst(map2, map1->getNodeNumElements());
    vecSub2->update(1.0, *vec2, 0.0);
    return vec;
}

Teuchos::RCP<TV> getTVFromVector(const std::vector<double> &in, const Teuchos::RCP<const TCOMM> &commRcp) {

    const int localSize = in.size();

    Teuchos::RCP<TMAP> contigMapRcp = getTMAPFromLocalSize(localSize, commRcp);

    Teuchos::RCP<TV> out = Teuchos::rcp(new TV(contigMapRcp, false));

    auto out_2d = out->getLocalView<Kokkos::HostSpace>();
    assert(out_2d.extent(0) == localSize);

    out->modify<Kokkos::HostSpace>();
    for (size_t c = 0; c < out_2d.extent(1); c++) {
#pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = 0; i < out_2d.extent(0); i++) {
            out_2d(i, c) = in[i];
        }
    }

    return out;
}