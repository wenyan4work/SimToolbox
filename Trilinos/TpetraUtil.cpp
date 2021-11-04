#include "TpetraUtil.hpp"
#include "Util/Logger.hpp"

#include <limits>

void dumpTCMAT(const Teuchos::RCP<const TCMAT> &A, std::string filename) {
  filename = filename + std::string("_TCMAT.mtx");
  spdlog::info("dumping " + filename);

  Tpetra::MatrixMarket::Writer<TCMAT> matDumper;
  matDumper.writeSparseFile(filename, A, filename, filename, true);
}

void dumpTV(const Teuchos::RCP<const TV> &A, std::string filename) {
  filename = filename + std::string("_TV.mtx");
  spdlog::info("dumping " + filename);

  const auto &fromMap = A->getMap();
  const auto &toMap =
      Teuchos::rcp(new TMAP(fromMap->getGlobalNumElements(), 0LL,
                            fromMap->getComm(), Tpetra::GloballyDistributed));
  Tpetra::Import<TV::local_ordinal_type, TV::global_ordinal_type, TV::node_type>
      importer(fromMap, toMap);
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

Teuchos::RCP<TMAP> getTMAPFromLocalSize(const size_t localSize,
                                        Teuchos::RCP<const TCOMM> &commRcp) {
  return Teuchos::rcp(
      new TMAP(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
               localSize, 0LL, commRcp));
}

Teuchos::RCP<TMAP>
getTMAPFromGlobalIndexOnLocal(const std::vector<TGO> &gidOnLocal,
                              const TGO globalSize,
                              Teuchos::RCP<const TCOMM> &commRcp) {
  return Teuchos::rcp(
      new TMAP(globalSize, gidOnLocal.data(), gidOnLocal.size(), 0LL, commRcp));
}

Teuchos::RCP<TMAP>
getTMAPFromTwoBlockTMAP(const Teuchos::RCP<const TMAP> &map1,
                        const Teuchos::RCP<const TMAP> &map2) {
  // assumption: both map1 and map2 are contiguous and start from 0-indexbase
  TEUCHOS_TEST_FOR_EXCEPTION(!map1->isContiguous(), std::invalid_argument,
                             "map1 must be contiguous");
  TEUCHOS_TEST_FOR_EXCEPTION(!map2->isContiguous(), std::invalid_argument,
                             "map2 must be contiguous");

  auto gid1 = map1->getMyGlobalIndices();
  auto gid2 = map2->getMyGlobalIndices();
  const auto localSize1 = map1->getNodeNumElements();
  const auto localSize2 = map2->getNodeNumElements();
  const auto globalSize1 = map1->getGlobalNumElements();
  const auto globalSize2 = map2->getGlobalNumElements();

  std::vector<TGO> gidOnLocal(localSize1 + localSize2, 0);
#pragma omp parallel for
  for (size_t i = 0; i < localSize1; i++) {
    gidOnLocal[i] = gid1[i];
  }
#pragma omp parallel for
  for (size_t i = 0; i < localSize2; i++) {
    // gidOnLocal[i + localSize1] = gid2[i] + map1->getGlobalNumElements();
    gidOnLocal[i + localSize1] = gid2[i] + globalSize1;
  }

  auto commRcp = map1->getComm();
  auto map = Teuchos::rcp(new TMAP(globalSize1 + globalSize2, gidOnLocal.data(),
                                   gidOnLocal.size(), 0, commRcp));
  return map;
}

Teuchos::RCP<TV> getTVFromTwoBlockTV(const Teuchos::RCP<const TV> &vec1,
                                     const Teuchos::RCP<const TV> &vec2) {
  const auto &map1 = vec1->getMap();
  const auto &map2 = vec2->getMap();
  Teuchos::RCP<TMAP> map = getTMAPFromTwoBlockTMAP(map1, map2);
  Teuchos::RCP<TV> vec = Teuchos::rcp(new TV(map, true));
  Teuchos::RCP<TV> vecSub1 = vec->offsetViewNonConst(map1, 0);
  vecSub1->update(1.0, *vec1, 0.0);
  Teuchos::RCP<TV> vecSub2 =
      vec->offsetViewNonConst(map2, map1->getNodeNumElements());
  vecSub2->update(1.0, *vec2, 0.0);
  return vec;
}

Teuchos::RCP<TV> getTVFromVector(const std::vector<double> &in,
                                 Teuchos::RCP<const TCOMM> &commRcp) {

  const int localSize = in.size();

  Teuchos::RCP<TMAP> contigMapRcp = getTMAPFromLocalSize(localSize, commRcp);

  Teuchos::RCP<TV> out =
      Teuchos::rcp(new TV(contigMapRcp, Teuchos::arrayViewFromVector(in)));

  return out;
}

Teuchos::RCP<TOP>
createIfpack2Preconditioner(const Teuchos::RCP<const TCMAT> &A,
                            const Teuchos::ParameterList &plist) {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Time;
  using Teuchos::TimeMonitor;

  using PrecType =
      Ifpack2::Preconditioner<TCMAT::scalar_type, TCMAT::local_ordinal_type,
                              TCMAT::global_ordinal_type>;

  // Create timers to show how long it takes for Ifpack2 to do various
  // operations.
  RCP<Time> initTimer =
      TimeMonitor::getNewCounter("Ifpack2::Preconditioner::initialize");
  RCP<Time> computeTimer =
      TimeMonitor::getNewCounter("Ifpack2::Preconditioner::compute");

  // Create the preconditioner and set parameters.
  // This doesn't actually _compute_ the preconditioner.
  // It just sets up the specific type of preconditioner and
  // its associated parameters (which depend on the type).
  RCP<PrecType> prec;
  Ifpack2::Factory factory;

  // Set up the preconditioner of the given type.
  prec = factory.create(plist.name(), A);
  prec->setParameters(plist);

  {
    TimeMonitor mon(*initTimer);
    prec->initialize();
  }

  {
    // THIS ACTUALLY COMPUTES THE PRECONDITIONER (e.g., does the incomplete
    // factorization).
    TimeMonitor mon(*computeTimer);
    prec->compute();
  }
  return prec;
}

Teuchos::RCP<TOP> createILUTPreconditioner(const Teuchos::RCP<const TCMAT> &A,
                                           double tol, double fill) {

  spdlog::info("Preconditioner ILUT setup");

  const auto &commRcp = A->getComm();

  Teuchos::ParameterList plist;

  plist.setName("ILUT");
  plist.set("fact: ilut level-of-fill", fill);
  plist.set("fact: drop tolerance", tol);

  // this stablizes the preconditioner but takes more iterations
  plist.set("fact: absolute threshold", 0.0001);

  return createIfpack2Preconditioner(A, plist);
}

Teuchos::RCP<TOP> createPlnPreconditioner(const Teuchos::RCP<const TCMAT> &A,
                                          int sweep, double damping) {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::Time;
  using Teuchos::TimeMonitor;

  spdlog::info("Preconditioner Pln setup");

  auto commRcp = A->getComm();

  Teuchos::ParameterList plist;
  plist.setName("RELAXATION");

  plist.set("relaxation: type", "Jacobi");
  plist.set("relaxation: sweeps", sweep);
  plist.set("relaxation: damping factor", damping);
  plist.set("relaxation: use l1", true);
  plist.set("relaxation: fix tiny diagonal entries", true);
  plist.set("relaxation: min diagonal value", 1e-5);

  // must be true. otherwise may give random or NAN result
  plist.set("relaxation: zero starting solution", true);

  return createIfpack2Preconditioner(A, plist);
}