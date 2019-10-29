#include "ConstraintOperator.hpp"

ConstraintOperator::ConstraintOperator(Teuchos::RCP<TOP> &mobOp_, Teuchos::RCP<TCMAT> &uniDcMatTrans_,
                                       Teuchos::RCP<TCMAT> &biDbMatTrans_, std::vector<double> &invKappaDiagMat_)
    : commRcp(mobOp_->getDomainMap()->getComm()), mobOpRcp(mobOp_), uniDcMatTransRcp(uniDcMatTrans_),
      biDbMatTransRcp(biDbMatTrans_), invKappaDiagMat(invKappaDiagMat_) {

    // explicit transpose
    Tpetra::RowMatrixTransposer<TCMAT::scalar_type, TCMAT::local_ordinal_type, TCMAT::global_ordinal_type> transposerDc(
        uniDcMatTransRcp);
    uniDcMatRcp = transposerDc.createTranspose();
    Tpetra::RowMatrixTransposer<TCMAT::scalar_type, TCMAT::local_ordinal_type, TCMAT::global_ordinal_type> transposerDb(
        biDbMatTransRcp);
    biDbMatRcp = transposerDb.createTranspose();

    // block maps
    mobMapRcp = mobOpRcp->getDomainMap(); // symmetric & domainmap=rangemap

    // setup global map
    // both Dc and Db are globally & contiguously partitioned by rows
    // The total map should include rows from both Dc and Db, where Db follows Dc
    buildBlockMaps();

    // initialize working multivectors, zero out
    gammaForceRcp = Teuchos::rcp(new TMV(mobMapRcp, 2, true)); // two columns: Dc gammac,  Db gammab
    mobVelRcp = Teuchos::rcp(new TMV(mobMapRcp, 2, true));     // two columns: M Dc gamma, M Db gammab
    deltaUniRcp = Teuchos::rcp(new TMV(uniDcMatTransRcp->getRangeMap(), 2, true)); // two columns: DcT M Dc, DcT M Db
    deltaBiRcp = Teuchos::rcp(new TMV(biDbMatTransRcp->getRangeMap(), 2, true));   // two columns: DbT M Dc, DbT M Db
}

void ConstraintOperator::apply(const TMV &X, TMV &Y, Teuchos::ETransp mode = Teuchos::NO_TRANS,
                               scalar_type alpha = Teuchos::ScalarTraits<scalar_type>::one(),
                               scalar_type beta = Teuchos::ScalarTraits<scalar_type>::zero()) const {
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                               "X and Y do not have the same numbers of vectors (columns).");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                               "X and Y do not have the same Map.\n");
    const int blockoffset = gammaUniBlockMapRcp->getNodeNumElements();

    const int numVecs = X.getNumVectors();
    for (int i = 0; i < numVecs; i++) {
        auto XcolRcp = X.getVector(i);
        auto YcolRcp = Y.getVectorNonConst(i);
        auto gammaUniBlock = X.offsetView(gammaUniBlockMapRcp, 0);
        auto gammaBiBlock = X.offsetView(gammaBiBlockMapRcp, blockoffset);
        auto deltaUniBlock = Y.offsetViewNonConst(gammaUniBlockMapRcp, 0);
        auto deltaBiBlock = Y.offsetViewNonConst(gammaBiBlockMapRcp, blockoffset);

        // step 1, Dc and Db multiply X, block by block
        auto ftCol0 = gammaForceRcp->getVectorNonConst(0);
        uniDcMatRcp->apply(*gammaUniBlock, *ftCol0); // Dc gammac
        auto ftCol1 = gammaForceRcp->getVectorNonConst(1);
        biDbMatRcp->apply(*gammaBiBlock, *ftCol1); // Db gammab

        // step 2, FT multiply mobility
        mobOpRcp->apply(*gammaForceRcp, *mobVelRcp);

        // step 3, Dc^T and Db^T multiply velocity
        uniDcMatTransRcp->apply(*mobVelRcp, *deltaUniRcp);
        biDbMatTransRcp->apply(*mobVelRcp, *deltaBiRcp);

        // step 4, add spring constant effect: K^{-1}gammab
        // deltaUniBlock[i] += K^{-1}[i]*gammab[i]
        auto deltaBiPtr = deltaBiBlock->getLocalView<Kokkos::HostSpace>();
        deltaBiBlock->modify<Kokkos::HostSpace>();
        auto gammaBiPtr = gammaBiBlock->getLocalView<Kokkos::HostSpace>();
        const int gammaBiSize = gammaBiPtr.dimension_0();
        TEUCHOS_TEST_FOR_EXCEPTION(gammaBiSize == invKappaDiagMat.size(), std::invalid_argument,
                                   "Kinv and gammaBi size error\n")
#pragma omp parallel for
        for (int k = 0; k < gammaBiSize; k++) {
            deltaBiPtr(k, 0) += invKappaDiagMat[k] * gammaBiPtr(k, 0);
        }
    }
}

Teuchos::RCP<const TMAP> ConstraintOperator::getDomainMap() const {}

Teuchos::RCP<const TMAP> ConstraintOperator::getRangeMap() const {}

void ConstraintOperator::buildBlockMaps() {
    /****
     *  For details about this sub map, see Tpetra::MultiVector::offsetView() and ::offsetViewNonConst()
     */

    auto map1 = uniDcMatRcp->getDomainMap(); // the first map, starting from 0, contiguous
    auto map2 = biDbMatRcp->getDomainMap();  // the second map, starting from 0, contiguous

    // assumption: both map1 and map2 are contiguous and start from 0-indexbase
    TEUCHOS_TEST_FOR_EXCEPTION(map1->isContiguous(), std::invalid_argument, "map1 must be contiguous");
    TEUCHOS_TEST_FOR_EXCEPTION(map2->isContiguous(), std::invalid_argument, "map2 must be contiguous");

    // the two sub block maps
    gammaUniBlockMapRcp = map1;
    gammaBiBlockMapRcp = map2;

    // assemble the global map
    const int localSize1 = map1->getNodeNumElements();
    const int globalSize1 = map1->getGlobalNumElements();
    const int localSize2 = map2->getNodeNumElements();
    const int globalSize2 = map2->getGlobalNumElements();
    const int map1GidMinOnLocal = map1->getMinGlobalIndex();
    const int map1GidMaxOnLocal = map1->getMaxGlobalIndex();
    const int map2GidMinOnLocal = map2->getMinGlobalIndex();
    const int map2GidMaxOnLocal = map2->getMaxGlobalIndex();

    std::vector<int> gammaGidAll(localSize1 + localSize2); // gids of rows for both blocks on local rank

#pragma omp parallel for
    // direct copy of map1 indices
    for (int i = 0; i < localSize1; i++) {
        gammaGidAll[i] = map1GidMaxOnLocal + i;
    }
#pragma omp parallel for
    // copy map2 indices and shift by globalSize1
    for (int i = 0; i < localSize2; i++) {
        gammaGidAll[i + localSize1] = map2GidMinOnLocal + i + globalSize1;
    }

    // the global map
    gammaMapRcp =
        Teuchos::rcp(new TMAP(globalSize1 + globalSize2, gammaGidAll.data(), localSize1 + localSize2, 0, commRcp));
}