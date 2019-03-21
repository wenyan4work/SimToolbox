/**
 * @file TpetraUtil.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief A collection of Tpetra definitions and utility functions
 * @version 1.0
 * @date 2016-12-20
 *
 * @copyright Copyright (c) 2018
 *
 */

#ifndef TPETRAUTIL_HPP_
#define TPETRAUTIL_HPP_

// Teuchos utility
#include <Teuchos_ArrayViewDecl.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_oblackholestream.hpp>

// Tpetra container
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

// Belos solver
#include <BelosOperatorTraits.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

// Preconditioner
#include <Ifpack2_Factory.hpp>

// no need to specify node type for new version of Tpetra. It defaults to
// Kokkos::default, which is openmp
// typedef Tpetra::Details::DefaultTypes::node_type TNODE;

using TCOMM = Teuchos::Comm<int>;                  ///< default Teuchos::Comm type
using TMAP = Tpetra::Map<int, int>;                ///< default Teuchos::Map type
using TOP = Tpetra::Operator<double, int, int>;    ///< default Tpetra::Operator type
using TCMAT = Tpetra::CrsMatrix<double, int, int>; ///< default Tpetra::CrsMatrix type
using TMV = Tpetra::MultiVector<double, int, int>; ///< default Tpetra::MultiVector type
using TV = Tpetra::Vector<double, int, int>;       ///< default to Tpetra::Vector type

/**
 * @brief inserting a specialization for Tpetra objects into Belos namespace
 *
 */
namespace Belos {
/**
 * @brief explicit full specialization of OperatorTraits for TOP and TMV.
 *
 * @tparam
 */
template <>
class OperatorTraits<::TOP::scalar_type, ::TMV, ::TOP> {
  public:
    /**
     * @brief Belos operator apply function, Y = Op X
     *
     * @param Op
     * @param X
     * @param Y
     * @param trans
     */
    static void Apply(const ::TOP &Op, const ::TMV &X, ::TMV &Y, const ETrans trans = NOTRANS) {
        Teuchos::ETransp teuchosTrans = Teuchos::NO_TRANS;
        if (trans == NOTRANS) {
            teuchosTrans = Teuchos::NO_TRANS;
        } else if (trans == TRANS) {
            teuchosTrans = Teuchos::TRANS;
        } else if (trans == CONJTRANS) {
            teuchosTrans = Teuchos::CONJ_TRANS;
        } else {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                                       "Belos::OperatorTraits::Apply: Invalid "
                                       "'trans' value "
                                           << trans << ".  Valid values are NOTRANS=" << NOTRANS << ", TRANS=" << TRANS
                                           << ", and CONJTRANS=" << CONJTRANS << ".");
        }
        Op.apply(X, Y, teuchosTrans);
    }

    /**
     * @brief return if the operator has transpose apply
     *
     * @param Op
     * @return true
     * @return false
     */
    static bool HasApplyTranspose(const ::TOP &Op) { return Op.hasTransposeApply(); }
};
} // namespace Belos

/**
 * @brief write TCMAT A to a file in MatrixMarket format
 *
 * @param A
 * @param filename
 */
void dumpTCMAT(const Teuchos::RCP<const TCMAT> &A, std::string filename);

/**
 * @brief write TMV A to a file in MatrixMarket format
 *
 * @param A
 * @param filename
 */
void dumpTMV(const Teuchos::RCP<const TMV> &A, std::string filename);

/**
 * @brief write TOP A to a file in MatrixMarket format
 *
 * TOP will be applied dim(A) times to compute the entries
 * @param A
 * @param filename
 */
void dumpTV(const Teuchos::RCP<const TV> &A, std::string filename);

/**
 * @brief the default TCOMM corresponding to MPI_COMM_WORLD
 *
 * @return Teuchos::RCP<const TCOMM>
 */
Teuchos::RCP<const TCOMM> getMPIWORLDTCOMM();

/**
 * @brief return a fully copied TMAP with a given global size
 *
 * @param globalSize
 * @param commRcp
 * @return Teuchos::RCP<TMAP>
 */
Teuchos::RCP<TMAP> getFullCopyTMAPFromGlobalSize(const int &globalSize, Teuchos::RCP<const TCOMM> &commRcp);

/**
 * @brief return a contiguous TMAP from local Size
 *
 * @param localSize
 * @param commRcp
 * @return Teuchos::RCP<TMAP>
 */
Teuchos::RCP<TMAP> getTMAPFromLocalSize(const int &localSize, Teuchos::RCP<const TCOMM> &commRcp);

/**
 * @brief contiguous TV from a local vector
 *
 * @param in
 * @param commRcp
 * @return Teuchos::RCP<TV> the local part of this TV will contain the same entries as given in the input vector
 */
Teuchos::RCP<TV> getTVFromVector(const std::vector<double> &in, Teuchos::RCP<const TCOMM> &commRcp);

/**
 * @brief contiguous TMV init from a vector of vector. localsize= min_k in[k].size()
 *
 * @param in
 * @param commRcp
 * @return Teuchos::RCP<TMV>
 */
Teuchos::RCP<TMV> getTMVFromVector(const std::vector<std::vector<double>> &in, Teuchos::RCP<const TCOMM> &commRcp);

#endif /* TPETRAUTIL_HPP_ */
