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
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_oblackholestream.hpp>

// Tpetra container
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_RowMatrixTransposer_decl.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

// Belos solver
#include <BelosOperatorTraits.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

// Preconditioner
#include <Ifpack2_Factory.hpp>

// Kokkos
#include <Kokkos_Core.hpp>

#include <type_traits>

// Trilinos-wide next generation stack
// Deprecation: Explicit instantiation of multiple different global ordinal
// types and multiple different local ordinal types Mitigation: Use default
// global ordinal type long long and local ordinal type int, or specify to CMake
// -DTpetra_INST_INT_INT=ON to use global ordinal type int and local ordinal
// type int. Justification: Building with multiple ordinal types increases build
// times and library sizes; all applications surveyed used only one global
// ordinal type

using TCOMM = Teuchos::Comm<int>; ///< default Teuchos::Comm type

using TMAP = Tpetra::Map<>;            ///< default Teuchos::Map type
using TLO = TMAP::local_ordinal_type;  ///< default local ordinal type
using TGO = TMAP::global_ordinal_type; ///< default global ordinal type

/**
 * @brief ensure type in
 *
 */
static_assert(std::is_same<TLO, int>::value, " TLO type error \n");
static_assert(std::is_same<TGO, long long int>::value, " TGO type error \n");

/**
 * @brief use double, TLO, TGO for all Tpetra objects
 *
 */
using TOP = Tpetra::Operator<double, TLO, TGO>;
using TCMAT = Tpetra::CrsMatrix<double, TLO, TGO>;
using TMV = Tpetra::MultiVector<double, TLO, TGO>;
using TV = Tpetra::Vector<double, TLO, TGO>;

// TCMAT local matrix row offset type
using TLRO = TCMAT::local_matrix_type::row_map_type::non_const_value_type;

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
  static void Apply(const ::TOP &Op, const ::TMV &X, ::TMV &Y,
                    const ETrans trans = NOTRANS) {
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
                                     << trans << ".  Valid values are NOTRANS="
                                     << NOTRANS << ", TRANS=" << TRANS
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
  static bool HasApplyTranspose(const ::TOP &Op) {
    return Op.hasTransposeApply();
  }
};
} // namespace Belos

/**
 * @brief describe obj
 *
 * @tparam T
 * @param obj
 */
template <class T>
void describe(const T &obj) {
  Teuchos::RCP<Teuchos::FancyOStream> out =
      Teuchos::VerboseObjectBase::getDefaultOStream();
  obj.describe(*out, Teuchos::EVerbosityLevel::VERB_HIGH);
}

/**
 * @brief write TCMAT A to a file in MatrixMarket format
 *
 * @param A
 * @param filename
 */
void dumpTCMAT(const Teuchos::RCP<const TCMAT> &A, std::string filename);

/**
 * @brief write TOP A to a file in MatrixMarket format
 *
 * TOP will be applied dim(A) times to compute the entries
 * @param A
 * @param filename
 */
void dumpTV(const Teuchos::RCP<const TV> &A, std::string filename);

/**
 * @brief write TMAP map to a file in MatrixMarket format
 *
 * @param map
 * @param filename
 */
void dumpTMAP(const Teuchos::RCP<const TMAP> &map, std::string filename);

/**
 * @brief return a contiguous TMAP from local Size
 *
 * @param localSize
 * @param commRcp
 * @return Teuchos::RCP<TMAP>
 */
Teuchos::RCP<TMAP> getTMAPFromLocalSize(const size_t localSize,
                                        Teuchos::RCP<const TCOMM> &commRcp);

/**
 * @brief get a TMAP from arbitrary global index on local
 *
 * @param gidOnLocal
 * @param globalSize total global size
 * @param commRcp
 * @return Teuchos::RCP<TMAP>
 */
Teuchos::RCP<TMAP>
getTMAPFromGlobalIndexOnLocal(const std::vector<TGO> &gidOnLocal,
                              const TGO globalSize,
                              Teuchos::RCP<const TCOMM> &commRcp);

/**
 * @brief create a map for vector with two blocks X=[X1;X2],
 * where X1, X2 are both partitioned with maps map1 and map2, respectively
 *
 * assuming both map1 and map2 are contiguous and start from 0-indexbase
 * assuming vec1=[0,1,2 | 3,4,5], vec2=[a,b | c]
 * newvec = [0,1,2|3,4,5 | a,b|c] (math order)
 * newmap = [0,1,2,a,b | 3,4,5,c]
 *
 * @param map1
 * @param map2
 * @return Teuchos::RCP<TMAP>
 */
Teuchos::RCP<TMAP>
getTMAPFromTwoBlockTMAP(const Teuchos::RCP<const TMAP> &map1,
                        const Teuchos::RCP<const TMAP> &map2);

/**
 * @brief contiguous TV from a local vector
 *
 * @param in
 * @param commRcp
 * @return Teuchos::RCP<TV> the local part of this TV will contain the same
 * entries as given in the input vector
 */
Teuchos::RCP<TV> getTVFromVector(const std::vector<double> &in,
                                 Teuchos::RCP<const TCOMM> &commRcp);

/**
 * @brief create a vector for two blocks X=[X1;X2]
 *
 * @param vec1
 * @param vec2
 * @return Teuchos::RCP<TV>
 */
Teuchos::RCP<TV> getTVFromTwoBlockTV(const Teuchos::RCP<const TV> &vec1,
                                     const Teuchos::RCP<const TV> &vec2);

/**
 * @brief Create a Ifpack2 Preconditioner object
 *
 * @param A
 * @param plist
 * @return Teuchos::RCP<TOP>
 */
Teuchos::RCP<TOP>
createIfpack2Preconditioner(const Teuchos::RCP<const TCMAT> &A,
                            const Teuchos::ParameterList &plist);

/**
 * @brief create an ILUT preconditioner from Ifpack2
 *
 * @param A
 * @param tol
 * @param fill
 * @return Teuchos::RCP<TOP>
 */
Teuchos::RCP<TOP> createILUTPreconditioner(const Teuchos::RCP<const TCMAT> &A,
                                           double tol = 1e-4,
                                           double fill = 2.0);

/**
 * @brief Create a Pln Preconditioner
 *
 * @param A
 * @return Teuchos::RCP<TOP>
 */
Teuchos::RCP<TOP> createPlnPreconditioner(const Teuchos::RCP<const TCMAT> &A,
                                          int sweep = 5, double damping = 2.0);

#endif /* TPETRAUTIL_HPP_ */
