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
#include <Tpetra_RowMatrixTransposer_decl.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Version.hpp>

// Belos solver
#include <BelosOperatorTraits.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>

// NOX solver
#include <NOX.H>
#include <NOX_Thyra.H>

/////////////////////////////
// Stuff we might not need //
/////////////////////////////
#include <Teuchos_ConfigDefs.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_FancyOStream.hpp>

#include <BelosTypes.hpp>
#include <Stratimikos_DefaultLinearSolverBuilder.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>

#include <NOX_Thyra_MatrixFreeJacobianOperator.hpp>
#include <NOX_MatrixFree_ModelEvaluatorDecorator.hpp>

// for solution IO
#include "Thyra_TpetraVector.hpp"

// Preconditioner
#include <Ifpack2_Factory.hpp>

// no need to specify node type for new version of Tpetra. It defaults to
// Kokkos::default, which is openmp
// typedef Tpetra::Details::DefaultTypes::node_type TNODE;
using Scalar = double;
using LO = int;
using GO = int;
using Node = Tpetra::Vector<>::node_type;

using TCOMM = Teuchos::Comm<GO>;                       ///< default Teuchos::Comm type
using TMAP = Tpetra::Map<LO, GO, Node>;                ///< default Teuchos::Map type
using TOP = Tpetra::Operator<Scalar, LO, GO, Node>;    ///< default Tpetra::Operator type
using TCMAT = Tpetra::CrsMatrix<Scalar, LO, GO, Node>; ///< default Tpetra::CrsMatrix type
using TMV = Tpetra::MultiVector<Scalar, LO, GO, Node>; ///< default Tpetra::MultiVector type
using TV = Tpetra::Vector<Scalar, LO, GO, Node>;       ///< default to Tpetra::Vector type

using thyra_vec_space = Thyra::VectorSpaceBase<Scalar>;
using thyra_vec = Thyra::VectorBase<Scalar>;
using thyra_op = Thyra::LinearOpBase<Scalar>;
using thyra_prec = Thyra::PreconditionerBase<Scalar>;

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
 * @brief the default TCOMM corresponding to MPI_COMM_WORLD
 *
 * @return Teuchos::RCP<const TCOMM>
 */
Teuchos::RCP<const TCOMM> getMPIWORLDTCOMM();

/**
 * @brief return a contiguous TMAP from local Size
 *
 * @param localSize
 * @param commRcp
 * @return Teuchos::RCP<TMAP>
 */
Teuchos::RCP<TMAP> getTMAPFromLocalSize(const int &localSize, const Teuchos::RCP<const TCOMM> &commRcp);

/**
 * @brief get a TMAP from arbitrary global index on local
 *
 * @param gidOnLocal
 * @param globalSize total global size
 * @param commRcp
 * @return Teuchos::RCP<TMAP>
 */
Teuchos::RCP<TMAP> getTMAPFromGlobalIndexOnLocal(const std::vector<int> &gidOnLocal, const int globalSize,
                                                 const Teuchos::RCP<const TCOMM> &commRcp);

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
Teuchos::RCP<TMAP> getTMAPFromTwoBlockTMAP(const Teuchos::RCP<const TMAP> &map1, const Teuchos::RCP<const TMAP> &map2);

/**
 * @brief create a vector for two blocks X=[X1;X2]
 * 
 * @param vec1 
 * @param vec2 
 * @return Teuchos::RCP<TV> 
 */
Teuchos::RCP<TV> getTVFromTwoBlockTV(const Teuchos::RCP<const TV> &vec1, const Teuchos::RCP<const TV> &vec2);

/**
 * @brief contiguous TV from a local vector
 *
 * @param in
 * @param commRcp
 * @return Teuchos::RCP<TV> the local part of this TV will contain the same entries as given in the input vector
 */
Teuchos::RCP<TV> getTVFromVector(const std::vector<double> &in, const Teuchos::RCP<const TCOMM> &commRcp);

#endif /* TPETRAUTIL_HPP_ */
