/**
 * @file ZDD.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief A wrapper for Zoltan Data Directory
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */

#ifndef ZDD_HPP_
#define ZDD_HPP_

#include <type_traits>
#include <vector>

#include <zoltan_dd_cpp.h>
#include <zoltan_types.h>

#include <mpi.h>

/**
 * @brief A wrapper for Zoltan Data directory
 *
 * @tparam DATA_TYPE the data type associated with each gid
 */
template <class DATA_TYPE>
class ZDD {
    int rankSize;      ///< mpi rank size
    int myRank;        ///< local mpi rank id
    Zoltan_DD findZDD; ///< Zoltan_DD object

  public:
    typedef ZOLTAN_ID_TYPE ID_TYPE;   ///< ID type of each gid
    std::vector<ID_TYPE> findID;      ///< a list of ID to find
    std::vector<DATA_TYPE> findData;  ///< data associated with ID to find
    std::vector<ID_TYPE> localID;     ///< ID located on local mpi rank
    std::vector<DATA_TYPE> localData; ///< data associated with ID on local mpi rank

    ZDD(const ZDD &) = delete;
    ZDD &operator=(const ZDD &a) = delete;

    /**
     * @brief Construct a new ZDD object with an estimate of buffer list size
     *
     * @param nEst estimate of buffer list size
     */
    ZDD(int nEst) {
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &rankSize);

        int error = 0;
#ifdef ZDDDEBUG
        // debug verbose level 9
        error = this->findZDD.Create(MPI_COMM_WORLD, 1, 0, 0, 0, 9);
        findZDD.Print();
        findZDD.Stats();
#else
        // normal mode, debug verbose level 0
        error = this->findZDD.Create(MPI_COMM_WORLD, 1, 0, 0, 0, 0);
#endif
        if (error != ZOLTAN_OK) {
            printf("ZDD Create error %d\n", error);
            exit(1);
        }

        static_assert(std::is_trivially_copyable<ID_TYPE>::value, "");
        static_assert(std::is_trivially_copyable<DATA_TYPE>::value, "");
        static_assert(std::is_default_constructible<DATA_TYPE>::value, "");
    }

    /**
     * @brief clean all ID and Data lists
     *
     */
    void clearAll() {
        findID.clear();
        findData.clear();
        localID.clear();
        localData.clear();
#ifdef ZDDDEBUG
        findZDD.Print();
        findZDD.Stats();
#endif
    }

    /**
     * @brief Destroy the ZDD object
     *
     */
    ~ZDD() {
        MPI_Barrier(MPI_COMM_WORLD);
#ifdef ZDDDEBUG
        findZDD.Print();
        findZDD.Stats();
        std::cout << "ZDD destructed" << std::endl;
#endif
    }

    /**
     * @brief build the Zoltan data directory
     *
     * must have an effective nblocal data list
     * @return int the error code returned by Zoltan_DD.Update()
     */
    int buildIndex() {

#ifdef ZDDDEBUG
        std::cout << localID.size() << std::endl;
        for (auto &id : localID) {
            std::cout << "id " << id << std::endl;
        }
        printf("ZDD before Update\n");
        findZDD.Print();
        findZDD.Stats();
#endif

        ZOLTAN_ID_PTR idPtr = localID.data();
        DATA_TYPE *dataPtr = localData.data();
        int error;
        error = findZDD.Update(idPtr, NULL, NULL, NULL, localID.size());
        if (error != ZOLTAN_OK) {
            printf("ZDD Update error %d\n", error);
            exit(1);
        }

#ifdef ZDDDEBUG
        printf("ZDD after update\n");
        findZDD.Print();
        findZDD.Stats();
#endif

        return error;
    }

    /**
     * @brief locate and put data to findData as requested by findID
     *
     * @return int the error code returned by Zoltan_DD.Find()
     */
    int find() {
        if (findData.size() < findID.size()) {
            findData.resize(findID.size()); // make sure size match
        }
        auto *idPtr = findID.data();
        auto *dataPtr = findData.data();
        // ZOLTAN_ID_PTR gid, ZOLTAN_ID_PTR lid, char *data, int *part, const int &
        // count, int *owner
        int status = findZDD.Find(idPtr, NULL, (char *)dataPtr, NULL, findID.size(), NULL);
        return status;
    }
};

#endif /* ZDD_HPP_ */
