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

#include "Util/Logger.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
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
    using GID_TYPE = ZOLTAN_ID_TYPE; ///< ID type of each gid

    std::vector<GID_TYPE> gidToFind;    ///< a list of ID to find
    std::vector<DATA_TYPE> dataToFind;  ///< data associated with ID to find
    std::vector<GID_TYPE> gidOnLocal;   ///< ID located on local mpi rank
    std::vector<DATA_TYPE> dataOnLocal; ///< data associated with ID on local mpi rank

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

        gidToFind.reserve(nEst);
        dataToFind.reserve(nEst);
        gidOnLocal.reserve(nEst);
        dataOnLocal.reserve(nEst);

        int error = 0;
#ifdef ZDDDEBUG
        // debug verbose level 9
        error = this->findZDD.Create(MPI_COMM_WORLD, 1, 0, sizeof(DATA_TYPE) / sizeof(char), 0, 9);
        findZDD.Print();
        findZDD.Stats();
#else
        // normal mode, debug verbose level 0
        error = this->findZDD.Create(MPI_COMM_WORLD, 1, 0, sizeof(DATA_TYPE) / sizeof(char), 0, 0);
#endif
        if (error != ZOLTAN_OK) {
            spdlog::critical("ZDD Create error {}", error);
            findZDD.Print();
            findZDD.Stats();
            std::exit(1);
        }

        static_assert(std::is_trivially_copyable<GID_TYPE>::value, "");
        static_assert(std::is_trivially_copyable<DATA_TYPE>::value, "");
        static_assert(std::is_default_constructible<DATA_TYPE>::value, "");
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
        spdlog::debug("ZDD destructed");
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
        for (auto &id : gidOnLocal) {
            printf("local ID %u on rank %d\n", id, myRank);
        }
        spdlog::debug("ZDD before Update");
        findZDD.Print();
        findZDD.Stats();
#endif

        ZOLTAN_ID_PTR idPtr = gidOnLocal.data();
        DATA_TYPE *dataPtr = dataOnLocal.data();
        int error = findZDD.Update(idPtr, NULL, (char *)dataPtr, NULL, gidOnLocal.size());
        if (error != ZOLTAN_OK) {
            spdlog::critical("ZDD Update error {}", error);
            std::exit(1);
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
        if (dataToFind.size() < gidToFind.size()) {
            dataToFind.resize(gidToFind.size()); // make sure size match
        }
        ZOLTAN_ID_PTR idPtr = gidToFind.data();
        DATA_TYPE *dataPtr = dataToFind.data();
        // ZOLTAN_ID_PTR gid, ZOLTAN_ID_PTR lid, char *data, int *part, const int &
        // count, int *owner
        int status = findZDD.Find(idPtr, NULL, (char *)dataPtr, NULL, gidToFind.size(), NULL);
        return status;
    }
};

#endif /* ZDD_HPP_ */
