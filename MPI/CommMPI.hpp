/**
 * @file CommMpi.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief run MPI AllToAll for abstract data type Data
 * @version 0.1
 * @date 2019-01-09
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef COMMMPI_HPP_
#define COMMMPI_HPP_

#include "Util/SortUtil.hpp"

#include <numeric>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>

/**
 * @brief create a MPI struct type from T
 *
 * @tparam T
 * @return MPI_Datatype
 */
template <class T>
inline MPI_Datatype createMPIStructType() {
    static_assert(std::is_trivially_copyable<T>::value, "");
    static_assert(std::is_default_constructible<T>::value, "");
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(sizeof(T), MPI_BYTE, &type);
        MPI_Type_commit(&type);
    }
    return type;
};

/**
 * @brief utility class for mpi comm handling
 *
 */
class CommMPI {

    int rank;
    int nProcs;

    // naming consistent with MPI funciton parameters
    std::vector<int> sendCounts;
    std::vector<int> sendDispls;
    std::vector<int> recvCounts;
    std::vector<int> recvDispls;

  public:
    /**
     * @brief Construct a new CommMPI object
     *
     */
    CommMPI() noexcept {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
        sendCounts.resize(nProcs, 0);
        sendDispls.resize(nProcs + 1, 0);
        recvCounts.resize(nProcs, 0);
        recvDispls.resize(nProcs + 1, 0);
    }

    int getRank() const { return rank; }
    int getSize() const { return nProcs; }

    /**
     * @brief clear internal counters
     *
     */
    void clear() {
        assert(sendCounts.size() == nProcs);
        assert(recvCounts.size() == nProcs);
        assert(sendDispls.size() == nProcs + 1);
        assert(recvDispls.size() == nProcs + 1);
#pragma omp parallel sections
        {
#pragma omp section
            { std::fill(sendCounts.begin(), sendCounts.end(), 0); }
#pragma omp section
            { std::fill(sendDispls.begin(), sendDispls.end(), 0); }
#pragma omp section
            { std::fill(recvCounts.begin(), recvCounts.end(), 0); }
#pragma omp section
            { std::fill(recvDispls.begin(), recvDispls.end(), 0); }
        }
    }

    /**
     * @brief alltoallv for Data to each rank
     *
     * @tparam Data
     * @param sendDestRank rank id for each data
     * @param sendData data holder
     * @param recvSrcRank the src rank id of each received data
     * @param recvData received data
     */
    template <class Data>
    void exchangeAllToAllV(std::vector<int> &sendDestRank, std::vector<Data> &sendData, //
                           std::vector<int> &recvSrcRank, std::vector<Data> &recvData);
};

template <class Data>
void CommMPI::exchangeAllToAllV(std::vector<int> &sendDestRank, std::vector<Data> &sendData, //
                                std::vector<int> &recvSrcRank, std::vector<Data> &recvData) {
    static_assert(std::is_trivially_copyable<Data>::value, "");
    static_assert(std::is_default_constructible<Data>::value, "");

    // forward scatter from sendData to recvData
    clear();
    auto mpiDataType = createMPIStructType<Data>();

    // step 1 sort send
    assert(sendData.size() == sendDestRank.size());
    sortDataWithTag(sendDestRank, sendData);

    // step 2 calc nSend and nSendDisp
    const int nSendTotal = sendData.size();
    assert(sendCounts.size() == nProcs);
    assert(recvCounts.size() == nProcs);
#pragma omp parallel for
    for (int i = 0; i < nSendTotal; i++) {
        const int dest = sendDestRank[i];
        assert(dest >= 0 && dest < nProcs);
#pragma omp atomic update
        sendCounts[dest]++;
    }

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // calc disp
    assert(sendDispls.size() == nProcs + 1);
    assert(recvDispls.size() == nProcs + 1);
#pragma omp parallel sections
    {
#pragma omp section
        {
            sendDispls[0] = 0;
            std::partial_sum(sendCounts.begin(), sendCounts.end(), sendDispls.begin() + 1);
        }
#pragma omp section
        {
            recvDispls[0] = 0;
            std::partial_sum(recvCounts.begin(), recvCounts.end(), recvDispls.begin() + 1);
        }
    }

    recvData.resize(recvDispls.back());
    recvSrcRank.resize(recvDispls.back());

    MPI_Alltoallv(sendData.data(), sendCounts.data(), sendDispls.data(), mpiDataType, //
                  recvData.data(), recvCounts.data(), recvDispls.data(), mpiDataType, MPI_COMM_WORLD);
#pragma omp parallel for
    for (int i = 0; i < nProcs; i++) {
        int lb = recvDispls[i];
        int ub = recvDispls[i + 1];
        std::fill(recvSrcRank.begin() + lb, recvSrcRank.begin() + ub, i);
    }
}

#endif