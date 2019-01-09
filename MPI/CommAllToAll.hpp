/**
 * @file CommAllToAll.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief run MPI AllToAll for abstract data type Data
 * @version 0.1
 * @date 2019-01-09
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef COMMALLTOALL_HPP_
#define COMMALLTOALL_HPP_

#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>

template <class Data>
class CommAllToAll {
    static_assert(std::is_trivially_copyable<Sylinder>::value, "");
    static_assert(std::is_default_constructible<Sylinder>::value, "");

    int rank;
    int nProcs;

    std::vector<int> nSend;
    std::vector<int> nSendDisp;
    std::vector<int> nRecv;
    std::vector<int> nRecvDisp;

  public:
    std::vector<int> sendDestRank;
    std::vector<Data> sendData;
    std::vector<int> recvSrcRank;
    std::vector<Data> recvData;

    CommAllToAll() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
        nSend.resize(nProcs);
        nSendDisp.resize(nProcs);
        nRecv.resize(nProcs);
        nRecvDisp.resize(nProcs);
    }

    void exchange() {
        if (sendData.size() != sendDestRank.size()) {
            printf("send data and rank error\n");
            exit(1);
        }

        



    }
};

#endif