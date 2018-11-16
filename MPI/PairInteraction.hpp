/**
 * class to hold interation list information
 *  The 4 FP types can be the same
 */

#ifndef PAIRINTERACTION_HPP_
#define PAIRINTERACTION_HPP_

#include "Util/Buffer.hpp"

#include <cassert>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <mpi.h>
#include <omp.h>

struct SrcGhostInfo {
    int gid;
    int homeRank;
    int destRank;
};

template <class T>
inline MPI_Datatype CreateMPIStructType() {
    static MPI_Datatype type = MPI_DATATYPE_NULL;
    if (type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(sizeof(T), MPI_BYTE, &type);
        MPI_Type_commit(&type);
    }
    return type;
};

template <class TrgFP, class TrgEP, class SrcFP, class SrcEP>
class PairInteraction {
  public:
    const std::vector<TrgFP> &trgFP; // reference to trg Full Particle
    const std::vector<SrgFP> &srcFP; // reference to src Full Particle

    std::vector<TrgEP> trgEP;      // 1 on 1 mapping to trgFP;
    std::vector<SrcEP> srcEP;      // 1 on 1 mapping to srcFP;
    std::vector<SrcEP> srcEPGhost; // corresponds to src received from other mpi ranks

    // Index of neighbor in srcEP and srcEPGhost, sequentially ordered
    std::vector<std::vector<int>> neighborIndex;

    // this is setup by FDPS, sorted by home rank
    std::vector<SrcGhostInfo> srcGhostRecvInfo; // 1 on 1 mapping to srcEPGhost, where are they from;
                                                // sorted: a.homerank < b.homerank, a.gid < b.gid
                                                // MPI relies on this

  private:
    int rank; // keep constant after initialization
    int nprocs;

    // computed with srcEPGhostGid and srcEPGhostHomeRank
    std::vector<SrcGhostInfo> srcGhostSendInfo;
    // the packed size of char of each srcEP
    std::vector<int> srcEPPackSize;

    std::unordered_map<int, int> srcEPGidIndex;

    std::vector<char> sendBuf;
    std::vector<char> recvBuf;

    std::vector<int> sendBufByteCnt;
    std::vector<int> sendBufByteDispls;
    std::vector<int> recvBufByteCnt;
    std::vector<int> recvBufByteDispls;

  public:
    // constructor
    PairInteraction(std::vector<TrgFP> &trgFPVec, std::vector<SrcFP> &srcFPVec) : trgFP(trgFPVec), srcFP(srcFPVec) {
        MPI_Comm_rank(&rank);
        MPI_Comm_size(&nprocs);

        const int trgCnt = trgFP.size();
        trgEP.resize(trgCnt);

        const int srcCnt = srcFP.size();
        srcEP.resize(srcCnt);
        srcEPDestRank.resize(srcCnt);

        srcEPGhost.clear();
        srcEPGhostHomeRank.clear();
        neighborIndex.resize(trgCnt);

        updateLocalEP();
    }

    // member function
    void updateLocalEP() {
        // keep information, only change data of EP
        const int trgCnt = trgFP.size();
        assert(trgCnt == TrgEP.size());
#pragma omp parallel for
        for (size_t i = 0; i < trgCnt; i++) {
            trgEP[i].CopyFromFP(trgFP[i]);
        }

        const int srcCnt = srcFP.size();
        assert(srcCnt == srcEP.size());
#pragma omp parallel for
        for (size_t i = 0; i < srcCnt; i++) {
            srcEP[i].CopyFromFP(srcFP[i]);
        }
    }

    void updateSrcGhostEP() {
        // forward scatter from SrcEP to SrcGhostEP
        // task: each rank pack and send srcEP
        //       each rank receive and unpack to srcEPGhost
        // precondition: calcSendInfo() has been called

        // loop over srcGhostSendInfo, prepare data structure for mpi send/recv
        // sequentially pack them into sendBuf
        // allow size of each srcEP to change from call to call

        sendBufByteCnt.resize(nprocs, 0);
        for (auto &g : srcGhostSendInfo) {
            auto &it = srcEPGidIndex.find(g.gid);
            int index;
            if (it != srcEPGidIndex.end()) {
                index = it->second;
            } else {
                printf("PairInteractionError, gid not found in local srcEP!\n");
                exit(1);
            }
            auto &srcEPJ = srcEP[index];
            const int sendBufSizeBefore = sendBuf.size();
            srcEPJ.pack(sendBuf);
            const int packByteCnt = sendBuf.size() - sendBufSizeBefore srcEPPackSize.push_back(packByteCnt);
            sendBufByteCnt[g.destRank] += (packByteCnt);
        }
        sendBufByteDispls.clear();
        sendBufByteDispls.resize(nprocs + 1, 0);
        for (int j = 0; j < nprocs; j++) {
            sendBufByteDispls[j + 1] = sendBufByteDispls[j] + sendBufByteCnt[j];
        }
        recvBufByteCnt.clear();
        recvBufByteCnt.resize(nprocs, 0);
        MPI_Alltoall(sendBufBytesCnt.data(), nprocs, MPI_INT, //
                     recvBufBytesCnt.data(), nprocs, MPI_INT, MPI_COMM_WORLD);

        recvBufByteDispls.clear();
        recvBufByteDispls.resize(nprocs + 1, 0);
        for (int j = 0; j < nprocs; j++) {
            recvBufByteDispls[j + 1] = recvBufByteDispls[j] + recvBufByteCnt[j];
        }

        MPI_Alltoallv(sendBuf.data(), sendBufByteCnt.data(), sendBufByteDispls.data(), MPI_CHAR, //
                      recvBuf.data(), recvBufByteCnt.data(), recvBufByteDispls.data(), MPI_CHAR, MPI_COMM_WORLD);

        // unpack to srcEPGhost
        const int srcEPGhostSize = srcGhostRecvInfo.size();
        srcEPGhost.resize(srcEPGhostSize);

        for (int i = 0; i < srcEPGhostSize; i++) {
            srcEPGhost[i].unpack(recvBuf, readPos);
            if (srcEPGhost[i].getGid() != srcGhostRecvInfo[i].gid) {
                printf("Send/Recv data mismatch error\n");
                exit(1);
            }
        }
    }

    void calcSendInfo() {
        // update srcEPDestRank with known srcEPGhostHomeRank
        const int srcEPGhostSize = srcGhostSendInfo.size();

        // sanity check
        for (auto &s : srcGhostSendInfo) {
            assert(s.destRank == rank); // ghost are received by the local rank
            assert(s.homeRank >= 0);
            assert(s.homeRank < nprocs);
        }

        std::vector<int> sendCnt(nprocs, 0); // send count to each rank
        std::vector<int> recvCnt(nprocs, 0); // recv count from each rank
#pragma omp parallel for
        for (int i = 0; i < srcGhostSendInfo; i++) {
#pragma omp atomic update
            sendCnt[srcGhostSendInfo[i].homeRank]++;
        }

        MPI_Alltoall(sendCnt.data(), nprocs, MPI_INT, //
                     recvCnt.data(), nprocs, MPI_INT, MPI_COMM_WORLD);

        std::vector<int> recvDispls(nprocs + 1, 0);
        for (int i = 0; i < nprocs; i++) {
            recvDispls[i + 1] = recvDispls[i] + recvCnt[i];
        }

        srcGhostSendInfo.clear();
        srcGhostSendInfo.resize(recvDispls.back());

        std::vector<int> sendDispls(nprocs + 1, 0);
        for (int i = 0; i < nprocs; i++) {
            sendDispls[i + 1] = sendDispls[i] + sendCnt[i];
        }

        auto &TYPE = CreateMPIStructType<SrcGhostInfo>();

        MPI_Alltoallv(srcGhostRecvInfo.data(), sendCnt.data(), sendDispls.data(), TYPE, //
                      srcGhostSendInfo.data(), recvCnt.data(), recvDispls.data(), TYPE, MPI_COMM_WORLD);
        // srcGhostSendInfo is sorted
        // after transfer, srcGhostSendInfo is sorted by destRank

        const int srcLocalSize = srcEP.size();
        assert(srcLocalSize == srcFP.size());
        for (int i = 0; i < srcLocalSize; i++) {
            srcEPGidIndex.insert(srcFP[i].getGid(), i);
        }
    }
};

#endif

/**
 * Explanation:
 * SrcGhostRecvInfo [gid, homeRank, destRank] , sorted with destRank and then gid
 * rank 0: [0,1,0] [1,1,0] [2,2,0]
 *      1: [3,0,1] [4,0,1] [5,2,1]
 *      2: [6,0,2] [7,0,2] [8,1,2] [9,1,2]
 *
 * after MPI_Alltoallv, received data are arranged with send rank id
 * srcGhostSendInfo
 * rank 0: [3,0,1] [4,0,1] [6,0,2] [7,0,2] (gid 3 with homeRank 0 goes to destRank 1, etc)
 *      1: [0,1,0] [1,1,0] [8,1,2] [9,1,2]
 *      2: [2,2,0] [5,2,1]
 */