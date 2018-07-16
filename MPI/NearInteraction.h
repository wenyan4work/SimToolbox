#ifndef NEARINTERAC_HPP_
#define NEARINTERAC_HPP_

#include "sctl/sctl.hpp"

template <class Real, int DIM>
class NearInteraction {
    using MID = sctl::Morton<DIM>;
    using Long = sctl::Long;

    struct ObjData {
        MID mid;         // Morton ID
        Long Rglb;       // Global ID, sequentially ordered from rank 0
        Real rad;        // search radius
        Real coord[DIM]; // coordinate

        // Morton ID Sort
        int operator<(const ObjData &p1) const { return mid < p1.mid; }
        // sctl::StaticArray<Real,DIM> coord; // not trivially copyable
    };

  public:
    NearInteraction() { Init(); }

    NearInteraction(sctl::Comm comm) : comm_(comm) { Init(); }

    void SetPeriodLength(sctl::Integer d, Real len) {
        assert(d < DIM);
        period_length[d] = len;
    }

    template <class SrcObj, class TrgObj>
    void SetupRepartition(const std::vector<SrcObj> &src_vec, const std::vector<TrgObj> &trg_vec);

    template <class SrcObj, class TrgObj>
    void SetupNearInterac(const std::vector<SrcObj> &src_vec, const std::vector<TrgObj> &trg_vec);

    const std::vector<std::pair<Long, Long>> &GetInteractionList() const { return trg_src_pair; }

    template <class ObjType>
    void ForwardScatterSrc(const std::vector<ObjType> &in, std::vector<ObjType> &out) const;

    template <class ObjType>
    void ForwardScatterTrg(const std::vector<ObjType> &in, std::vector<ObjType> &out) const;

    template <class ObjType>
    void ReverseScatterTrg(const std::vector<ObjType> &in, std::vector<ObjType> &out) const;

    void Barrier() { comm_.Barrier(); }

  private:
    void Init() {
        for (sctl::Integer i = 0; i < DIM; i++) {
            period_length[i] = 0;  // real period length
            period_length0[i] = 0; // scaled period length
        }
    }

    template <class ObjType>
    void ForwardScatter(const std::vector<ObjType> &in_vec, std::vector<ObjType> &out_vec,
                        const sctl::Vector<Long> &recv_idx) const;

    template <class ObjType>
    void ReverseScatter(const std::vector<ObjType> &in_vec, std::vector<ObjType> &out_vec,
                        const sctl::Vector<Long> &send_idx) const;

    sctl::Comm comm_; // sctl communicator
    sctl::Integer depth;

    sctl::Vector<Long> TRglb, SRglb;      // Trg and Src global sequentially ordered id
    sctl::Vector<ObjData> SDataSorted, TDataSorted; // Globally sorted

    // To be changed to avoid length overflow
    sctl::Vector<std::pair<Long, Long>> TSPair;
    std::vector<std::pair<Long, Long>> trg_src_pair;

    // periodic length and scaled periodic length
    sctl::StaticArray<Real, DIM> period_length, period_length0;
};

#endif //_NEAR_INTERAC_HPP_
