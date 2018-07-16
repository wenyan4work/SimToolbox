#ifndef _NEAR_INTERAC_HPP_
#define _NEAR_INTERAC_HPP_

#include "sctl/sctl.hpp"

// Interface of OBJ:
// const double * Coord() const;
// const double Rad() const;
// void Pack(std::vector<char> &buff) const ;
// void Unpack(const std::vector<char> &buff) ;
// optional:  void CopyFromFull(const FullType &) ;

template <class Real, int DIM>
struct Pair {
    using Long = sctl::Long;

    Long trg_idx, src_idx;
    Real srcShift[DIM];
    // the shift added to src.coord() where the pair is detected

    int operator<(const Pair<Real, DIM> &p1) const {
        return std::pair<Long, Long>(trg_idx, src_idx) < std::pair<Long, Long>(p1.trg_idx, p1.src_idx);
    }
};

template <class Real, int DIM>
class NearInteraction {
    typedef sctl::Morton<DIM> MID;
    typedef sctl::Long Long;

    struct ObjData {
        int operator<(const ObjData &p1) const { return mid < p1.mid; }
        MID mid;   // Morton ID
        Long Rglb; // global sorted unique id

        Real rad;
        Real coord[DIM]; // coord supplied, not necessarily in the original box
        // sctl::StaticArray<Real,DIM> coord; // not trivially copyable
    };

  public:
    NearInteraction() { Init(); }

    NearInteraction(sctl::Comm comm) : comm_(comm) { Init(); }

    void SetPeriodLength(sctl::Integer d, Real len) {
        SCTL_ASSERT(d < DIM);
        period_length[d] = len;
    }

    template <class SrcObj, class TrgObj>
    void SetupRepartition(const std::vector<SrcObj> &src_vec, const std::vector<TrgObj> &trg_vec);

    template <class SrcObj, class TrgObj>
    void SetupNearInterac(const std::vector<SrcObj> &src_vec, const std::vector<TrgObj> &trg_vec);

    const std::vector<Pair<Real, DIM>> &GetInteractionList() const { return trg_src_pair; }

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
            period_length[i] = 0;
            period_length0[i] = 0;
        }
    }

    template <class ObjType>
    void ForwardScatter(const std::vector<ObjType> &in_vec, std::vector<ObjType> &out_vec,
                        const sctl::Vector<Long> &recv_idx) const;

    template <class ObjType>
    void ReverseScatter(const std::vector<ObjType> &in_vec, std::vector<ObjType> &out_vec,
                        const sctl::Vector<Long> &send_idx) const;

    sctl::Comm comm_;
    sctl::Integer depth;

    sctl::Vector<Long> TRglb, SRglb; // globally sorted sequential internal ID

    // sctl::Vector<std::pair<Long, Long>> TSPair;
    // std::vector<std::pair<Long, Long>> trg_src_pair;

    sctl::Vector<Pair<Real, DIM>> TSPair;
    std::vector<Pair<Real, DIM>> trg_src_pair;

    sctl::StaticArray<Real, DIM> period_length, period_length0;
    // Real s = sctl::pow<Real>(2, depth);
    // period_length0[i] = std::floor((period_length[i] / BBox.L) * s) / s;

    sctl::Vector<ObjData> SData_, TData_; // Globally sorted
};

#include "NearInteraction.txx"

#endif //_NEAR_INTERAC_HPP_
