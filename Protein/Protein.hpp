#ifndef PROTEIN_HPP
#define PROTEIN_HPP

#include <unordered_map>
#include <vector>

#include "Util/Buffer.hpp"
#include "Util/EigenDef.hpp"
#include "Util/IOHelper.hpp"

constexpr int INVALID = -1;

class Protein {
  public:
    int gid = INVALID;
    // bool walkOff;
    // bool bindAntiParallel;
    // bool fixedEnd0;
    // double fixedLocation;
    // double lenMP0; // original length

    // time varying status
    int idBind[2]; // = INVALID if not bind
                   // 0 at MT center, positive toward plus end
    Evec3 posEnd[2];

    static void writeVTP(const std::vector<Protein> &protein, const std::string &prefix, const std::string &postfix,
                         int rank);
    static void writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs);
};

#endif