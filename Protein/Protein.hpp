#ifndef PROTEIN_HPP_
#define PROTEIN_HPP_

#include "Util/Buffer.hpp"
#include "Util/EigenDef.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"

#include <type_traits>
#include <unordered_map>
#include <vector>

class Protein {
  public:
    int gid = GEO_INVALID_INDEX;
    // bool walkOff;
    // bool bindAntiParallel;
    // bool fixedEnd0;
    // double fixedLocation;
    // double lenMP0; // original length

    // time varying status
    int idBind[2]; // = INVALID if not bind
                   // 0 at MT center, positive toward plus end
    double posEnd[2][3];

    static void writeVTP(const std::vector<Protein> &protein, const std::string &prefix, const std::string &postfix,
                         int rank);
    static void writePVTP(const std::string &prefix, const std::string &postfix, const int nProcs);
};

static_assert(std::is_trivially_copyable<Protein>::value, "");
static_assert(std::is_default_constructible<Protein>::value, "");

#endif
