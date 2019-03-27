#include "GeoUtil.hpp"
#include "Util/TRngPool.hpp"

#include <cmath>
#include <cstdio>

bool testFindPBCImage(double lb, double ub, double x) {
    double ximg = x;
    findPBCImage(lb, ub, ximg);

    bool pass = (ximg < ub) && (ximg > lb);
    double jump = (ximg - x) / (ub - lb); // should be an integer
    pass = pass && (jump - round(jump) < 1e-8);

    if (!pass) {
        printf("%g,%g,%g,%g\n", lb, ub, x, ximg);
    }

    return pass;
}

bool testFindPBCImageTrg(double lb, double ub, double x, double trg) {

    double trgimg = trg;
    findPBCImage(lb, ub, trgimg);

    findPBCImage(lb, ub, x, trg); // x can be out of bound [lb,ub)

    // x-trg has minimum distance between all images of x
    bool pass = (fabs(trg - trgimg) < (ub - lb) / 1e10)          //
                && fabs(x - trg) < fabs(x + (ub - lb) - trg)     //
                && fabs(x - trg) < fabs(x - (ub - lb) - trg)     //
                && fabs(x - trg) < fabs(x + 2 * (ub - lb) - trg) //
                && fabs(x - trg) < fabs(x - 2 * (ub - lb) - trg);

    if (!pass) {
        printf("%g,%g,%g,%g,%g\n", lb, ub, x, trg, trgimg);
    }

    return pass;
}

int main() {
    bool pass = true;

    TRngPool rngPool(10);

    for (int i = 0; i < 1000; i++) {
        double lb = -rngPool.getU01() * 10 - 1;
        double ub = rngPool.getU01() * 10 + 1;
        double x = rngPool.getU01() * 1000 - 500;
        pass = (pass && testFindPBCImage(lb, ub, x));
    }

    for (int i = 0; i < 1000; i++) {
        double lb = -rngPool.getU01() * 10 - 1;
        double ub = rngPool.getU01() * 10 + 1;
        double x = rngPool.getU01() * 1000 - 500;
        double trg = rngPool.getU01() * 1000 - 500;
        pass = (pass && testFindPBCImageTrg(lb, ub, x, trg));
    }

    if (pass) {
        printf("TestPassed\n");
    }

    return 0;
}