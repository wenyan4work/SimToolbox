#include "GeoUtil.hpp"

#include <cstdio>
#include <cmath>

void testFindImage() {
    const double lb = 0, ub = 1;
    double x;
    x = -0.1;
    findPBCImage(lb, ub, x);
    printf("%g\n", x);
    x = 0.1;
    findPBCImage(lb, ub, x);
    printf("%g\n", x);
    x = 1.1;
    findPBCImage(lb, ub, x);
    printf("%g\n", x);
}

void testFindImage(double trg) {
    const double lb = 0, ub = 1;
    double x;
    x = -0.1;
    findPBCImage(lb, ub, x, trg);
    printf("%g\n", x);
    x = 0.1;
    findPBCImage(lb, ub, x, trg);
    printf("%g\n", x);
    x = 1.1;
    findPBCImage(lb, ub, x, trg);
    printf("%g\n", x);
}

int main() {
    testFindImage();
    testFindImage(0.3);
    testFindImage(1.9);
    return 0;
}