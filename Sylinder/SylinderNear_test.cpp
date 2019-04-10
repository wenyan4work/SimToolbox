#include "SylinderNear.hpp"
#include "Util/EigenDef.hpp"
#include <cmath>

void testEpsilon() {
    constexpr double epsilonConst[3][3][3] = {{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, //
                                              {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, //
                                              {{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}};
    double epsilon[3][3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                epsilon[i][j][k] = 0.0;
            }
        }
    }
    epsilon[0][1][2] = 1.0;
    epsilon[1][2][0] = 1.0;
    epsilon[2][0][1] = 1.0;
    epsilon[1][0][2] = -1.0;
    epsilon[2][1][0] = -1.0;
    epsilon[0][2][1] = -1.0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                if (epsilon[i][j][k] != epsilonConst[i][j][k]) {
                    printf("epsilonConst not correct\n");
                    exit(1);
                }
            }
        }
    }
}

void testFixedPair() {
    omp_set_num_threads(1);
    Evec3 P0(1, 0, 0);
    Evec3 P1(0, sqrt(3), 0);
    Evec3 Q0(0, 0, 1);
    Evec3 Q1(2, 2 * sqrt(3), 1);

    CalcSylinderNearForce calc;
    calc.colPoolPtr = std::make_shared<CollisionBlockPool>();
    calc.colPoolPtr->resize(1);

    std::vector<SylinderNearEP> sylinderP(1);
    std::vector<SylinderNearEP> sylinderQ(1);

    { // setup P
        Evec3 center = (P0 + P1) / 2;
        Evec3 direction = (P1 - P0).normalized();
        double length = (P1 - P0).norm();
        sylinderP[0].gid = 0;
        sylinderP[0].globalIndex = 0;
        sylinderP[0].rank = 0;
        sylinderP[0].radius = 0.4;
        sylinderP[0].radiusCollision = 0.4;
        sylinderP[0].length = length;
        sylinderP[0].lengthCollision = length;
        sylinderP[0].pos[0] = center[0];
        sylinderP[0].pos[1] = center[1];
        sylinderP[0].pos[2] = center[2];
        sylinderP[0].direction[0] = direction[0];
        sylinderP[0].direction[1] = direction[1];
        sylinderP[0].direction[2] = direction[2];
    }
    { // setup Q
        Evec3 center = (Q0 + Q1) / 2;
        Evec3 direction = (Q1 - Q0).normalized();
        double length = (Q1 - Q0).norm();
        sylinderQ[0].gid = 1;
        sylinderQ[0].globalIndex = 1;
        sylinderQ[0].rank = 0;
        sylinderQ[0].radius = 0.5;
        sylinderQ[0].radiusCollision = 0.5;
        sylinderQ[0].length = length;
        sylinderQ[0].lengthCollision = length;
        sylinderQ[0].pos[0] = center[0];
        sylinderQ[0].pos[1] = center[1];
        sylinderQ[0].pos[2] = center[2];
        sylinderQ[0].direction[0] = direction[0];
        sylinderQ[0].direction[1] = direction[1];
        sylinderQ[0].direction[2] = direction[2];
    }
    ForceNear fnear;
    calc(sylinderP.data(), 1, sylinderQ.data(), 1, &fnear);
    printf("%d collisions recorded\n", calc.colPoolPtr->front().size());
    auto stress = calc.colPoolPtr->front().front().stress;
    printf("%g,%g,%g,%g,%g,%g,%g,%g,%g\n", stress[0], stress[1], stress[2], stress[3], stress[4], stress[5], stress[6],
           stress[7], stress[8]);
    bool pass = true;
    pass = pass && fabs(stress[0] - 0) < 1e-6;
    pass = pass && fabs(stress[1] - 0) < 1e-6;
    pass = pass && fabs(stress[2] - 0.0160681) < 1e-6;
    pass = pass && fabs(stress[3] - 0) < 1e-6;
    pass = pass && fabs(stress[4] - 0) < 1e-6;
    pass = pass && fabs(stress[5] - 0.0278307) < 1e-6;
    pass = pass && fabs(stress[6] - 0.0160681) < 1e-6;
    pass = pass && fabs(stress[7] - 0.0278307) < 1e-6;
    pass = pass && fabs(stress[8] - 1) < 1e-6;

    if (pass) {
    } else {
        printf("stress wrong\n");
        exit(1);
    }
}

int main() {
    testEpsilon();
    testFixedPair();
    return 0;
}