#include "Boundary.hpp"
#include "Util/EigenDef.hpp"

#include <iostream>
#include <memory>

int main() {
    double center[3] = {2, 2, 2};
    double axis[3] = {1, 0, 0};

    bool pass = true;

    { // test spherical shell inside
        std::unique_ptr<Boundary> b0 = std::make_unique<SphereShell>(center, 2.0, true);
        for (int i = 0; i < 1000; i++) {
            Evec3 query = Evec3::Random() * 5;
            Evec3 projection = Evec3::Zero();
            Evec3 normI = Evec3::Zero();
            b0->project(query.data(), projection.data(), normI.data());
            if (!b0->check(query.data(), projection.data(), normI.data())) {
                pass = false;
                break;
            }
        }
        if (!pass) {
            std::exit(1);
        }
    }
    { // test spherical shell outside
        std::unique_ptr<Boundary> b0 = std::make_unique<SphereShell>(center, 2.0, false);
        for (int i = 0; i < 1000; i++) {
            Evec3 query = Evec3::Random() * 5;
            Evec3 projection = Evec3::Zero();
            Evec3 normI = Evec3::Zero();
            b0->project(query.data(), projection.data(), normI.data());
            if (!b0->check(query.data(), projection.data(), normI.data())) {
                pass = false;
                break;
            }
        }
        if (!pass) {
            std::exit(1);
        }
    }
    // { std::unique_ptr<Boundary> b1 = std::make_unique<Wall>(center, axis); }
    // { std::unique_ptr<Boundary> b2 = std::make_unique<Tube>(center, axis, 2.0, true); }

    if (pass) {
        printf("TestPassed\n");
    }
    return 0;
}