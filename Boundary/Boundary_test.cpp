#include "Boundary.hpp"
#include "Util/EigenDef.hpp"

#include <iostream>
#include <memory>

int main() {
    double center[3] = {2, 2, 2};
    double axis[3] = {1, 2, 3};

    bool pass = true;

    {
        std::cout << "test spherical shell inside" << std::endl;
        std::unique_ptr<Boundary> b = std::make_unique<SphereShell>(center, 2.0, true);
        for (int i = 0; i < 1000; i++) {
            Evec3 query = (Evec3::Random() * 5) + Emap3(center);
            Evec3 projection = Evec3::Zero();
            Evec3 normI = Evec3::Zero();
            b->project(query.data(), projection.data(), normI.data());
            if (!b->check(query.data(), projection.data(), normI.data())) {
                pass = false;
                break;
            }
        }
        if (!pass) {
            std::exit(1);
            printf("Error\n");
        }
    }
    {
        std::cout << "test spherical shell outside" << std::endl;
        std::unique_ptr<Boundary> b = std::make_unique<SphereShell>(center, 2.0, false);
        for (int i = 0; i < 1000; i++) {
            Evec3 query = (Evec3::Random() * 5) + Emap3(center);
            Evec3 projection = Evec3::Zero();
            Evec3 normI = Evec3::Zero();
            b->project(query.data(), projection.data(), normI.data());
            if (!b->check(query.data(), projection.data(), normI.data())) {
                pass = false;
                break;
            }
        }
        if (!pass) {
            std::exit(1);
            printf("Error\n");
        }
    }
    {
        std::cout << "test wall" << std::endl;
        std::unique_ptr<Boundary> b = std::make_unique<Wall>(center, axis);
        for (int i = 0; i < 1000; i++) {
            Evec3 query = (Evec3::Random() * 5) + Emap3(center);
            Evec3 projection = Evec3::Zero();
            Evec3 normI = Evec3::Zero();
            b->project(query.data(), projection.data(), normI.data());
            if (!b->check(query.data(), projection.data(), normI.data())) {
                pass = false;
                break;
            }
        }
        if (!pass) {
            std::exit(1);
            printf("Error\n");
        }
    }

    {
        std::cout << "test tube inside" << std::endl;
        std::unique_ptr<Boundary> b = std::make_unique<Tube>(center, axis, 2.0, true);
        for (int i = 0; i < 1000; i++) {
            Evec3 query = (Evec3::Random() * 5) + Emap3(center);
            Evec3 projection = Evec3::Zero();
            Evec3 normI = Evec3::Zero();
            b->project(query.data(), projection.data(), normI.data());
            if (!b->check(query.data(), projection.data(), normI.data())) {
                pass = false;
                break;
            }
        }
        if (!pass) {
            std::exit(1);
            printf("Error\n");
        }
    }
    {
        std::cout << "test tube outside" << std::endl;
        std::unique_ptr<Boundary> b = std::make_unique<Tube>(center, axis, 2.0, false);
        for (int i = 0; i < 1000; i++) {
            Evec3 query = (Evec3::Random() * 5) + Emap3(center);
            Evec3 projection = Evec3::Zero();
            Evec3 normI = Evec3::Zero();
            b->project(query.data(), projection.data(), normI.data());
            if (!b->check(query.data(), projection.data(), normI.data())) {
                pass = false;
                break;
            }
        }
        if (!pass) {
            std::exit(1);
            printf("Error\n");
        }
    }

    if (pass) {
        printf("TestPassed\n");
    }
    return 0;
}