#include "../../Collision/CollisionSylinder.hpp"

int main() {
    double pos1[3] = {0.1, 0.2, 0.3};
    double pos2[3] = {0.4, 0.5, 0.6};
    Sylinder sy1(1, 1.0, 1.5, 5.0, 5.0, pos1);
    Sylinder sy2(2, 1.0, 1.5, 5.0, 5.0, pos2);

    CollisionSylinder sycol1, sycol2;
    sycol1.CopyFromFull(sy1);
    sycol2.CopyFromFull(sy2);

    CollisionBlock cb;
    bool col = sycol1.collide(sycol2, cb);
    std::cout << "locI " << cb.posI.transpose() << ", locJ " << cb.posJ.transpose() << std::endl;
    std::cout << "posI " << Emap3(sy1.pos).transpose() << ", posJ " << Emap3(sy2.pos).transpose() << std::endl;
    std::cout << "phi0: " << cb.phi0 << std::endl;

    return 0;
}