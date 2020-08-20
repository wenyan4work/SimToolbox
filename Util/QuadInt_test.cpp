#include "QuadInt.hpp"

int main() {
    struct Integrand {
        int e = 2;
        double operator()(double s) const { return pow(s, e)*cos(s); }
    };
    Integrand f;
    {
        QuadInt<2> quad;
        quad.print();
        printf("integral %12g\n", quad.integrate(f));
    }
    {
        QuadInt<4> quad;
        quad.print();
        printf("integral %12g\n", quad.integrate(f));
    }
    {
        QuadInt<6> quad;
        quad.print();
        printf("integral %12g\n", quad.integrate(f));
    }
    {
        QuadInt<8> quad;
        quad.print();
        printf("integral %12g\n", quad.integrate(f));
    }
    return 0;
}