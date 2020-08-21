#include "QuadInt.hpp"

struct Integrand {
    int e = 2;
    double operator()(double s) const { return sin(exp(-2 * s)); }
};

bool test(char choice) {
    Integrand f;
    using Quad = QuadInt<48>;
    std::vector<Quad> quads;
    for (int i = 0; i < 20; i++) {
        quads.push_back(Quad(i * 2 + 2, choice));
    }
    for (auto &q : quads) {
        q.print();
    }

    for (auto &q : quads) {
        printf("npts: %d, int[-1,1]: %18.10g, ints[-1,1]: %18.10g\n", q.getSize(), q.integrate(f), q.integrateS(f));
    }

    auto &q = quads.back();
    double a = q.integrate(f);
    double b = q.integrateS(f);

    if (fabs(a - 0.680910296840562) < 1e-8 && fabs(b - 0.159315026657831) < 1e-8)
        return true;
    else
        return false;
}

int main(int argc, char **argv) {
    if (test('c') && test('g'))
        printf("TestPassed\n");
    return 0;
}