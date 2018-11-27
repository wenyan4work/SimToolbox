
#include "MPI/MsgBuffer.hpp"

#include <cstdio>
#include <iostream>
#include <type_traits>
#include <vector>

struct Quad {
    int i, j, k, l;
    std::string name;

    Quad(int i_, int j_, int k_, int l_, std::string name_) : i(i_), j(j_), k(k_), l(l_), name(name_){};
    Quad() = default;

    void pack(MsgBuffer &buf) {
        buf.pack(i);
        buf.pack(j);
        buf.pack(k);
        buf.pack(l);
        buf.pack(name);
    }

    void unpack(MsgBuffer &buf) {
        buf.unpack(i);
        buf.unpack(j);
        buf.unpack(k);
        buf.unpack(l);
        buf.unpack(name);
    }
};

void testObj() {
    Quad q1(0, 1, 2, 3, std::string("first"));
    Quad q2(4, 5, 6, 7, std::string("second"));

    MsgBuffer mybuffer;
    mybuffer.packObj(q1);
    mybuffer.packObj(q2);

    std::vector<Quad> quadvec(2);
    mybuffer.unpackObj(quadvec[0]);
    mybuffer.unpackObj(quadvec[1]);

    for (auto &q : quadvec) {
        std::cout << q.i << q.j << q.k << q.l << q.name << std::endl;
    }
    mybuffer.dump();
}

struct Triplet {
    double a, b, c;
};

void testPOD() {
    Triplet t1{0.1, 0.2, 0.3};
    Triplet t2{0.4, 0.5, 0.6};

    MsgBuffer mybuffer;
    mybuffer.packTrivCopyable(t1);
    mybuffer.packTrivCopyable(t2);

    std::vector<Triplet> tripletvec(2);
    mybuffer.unpackTrivCopyable(tripletvec[0]);
    mybuffer.unpackTrivCopyable(tripletvec[1]);
    for (auto &t : tripletvec) {
        std::cout << t.a << t.b << t.c << std::endl;
    }

    mybuffer.dump();
};

void testConcate() {
    Triplet t1{0.1, 0.2, 0.3};
    Triplet t2{0.4, 0.5, 0.6};

    MsgBuffer bufT;
    bufT.packTrivCopyable(t1);
    bufT.packTrivCopyable(t2);

    Quad q1(0, 1, 2, 3, std::string("first"));
    Quad q2(4, 5, 6, 7, std::string("second"));

    MsgBuffer mybuffer;
    mybuffer.packObj(q1);
    mybuffer.packObj(q2);

    mybuffer.dump();
    std::cout << std::endl;
    bufT.dump();
    std::cout << std::endl;

    mybuffer.concatenate(bufT);
    mybuffer.dump();
    std::cout << std::endl;
}

template <int N>
class Array {
    double test[N];
};

int main() {
    testObj();
    testPOD();
    testConcate();

    using A = Array<5>;
    static_assert(std::is_trivially_copyable<A>::value);

    return 0;
}