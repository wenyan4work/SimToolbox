/**
 * @file QuadInt.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2020-08-20
 *
 * @copyright Copyright (c) 2020
 *
 */
#ifndef QUADINT_HPP
#define QUADINT_HPP

#include "Gauss_Legendre_Nodes_and_Weights.hpp"

#include <cmath>
#include <cstdio>

/**
 * @brief class to compute Chebyshev-Gauss Quadrature
 * https://mathworld.wolfram.com/Chebyshev-GaussQuadrature.html
 * @tparam N
 */
template <int N>
class QuadInt {
    double points[N];
    double weights[N];

  public:
    QuadInt() {
        // initialize points and weights
#ifdef CHEBYSHEV
        const double lowerB = -1;
        const double upperB = 1;
        for (int iQuadPt = 0; iQuadPt < N; iQuadPt++) {
            double k = iQuadPt + 1;
            double xi = -std::cos((2 * k - 1) * M_PI / (2 * N));
            weights[iQuadPt] = (upperB - lowerB) / 2 * M_PI / N * std::sqrt(1 - xi * xi);
            points[iQuadPt] = lowerB + (upperB - lowerB) * (xi + 1) / 2;
        }
#else
        std::vector<double> s, w;
        Gauss_Legendre_Nodes_and_Weights<double>(N, s, w);
        for (int i = 0; i < N; i++) {
            weights[i] = w[i];
            points[i] = s[i];
        }
#endif
    }

    int getSize() const { return N; }
    double *getPoints() const { return points; }
    double *getWeights() const { return weights; }

    void print() const {
        for (int i = 0; i < N; i++) {
            printf("s %12g, w %12g\n", points[i], weights[i]);
        }
    }

    template <class Func>
    double integrate(Func f) {
        double sum = 0;
        for (int i = 0; i < N; i++) {
            sum += f(points[i]) * weights[i];
        }
        return sum;
    }
};

#endif