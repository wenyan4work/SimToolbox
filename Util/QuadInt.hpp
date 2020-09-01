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
#include <cstdlib>

/**
 * @brief class to compute Chebyshev-Gauss Quadrature
 * https://mathworld.wolfram.com/Chebyshev-GaussQuadrature.html
 * @tparam N: max number of points
 */
template <int N>
class QuadInt {
    int npts; // actual number of points
    char choice;
    double points[N];
    double weights[N];

    void calcChebQuad() {
        /* Python code:
         * Dkn=np.zeros((pCheb+1,pCheb+1))
         for k in range(pCheb+1):
         for n in range(pCheb+1):
         Dkn[k,n]=np.cos(k*n/pCheb*np.pi)*2.0/pCheb
         if(n==0 or n==pCheb):
         Dkn[k,n]=np.cos(k*n/pCheb*np.pi)*1.0/pCheb
         dvec=np.zeros(pCheb+1)
         for i in range(pCheb+1):
         if(i%2==1):
         dvec[i]=0
         else:
         dvec[i]=2/(1.0-i**2)
         dvec[0]=1
         weightCC=np.dot(Dkn.transpose(),dvec)
         *
         * */
        const int chebN = npts - 1;
        double *Dkn = new double[(chebN + 1) * (chebN + 1)];
        for (int k = 0; k < chebN + 1; k++) {
            int n = 0;
            Dkn[k * (chebN + 1) + n] = cos(k * n * M_PI / chebN) / chebN;
            for (n = 1; n < chebN; n++) {
                Dkn[k * (chebN + 1) + n] = cos(k * n * M_PI / chebN) * 2 / chebN;
            }
            n = chebN;
            Dkn[k * (chebN + 1) + n] = cos(k * n * M_PI / chebN) / chebN;
        }
        double *dvec = new double[chebN + 1];
        for (int i = 0; i < chebN + 1; i++) {
            dvec[i] = i % 2 == 1 ? 0 : 2 / (1.0 - static_cast<double>(i * i));
        }
        dvec[0] = 1;
        for (int i = 0; i < chebN + 1; i++) {
            double temp = 0;
            for (int j = 0; j < chebN + 1; j++) {
                temp += Dkn[j * (chebN + 1) + i] * dvec[j]; // not optimal layout for speed.
            }
            weights[i] = temp;
            points[i] = -cos(i * M_PI / chebN);
        }

        delete[] Dkn;
        delete[] dvec;
    };

  public:
    QuadInt() : QuadInt(2) {} // default to minimal order

    QuadInt(int n, char quad = 'g') : npts(n), choice(quad) {
        if (n > N) {
            printf("npts larger than max N.\n");
            std::exit(1);
        }
        // initialize points and weights
        if (quad != 'g')
            calcChebQuad();
        else {
            std::vector<double> s, w;
            Gauss_Legendre_Nodes_and_Weights<double>(npts, s, w);
            for (int i = 0; i < npts; i++) {
                weights[i] = w[i];
                points[i] = s[i];
            }
        }
    }

    // copy constructor
    QuadInt(const QuadInt &other) {
        npts = other.npts;
        choice = other.choice;
        for (int i = 0; i < npts; i++) {
            points[i] = other.points[i];
            weights[i] = other.weights[i];
        }
    }
    QuadInt &operator=(const QuadInt &other) {
        npts = other.npts;
        choice = other.choice;
        for (int i = 0; i < npts; i++) {
            points[i] = other.points[i];
            weights[i] = other.weights[i];
        }
    }

    int getSize() const { return npts; }
    int getMaxSize() const { return N; }
    const double *getPoints() const { return points; }
    const double *getWeights() const { return weights; }

    void print() const {
        if (choice == 'g') {
            printf("----Gauss-Legendre Quadrature n=%d ----\n", npts);
        } else {
            printf("----Clenshaw-Curtis Quadrature n=%d----\n", npts);
        }
        for (int i = 0; i < npts; i++) {
            printf("s %12g, w %12g\n", points[i], weights[i]);
        }
    }

    template <class Func>
    double integrate(Func f) const {
        double fval[N];
        for (int i = 0; i < npts; i++) {
            fval[i] = f(points[i]);
        }
        return intSamples(fval);
    }

    template <class Func>
    double integrateS(Func f) const {
        double fval[N];
        for (int i = 0; i < npts; i++) {
            fval[i] = f(points[i]);
        }
        return intSSamples(fval);
    }

    double intSamples(const double *f) const {
        double sum = 0;
        for (int i = 0; i < npts; i++) {
            sum += f[i] * weights[i];
        }
        return sum;
    }

    double intSSamples(const double *f) const {
        double sum = 0;
        for (int i = 0; i < npts; i++) {
            sum += f[i] * points[i] * weights[i];
        }
        return sum;
    }
};

#endif
