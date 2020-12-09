/**
 * @file SpecialQuadWeights.hpp
 * @author Bryce Palmer (palme200@msu.edu)
 * @brief
 * @version 0.2
 * @date 2020-12-8
 *
 * @copyright Copyright (c) 2020
 *
 */
#ifndef SPECIALQUADWEIGHTS_HPP
#define SPECIALQUADWEIGHTS_HPP

#include "EigenDef.hpp"
#include "Gauss_Legendre_Nodes_and_Weights.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numeric>

/**
 * @brief class to compute special quadrature between a line source and point target
 * https://github.com/ludvigak/linequad
 * @tparam N: max number of points
 */
template <int N>
class SpecialQuadWeights {
    int nQuads;          ///< actual number of special quadrature points
    double glPoints[N];  ///< Gauss Legendre quadrature points
    double glWeights[N]; ///< Gauss Legendre quadrature weights
    double w1[N];        ///< interpolated quadrature weights for 1/R singularity integrals
    double w3[N];        ///< interpolated quadrature weights for 1/R**3 singularity integrals
    double w5[N];        ///< interpolated quadrature weights for 1/R**5 singularity integrals

    void rsqrt_pow_integrals(const std::complex<double> &z, Evec &I1, Evec &I3, Evec &I5) {
        // rsqrt_pow_integrals(z,N)
        // Recursively compute values of integrals
        // Ip(k) = \int_{-1}^{1} t^{k-1}/|t-z|^p dt
        // for k=0,...,N-1 and z not in [-1,1]
        //
        // Recursions by Anna-Karin Tornberg & Katarina Gustavsson
        // Journal of Computational Physics 215 (2006) 172–196
        //
        // Ludvig af Klinteberg, May 2018

        // clang-format off
        const std::vector<double> coeffsI1{
            0.5,                   -0.125,                0.0625,                -0.0390625,            0.02734375,
            -0.0205078125,         0.01611328125,         -0.013092041015625,    0.0109100341796875,    -0.00927352905273438,
            0.00800895690917969,   -0.00700783729553223,  0.00619924068450928,   -0.00553503632545471,  0.00498153269290924,
            -0.004514514002949,    0.00411617453210056,   -0.00377315998775885,  0.00347527893609367,   -0.00321463301588665,
            0.00298501637189474,   -0.00278149252835647,  0.00260009084172452,   -0.00243758516411674,  0.00229133005426974,
            -0.00215913793575417,  0.00203918582821228,   -0.00192994373027233,  0.00183011905456859,   -0.00173861310184016,
            0.00165448666142854,   -0.00157693259917408,  0.00150525384466617,   -0.00143884558681325,  0.00137718077594982,
            -0.00131979824361858,  0.00126629290941783,   -0.00121630766299344,  0.00116952659903215,   -0.00112566935156845,
            0.00108448632651106,   -0.00104575467199281,  0.00100927485785353,   -0.000974867760426703, 0.000942372168412479,
            -0.000911642641181637, 0.000882547663271585,  -0.000854968048794348, 0.000828795557504725,  -0.000803931690779583,
            0.000780286641050772,  -0.000757778372558923, 0.000736331814844991,  -0.000715878153321519, 0.000696354203685478,
            -0.000677701858943903, 0.00065986759949801,   -0.000642802058131682, 0.000626459632924945,  -0.000610798142101821,
            0.000595778515656694,  -0.000581364519310162, 0.000567522506945634,  -0.000554221198189095, 0.000541431478230886,
            -0.000529126217362002, 0.000517280108018076,  -0.000505869517400031, 0.000494872353978291,  -0.000484267946393042,
            0.000474036933441076,  -0.000464161163994387, 0.000454623605830119,  -0.000445408262468697, 0.000436500097219324,
            -0.000427884963721574, 0.000419549542350374,  -0.000411481281920559, 0.00040366834618789,   -0.000396099564696868};
        const std::vector<double> coeffsI3{
            0.375,              -0.3125,            0.2734375,          -0.24609375,        0.2255859375,
            -0.20947265625,     0.196380615234375,  -0.185470581054688, 0.176197052001953,  -0.168188095092773,
            0.161180257797241,  -0.154981017112732, 0.149445980787277,  -0.144464448094368, 0.139949934091419,
            -0.135833759559318, 0.13206059957156,   -0.128585320635466, 0.125370687619579,  -0.122385671247685,
            0.119604178719328,  -0.117004087877604, 0.114566502713487,  -0.112275172659217, 0.110116034723463,
            -0.108076848895251, 0.106146905164978,  -0.10431678611041,  0.10257817300857,   -0.100923686347141};
        const std::vector<double> coeffsI5{
            0.416666666666667, -0.546875,         0.65625,           -0.751953125,      0.837890625,
            -0.91644287109375, 0.989176432291667, -1.05718231201172, 1.12125396728516,  -1.18198855717977,
            1.23984813690186,  -1.2951985001564,  1.34833484888077,  -1.39949934091419, 1.4488934352994,
            -1.49668679514434, 1.54302384762559,  -1.588028709848,   1.63180894996913,  -1.67445850207059,
            1.71605995553819,  -1.75668637494013, 1.79640276254747,  -1.83526724539105, 1.87333204751768,
            -1.91064429296961, 1.94724667406098,  -1.98317801149901, 2.01847372694282,  -2.05316624412465,
            2.08728533127042,  -2.12085839496272, 2.15391073358552,  -2.18646575692811, 2.21854517730003,
            -2.25016917653786, 2.28135655251205,  -2.31212484812159, 2.3424904652639,   -2.37246876586102,
            2.4020741616914,   -2.4313201945042,  2.46021960766885,  -2.48878441042587, 2.51702593565056,
            -2.54495489191118, 2.57258141049467,  -2.59991508798117, 2.62696502487093,  -2.65373986070135};
        // clang-format on

        double zr = std::real(z);
        double zi = std::imag(z);

        // (t-zr)^2+zi^2 = t^2-2*zr*t+zr^2+zi^2 = t^2 + b*t + c
        double b = -2 * zr;
        double c = zr * zr + zi * zi;
        double d = zi * zi; // d = c - b^2/4;

        double u1 = std::sqrt((1 + zr) * (1 + zr) + zi * zi);
        double u2 = std::sqrt((1 - zr) * (1 - zr) + zi * zi);

        int Ns;

        // Compute I1
        // Evaluate after substitution zr -> -|zr|
        double arg1;
        double arg2 = 1 + std::abs(zr) + std::sqrt((1 + std::abs(zr)) * (1 + std::abs(zr)) + zi * zi);
        bool in_rhomb = 4 * std::abs(zi) < 1 - std::abs(zr);
        if (~in_rhomb) {
            arg1 = -1 + std::abs(zr) + std::sqrt((-1 + std::abs(zr)) * (-1 + std::abs(zr)) + zi * zi);
        } else {
            // Series evaluation needed inside
            // rhombus [-1, i/4, 1, -i/4, -1].
            // Here arg1 has cancellation due to structure
            // -x + sqrt(x^2+b^2)
            Ns = 11;
            arg1 = eval_series(coeffsI1, 1 - std::abs(zr), zi, Ns);
            arg1 = (1 - std::abs(zr)) * arg1;
        }

        I1[0] = std::log(arg2) - std::log(arg1);
        if (nQuads > 1) {
            I1[1] = u2 - u1 - b / 2 * I1[0];
        }
        int s = 1;
        for (int n = 1; n < nQuads - 1; n++) {
            s = -s; // (-1)^(n-1)
            I1[n + 1] = (u2 - s * u1 + (1 - 2 * (n + 1)) * b / 2 * I1[n] - n * c * I1[n - 1]) / (n + 1);
        }

        // Compute I3
        // Series is needed in cones extending around real axis from
        // interval endpoints
        double w = std::min(std::abs(1 + zr), std::abs(1 - zr)); // distance interval-singularity
        bool outside_interval = (abs(zr) > 1);

        zi = std::abs(zi);
        bool in_cone = (zi < 0.6 * w);
        bool use_series = (outside_interval && in_cone);

        if (not use_series) {
            I3[0] = (b + 2) / (2 * d * u2) - (b - 2) / (2 * d * u1);
        } else {
            // Power series for shifted integral
            // pick reasonable number of terms
            if (zi < 0.01 * w) {
                Ns = 4;
            } else if (zi < 0.1 * w) {
                Ns = 10;
            } else if (zi < 0.2 * w) {
                Ns = 15;
            } else { // zi/w < 0.6
                Ns = 30;
            }

            I3[0] =
                std::abs(1 - zr) / ((1 - zr) * (1 - zr) * (1 - zr)) * (-0.5 + eval_series(coeffsI3, (1 - zr), zi, Ns)) -
                std::abs(-1 - zr) / ((-1 - zr) * (-1 - zr) * (-1 - zr)) *
                    (-0.5 + eval_series(coeffsI3, (-1 - zr), zi, Ns));
        }
        if (nQuads > 1) {
            I3[1] = 1.0 / u1 - 1.0 / u2 - b / 2 * I3[0];
        }
        for (int n = 1; n < nQuads - 1; n++) {
            I3[n + 1] = I1[n - 1] - b * I3[n] - c * I3[n - 1];
        }

        // Compute I5
        // Here too use power series for first integral, in cone around real axis
        in_cone = (zi < 0.7 * w);
        use_series = (outside_interval && in_cone);
        if (not use_series) {
            I5[0] = (2 + b) / (6 * d * u2 * u2 * u2) - (-2 + b) / (6 * d * u1 * u1 * u1) + 2 / (3 * d) * I3[0];
        } else {
            // Power series for shifted integral
            if (zi < 0.01 * w) {
                Ns = 4;
            } else if (zi < 0.2 * w) {
                Ns = 10;
            } else if (zi < 0.5 * w) {
                Ns = 24;
            } else if (zi < 0.6 * w) {
                Ns = 35;
            } else { // zi/w < 0.7
                Ns = 50;
            }
            I5[0] = 1 / ((1 - zr) * (1 - zr) * (1 - zr) * std::abs(1 - zr)) *
                        (-0.25 + eval_series(coeffsI5, (1 - zr), zi, Ns)) -
                    1 / ((-1 - zr) * (-1 - zr) * (-1 - zr) * std::abs(-1 - zr)) *
                        (-0.25 + eval_series(coeffsI5, (-1 - zr), zi, Ns));
        }

        if (nQuads > 1) {
            // Second integral computed using shifted version, and then corrected by first
            // integral (which is correctly evaluated using power series)
            // \int_{-1}^1 \frac{t \dif t}{ \pars{ (t-z_r)^2 + z_i^2) }^{5/2}} =
            // \int_{-1-z_r}^{1-z_r} \frac{t \dif t}{ (t^2 + z_i^2)^{5/2} }
            // + z_r \int_{-1}^1 \frac{t \dif t}{ \pars{ (t-z_r)^2 + z_i^2) }^{5/2} }
            // This is analogous to the formula for I3(1), but was overlooked by Tornberg & Gustavsson
            I5[1] = 1 / (3 * u1 * u1 * u1) - 1 / (3 * u2 * u2 * u2) - b / 2 * I5[0];
        }
        for (int n = 1; n < nQuads - 1; n++) {
            I5[n + 1] = I3[n - 1] - b * I5[n] - c * I5[n - 1];
        }
    }

    double eval_series(const std::vector<double> &coeffs, const double x, const double b, const int Ns) {
        // Evaluate power series with terms
        // c(p)*(b/x)^2p

        double f = 0.0;
        double bx2 = b * b / (x * x);
        double bx2p = 1;
        for (int p = 0; p < Ns; p++) {
            bx2p = bx2p * bx2;
            f = f + bx2p * coeffs[p];
        }
        return f;
    }

    Evec dvand(const Evec &alpha, const Evec &b) {
        // x = dvand(alpha, b)
        //
        // Solves transposed system A^T*x = b
        // A is Vandermonde matrix, with nonstandard definition
        // A(i,j) = alpha(j)^i
        //
        // Algorithm by Bjorck & Pereyra
        // Mathematics of Computation, Vol. 24, No. 112 (1970), pp. 893-903
        // https://doi.org/10.2307/2004623

        int n = b.size();
        Evec x = b;

        for (int k = 0; k < n; k++) {
            for (int j = n - 1; j > k + 1; j--) {
                x[j] = (x[j] - x[j - 1]) / (alpha[j] - alpha[j - k]);
            }
        }
        for (int k = n - 2; k > 1; k--) {
            for (int j = k; j < n - 1; j++) {
                x[j] = x[j] - alpha[k] * x[j + 1];
            }
        }
        return x;
    }

    Evec pvand(const Evec &alpha, const Evec &b) {
        // x = pvand(alpha, b)
        //
        // Solves system A*x = b
        // A is Vandermonde matrix, with nonstandard definition
        // A(i,j) = alpha(j)^i
        //
        // Algorithm by Bjorck & Pereyra
        // Mathematics of Computation, Vol. 24, No. 112 (1970), pp. 893-903
        // https://doi.org/10.2307/2004623

        Evec x = b;

        for (int k = 1; k <= nQuads; k++) {
            for (int j = nQuads; j >= k + 1; j--) {
                x[j - 1] = x[j - 1] - alpha[k - 1] * x[j - 2];
            }
        }
        for (int k = nQuads - 1; k >= 1; k--) {
            for (int j = k + 1; j <= nQuads; j++) {
                x[j - 1] = x[j - 1] / (alpha[j - 1] - alpha[j - k - 1]);
            }
            for (int j = k; j <= nQuads - 1; j++) {
                x[j - 1] = x[j - 1] - x[j];
            }
        }
        return x;
    }

  public:
    SpecialQuadWeights(const int numQuadPt) : nQuads(numQuadPt) {
        // fill the Gauss Legendre weights and points
        std::vector<double> s, w;
        Gauss_Legendre_Nodes_and_Weights<double>(nQuads, s, w);
        for (int i = 0; i < nQuads; i++) {
            glWeights[i] = w[i];
            glPoints[i] = s[i];
        }
    }

    void calcWeights(const double lineHalfLength, const double *lineCenterCoord, const double *targetCoord,
                     const double *lineDirection) {
        // construct Eigen vectors
        const Evec3 p = ECmap3(lineDirection);
        const Evec3 center = ECmap3(lineCenterCoord);
        const Evec3 target = ECmap3(targetCoord);
        const Evec3 Rvec = (1.0 / lineHalfLength) * (target.array() - center.array());

        // parallel and perpendicular coordinates (x0,y0)
        double x0 = Rvec.dot(p);
        double y0 = (Rvec.array() - x0 * p.array()).matrix().norm();

        // rod descretization (the method outlined in linequad is unnecessary for straight rods)
        std::complex<double> troot(x0, y0);

        // compute the series integrals
        Evec p1(nQuads);
        Evec p3(nQuads);
        Evec p5(nQuads);
        rsqrt_pow_integrals(troot, p1, p3, p5);

        // Bjorck-Pereyra to solve Vandermonde system
        const ECmap glPointsEvec(glPoints, nQuads);
        const Evec vandermond1 = pvand(glPointsEvec, p1);
        const Evec vandermond3 = pvand(glPointsEvec, p3);
        const Evec vandermond5 = pvand(glPointsEvec, p5);

        // store the data in an eigen vector
        Evec tdist(nQuads);
        for (int i = 0; i < nQuads; i++) {
            tdist[i] = std::abs(glPoints[i] - troot);
        }

        // modified quadrature weights
        Evec w1_ = vandermond1.array() * tdist.array();
        Evec w3_ = vandermond3.array() * tdist.array().pow(3);
        Evec w5_ = vandermond5.array() * tdist.array().pow(5);

        for (int i = 0; i < nQuads; i++) {
            w1[i] = w1_[i];
            w3[i] = w3_[i];
            w5[i] = w5_[i];
        }
    }

    int getSize() const { return nQuads; }
    const double *getGLPoints() const { return glPoints; }
    const double *getGLWeights() const { return glWeights; }
    const double *getWeights1() const { return w1; }
    const double *getWeights3() const { return w3; }
    const double *getWeights5() const { return w5; }

    void print() const {
        printf("---- Gauss-Legendre Special Quadrature n=%d ----\n", nQuads);
        for (int i = 0; i < nQuads; i++) {
            printf("glp %12g, glw %12g, w1 %12g, w3 %12g, w5 %12g\n", glPoints[i], glWeights[i], w1[i], w3[i], w5[i]);
        }
    }
};

#endif
