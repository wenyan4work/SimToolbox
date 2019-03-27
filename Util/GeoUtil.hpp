/**
 * @file GeoUtil.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief a few utility template functions
 * @version 1.0
 * @date 2019-01-07
// TODO: More thorough tests
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef GEOUTIL_HPP_
#define GEOUTIL_HPP_

#include <cmath>

/**
 * @brief find PBC Image of x in range[lb,ub)
 *
 * @tparam Real
 * @param lb
 * @param ub
 * @param x
 * @return Real
 */
template <class Real>
inline void findPBCImage(const Real &lb, const Real &ub, Real &x) {
    const Real L = ub - lb;
    while (x >= ub) {
        x -= L;
    }
    while (x < lb) {
        x += L;
    }
}

/**
 * @brief find PBC Image of x closet to trg
 *
 * step 1: PBC image of trg in range[lb,ub)
 * step 2: find PBC Image of x closet to trg
 * x may be out of range [lb,ub)
 * @tparam Real
 * @param lb
 * @param ub
 * @param x
 * @param trg
 * @return Real
 */
template <class Real>
inline void findPBCImage(const Real &lb, const Real &ub, Real &x, Real &trg) {
    findPBCImage(lb, ub, trg);
    Real dist = x - trg;
    findPBCImage(0.0, ub - lb, dist);
    if (dist > (ub - lb) * 0.5) {
        x = trg + dist - (ub - lb);
    } else {
        x = trg + dist;
    }
}

/**
 * @brief Get a random point (x,y) in a circle uniformly
 *
 * @tparam Real
 * @param radius
 * @param U01a rng U01
 * @param U01b rng U01
 * @param x
 * @param y
 */
template <class Real>
inline void getRandPointInCircle(const Real &radius, const Real &U01a, const Real &U01b, //
                                 Real &x, Real &y) {
    constexpr Real Pi = 3.14159265358979323846;
    double theta = 2 * Pi * U01a;   /* angle is uniform */
    double r = radius * sqrt(U01b); /* radius proportional to sqrt(U), U~U(0,1) */
    x = r * cos(theta);
    y = r * sin(theta);
}

/**
 * @brief Get a random point on unit sphere
 *
 * @tparam Real
 * @param U01a rng U01
 * @param U01b rng U01
 * @param theta
 * @param phi
 */
template <class Real>
inline void getRandPointAngleOnSphere(const Real &U01a, const Real &U01b, Real &theta, Real &phi) {
    constexpr Real Pi = 3.14159265358979323846;
    theta = 2 * Pi * U01a;
    phi = acos(2 * U01b - 1.0);
}

/**
 * @brief Get a random point on unit sphere
 *
 * @tparam Real
 * @param U01a rng U01
 * @param U01b rng U01
 * @param pos
 */
template <class Real>
inline void getRandPointOnSphere(const Real &radius, const Real &U01a, const Real &U01b, Real pos[3]) {
    Real theta, phi;
    getRandPointAngleOnSphere(U01a, U01b, theta, phi);
    pos[0] = cos(theta) * sin(phi) * radius;
    pos[1] = sin(theta) * sin(phi) * radius;
    pos[2] = cos(phi) * radius;
}

#endif