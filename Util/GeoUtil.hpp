/**
 * @file GeoUtil.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief a few utility template functions
 * @version 1.0
 * @date 2019-01-07
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef GEOUTIL_HPP_
#define GEOUTIL_HPP_

//TODO: More thorough tests

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
void findPBCImage(const Real &lb, const Real &ub, Real &x) {
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
void findPBCImage(const Real &lb, const Real &ub, Real &x, Real &trg) {
    findPBCImage(lb, ub, trg);
    x = x - trg;
    findPBCImage(lb, ub, x);
    if (x > 0.5 * (lb + ub)) {
        x = trg + x - (ub - lb);
    } else {
        x = trg + x;
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
void getRandPointInCircle(const Real &radius, const Real &U01a, const Real &U01b, //
                          Real &x, Real &y) {
    constexpr Real Pi = 3.14159265358979323846;
    double theta = 2 * Pi * U01a;   /* angle is uniform */
    double r = radius * sqrt(U01b); /* radius proportional to sqrt(U), U~U(0,1) */
    x = r * cos(theta);
    y = r * sin(theta);
}

#endif