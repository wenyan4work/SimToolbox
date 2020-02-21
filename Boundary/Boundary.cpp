#include "Boundary.hpp"
#include "Util/EigenDef.hpp"

#include <limits>
constexpr double eps = std::numeric_limits<double>::epsilon() * 1e4;

/************************************************************
 *
 *     spherical shell
 *
 ***********************************************************/
SphereShell::SphereShell(double center_[3], double radius_, bool inside_) {
    std::copy(center_, center_ + 3, center);
    radius = radius_;
    inside = inside_;
}

void SphereShell::initialize(const YAML::Node &config) {
    readConfig(config, VARNAME(center), center, 3, "");
    readConfig(config, VARNAME(radius), radius, "");
    readConfig(config, VARNAME(inside), inside, "");
}

void SphereShell::project(const double query[3], double project[3], double delta[3]) const {
    Evec3 Query = ECmap3(query) - ECmap3(center); // center to query
    double QueryR = Query.norm();
    Evec3 Proj = radius * (1 / QueryR) * Query; // radius * norm direction of Query
    Evec3 PQ = Query - Proj;
    bool out = QueryR > radius;
    if ((inside && out) || (!inside && !out)) {
        PQ *= -1;
    }
    Proj += ECmap3(center);
    project[0] = Proj[0];
    project[1] = Proj[1];
    project[2] = Proj[2];
    delta[0] = PQ[0];
    delta[1] = PQ[1];
    delta[2] = PQ[2];
}

bool SphereShell::check(const double query[3], const double project[3], const double delta_[3]) const {
    std::cout << "query: " << ECmap3(query).transpose() << "\t";
    std::cout << "project: " << ECmap3(project).transpose() << "\t";
    std::cout << "delta: " << ECmap3(delta_).transpose() << std::endl;
    Evec3 Query = ECmap3(query);
    Evec3 Proj = ECmap3(project);
    Evec3 delta = ECmap3(delta_);
    Query -= ECmap3(center);
    Proj -= ECmap3(center);
    // 1. project on sphere
    if (fabs(Proj.norm() - radius) > eps) {
        return false;
    }
    // 2. |query - project| = |delta|
    Evec3 PQ = Query - Proj;
    if (fabs(PQ.norm() - delta.norm()) > eps) {
        return false;
    }
    // 3. inside outside direction
    double Qnorm = Query.norm();
    Evec3 norm = Proj * (1 / Proj.norm()); // toward sphere outside
    if (inside) {
        norm *= -1; // toward sphere inside
    }
    if (fabs(1 - norm.dot(delta.normalized())) > eps) {
        return false;
    }
    return true;
}

/************************************************************
 *
 *     flat wall
 *
 ***********************************************************/
// Wall::Wall(double center_[3], double norm_[3]) {
//     std::copy(center_, center_ + 3, center);
//     std::copy(norm_, norm_ + 3, norm);
// }

// void Wall::initialize(const YAML::Node &config) {
//     readConfig(config, VARNAME(center), center, 3, "");
//     readConfig(config, VARNAME(norm), norm, 3, "");
// }

// void Wall::project(double query[3], double project[3], double normI[3]) const {}

/************************************************************
 *
 *     cylinderical infinitely long tube
 *
 ***********************************************************/
// Tube::Tube(double center_[3], double axis_[3], double radius_, bool inside_) {
//     std::copy(center_, center_ + 3, center);
//     std::copy(axis_, axis_ + 3, axis);
//     radius = radius_;
//     inside = inside_;
// }

// void Tube::initialize(const YAML::Node &config) {
//     readConfig(config, VARNAME(center), center, 3, "");
//     readConfig(config, VARNAME(axis), axis, 3, "");
//     readConfig(config, VARNAME(inside), inside, "");
// }

// void Tube::project(double query[3], double project[3], double normI[3]) const {}