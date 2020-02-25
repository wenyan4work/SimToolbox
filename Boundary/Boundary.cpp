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
        printf("1\n");
        return false;
    }
    // 2. |query - project| = |delta|
    Evec3 PQ = Query - Proj;
    if (fabs(PQ.norm() - delta.norm()) > eps) {
        printf("2\n");
        return false;
    }
    // 3. inside outside direction
    double Qnorm = Query.norm();
    Evec3 norm = Proj * (1 / Proj.norm()); // toward sphere outside
    if (inside) {
        norm *= -1; // toward sphere inside
    }
    if (fabs(1 - norm.dot(delta.normalized())) > eps) {
        printf("3\n");
        return false;
    }
    return true;
}

void SphereShell::echo() const {
    printf("------------------------\n");
    printf("Spherical shell boundary\n");
    if (inside)
        printf("particles inside\n");
    else
        printf("particles outside\n");
    printf("radius: %g\n", radius);
    printf("center: %g, %g, %g\n", center[0], center[1], center[2]);
    printf("------------------------\n");
}

/************************************************************
 *
 *     flat wall
 *
 ***********************************************************/
Wall::Wall(double center_[3], double norm_[3]) {
    std::copy(center_, center_ + 3, center);
    std::copy(norm_, norm_ + 3, norm);
    Emap3(norm).normalize();
}

void Wall::initialize(const YAML::Node &config) {
    readConfig(config, VARNAME(center), center, 3, "");
    readConfig(config, VARNAME(norm), norm, 3, "");
    Emap3(norm).normalize();
}

void Wall::project(const double query[3], double project[3], double delta[3]) const {
    Evec3 Query = ECmap3(query);
    Evec3 CQ = Query - ECmap3(center);
    double t = CQ.dot(ECmap3(norm));
    Evec3 Proj = Query - t * ECmap3(norm);
    Evec3 PQ = Query - Proj;
    if (t < 0) {
        PQ *= -1;
    }
    project[0] = Proj[0];
    project[1] = Proj[1];
    project[2] = Proj[2];
    delta[0] = PQ[0];
    delta[1] = PQ[1];
    delta[2] = PQ[2];
}

bool Wall::check(const double query[3], const double project[3], const double delta_[3]) const {
    std::cout << "query: " << ECmap3(query).transpose() << "\t";
    std::cout << "project: " << ECmap3(project).transpose() << "\t";
    std::cout << "delta: " << ECmap3(delta_).transpose() << std::endl;
    Evec3 Query = ECmap3(query);
    Evec3 Proj = ECmap3(project);
    Evec3 delta = ECmap3(delta_);
    // 1. project on plane
    if (fabs((Proj - ECmap3(center)).dot(ECmap3(norm))) > eps) {
        printf("1\n");
        return false;
    }
    // 2. |query - project| = |delta|
    Evec3 PQ = Query - Proj;
    if (fabs(PQ.norm() - delta.norm()) > eps) {
        printf("2\n");
        return false;
    }
    // 3. inside outside direction
    if (fabs(1 - ECmap3(norm).dot(delta.normalized())) > eps) {
        printf("3\n");
        return false;
    }
    return true;
}

void Wall::echo() const {
    printf("------------------------\n");
    printf("flat wall boundary\n");
    printf("center: %g, %g, %g\n", center[0], center[1], center[2]);
    printf("norm:   %g, %g, %g\n", norm[0], norm[1], norm[2]);
    printf("------------------------\n");
}

/************************************************************
 *
 *     cylinderical infinitely long tube
 *
 ***********************************************************/
Tube::Tube(double center_[3], double axis_[3], double radius_, bool inside_) {
    std::copy(center_, center_ + 3, center);
    std::copy(axis_, axis_ + 3, axis);
    radius = radius_;
    inside = inside_;
    Emap3(axis).normalize();
}

void Tube::initialize(const YAML::Node &config) {
    readConfig(config, VARNAME(center), center, 3, "");
    readConfig(config, VARNAME(axis), axis, 3, "");
    readConfig(config, VARNAME(inside), inside, "");
    readConfig(config, VARNAME(radius), radius, "");
    Emap3(axis).normalize();
}

void Tube::project(const double query[3], double project[3], double delta_[3]) const {
    Evec3 Query = ECmap3(query);
    double t = (Query - ECmap3(center)).dot(ECmap3(axis));
    Evec3 ProjAxis = ECmap3(center) + t * ECmap3(axis);
    Evec3 ProjAxisQ = Query - ProjAxis;
    Evec3 Proj = ProjAxis + radius * (ProjAxisQ.normalized());
    Evec3 delta = Query - Proj;
    if (ProjAxisQ.norm() > radius) { // Q is outside the tube
        if (inside) {
            delta *= -1;
        }
    } else { // Q is inside the tube
        if (!inside) {
            delta *= -1;
        }
    }
    project[0] = Proj[0];
    project[1] = Proj[1];
    project[2] = Proj[2];
    delta_[0] = delta[0];
    delta_[1] = delta[1];
    delta_[2] = delta[2];
}

bool Tube::check(const double query[3], const double project[3], const double delta_[3]) const {
    std::cout << "query: " << ECmap3(query).transpose() << "\t";
    std::cout << "project: " << ECmap3(project).transpose() << "\t";
    std::cout << "delta: " << ECmap3(delta_).transpose() << std::endl;
    Evec3 Query = ECmap3(query);
    Evec3 Proj = ECmap3(project);
    Evec3 delta = ECmap3(delta_);
    double t = (Proj - ECmap3(center)).dot(ECmap3(axis));
    Evec3 ProjAxis = ECmap3(center) + t * ECmap3(axis);
    // 1. project on tube
    if (fabs(radius - (Proj - ProjAxis).norm()) > eps) {
        printf("1\n");
        return false;
    }
    // 2. |query - project| = |delta|
    Evec3 PQ = Query - Proj;
    if (fabs(PQ.norm() - delta.norm()) > eps) {
        printf("2\n");
        return false;
    }
    // 3. inside outside direction
    Evec3 norm;
    if (inside) {
        norm = ProjAxis - Proj;
    } else {
        norm = Proj - ProjAxis;
    }
    norm.normalize();
    if (fabs(1 - (delta.normalized()).dot(norm)) > eps) {
        printf("3\n");
        return false;
    }
    return true;
}

void Tube::echo() const {
    printf("------------------------\n");
    printf("Tube boundary\n");
    if (inside)
        printf("particles inside\n");
    else
        printf("particles outside\n");
    printf("radius: %g\n", radius);
    printf("center: %g, %g, %g\n", center[0], center[1], center[2]);
    printf("axis: %g, %g, %g\n", axis[0], axis[1], axis[2]);
    printf("------------------------\n");
}