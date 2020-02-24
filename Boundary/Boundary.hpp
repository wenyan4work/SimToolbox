#ifndef BOUNDARY_HPP_
#define BOUNDARY_HPP_

#include "Util/YamlHelper.hpp"

/**
 * @brief interface to all boundary types
 *
 */
class Boundary {
  public:
    Boundary() = default;
    virtual ~Boundary(){};

    /**
     * @brief initialize from a yaml node
     *
     * @param config
     */
    virtual void initialize(const YAML::Node &config) = 0;

    /**
     * @brief find the projection of query point on the boundary
     *
     * @param query
     * @param project
     * @param delta vector of the constraint force. |delta| = |query-project|
     */
    virtual void project(const double query[3], double project[3], double delta[3]) const = 0;

    /**
     * @brief check the correctness of query/project/delta
     *
     * @param query
     * @param project
     * @param delta
     * @return true
     * @return false
     */
    virtual bool check(const double query[3], const double project[3], const double delta[3]) const = 0;

    /**
     * @brief print the configuration of this boundary
     *
     */
    virtual void echo() const = 0;
};

class SphereShell : public Boundary {
  public:
    SphereShell(const YAML::Node &config) { initialize(config); };
    SphereShell(double center_[3], double radius_, bool inside_);
    virtual ~SphereShell() = default;
    virtual void initialize(const YAML::Node &config);
    virtual void project(const double query[3], double project[3], double delta[3]) const;
    virtual bool check(const double query[3], const double project[3], const double delta[3]) const;
    virtual void echo() const;

  private:
    double center[3] = {0, 0, 0};
    double radius = 0;
    bool inside = true; ///< if particles should be confined inside this spherical shell
};

class Wall : public Boundary {
  public:
    Wall(const YAML::Node &config) { initialize(config); };
    Wall(double center_[3], double norm_[3]);
    ~Wall() = default;
    void initialize(const YAML::Node &config);
    virtual void project(const double query[3], double project[3], double delta[3]) const;
    virtual bool check(const double query[3], const double project[3], const double delta[3]) const;
    virtual void echo() const;

  private:
    double center[3] = {0, 0, 0};
    double norm[3] = {0, 0, 1}; ///< the direction toward the particles
};

class Tube : public Boundary {
  public:
    Tube(const YAML::Node &config) { initialize(config); };
    Tube(double center_[3], double axis_[3], double radius_, bool inside_);
    ~Tube() = default;
    void initialize(const YAML::Node &config);
    virtual void project(const double query[3], double project[3], double delta[3]) const;
    virtual bool check(const double query[3], const double project[3], const double delta[3]) const;
    virtual void echo() const;

  private:
    double center[3] = {0, 0, 0};
    double axis[3] = {0, 0, 1};
    double radius = 0;
    bool inside = true; ///< if particles should be confined inside this tube
};

#endif