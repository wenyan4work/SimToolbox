#include "Util/YamlHelper.hpp"

/**
 * @brief interface to all boundary types
 *
 */
class Boundary {
  public:
    virtual ~Boundary(){};

    /**
     * @brief initialize from a yaml node
     *
     * @param config
     */
    virtual void initialize(const YAML::Node &config) = 0;

    virtual void project(const double query[3], double project[3], double delta[3]) const = 0;

    virtual bool check(const double query[3], const double project[3], const double delta[3]) const = 0;
};

class SphereShell : public Boundary {
  public:
    SphereShell() = default;
    SphereShell(double center_[3], double radius_, bool inside_);
    virtual ~SphereShell() = default;
    virtual void initialize(const YAML::Node &config);
    virtual void project(const double query[3], double project[3], double delta[3]) const;
    virtual bool check(const double query[3], const double project[3], const double delta[3]) const;

  private:
    double center[3] = {0, 0, 0};
    double radius = 0;
    bool inside = true; ///< if particles should be confined inside this spherical shell
};

class Wall : public Boundary {
  public:
    Wall() = default;
    Wall(double center_[3], double norm_[3]);
    ~Wall() = default;
    void initialize(const YAML::Node &config);
    virtual void project(const double query[3], double project[3], double delta[3]) const;
    virtual bool check(const double query[3], const double project[3], const double delta[3]) const;

  private:
    double center[3] = {0, 0, 0};
    double norm[3] = {0, 0, 1}; ///< the direction toward the particles
};

class Tube : public Boundary {
  public:
    Tube() = default;
    Tube(double center_[3], double axis_[3], double radius_, bool inside_);
    ~Tube() = default;
    void initialize(const YAML::Node &config);
    virtual void project(const double query[3], double project[3], double delta[3]) const;
    virtual bool check(const double query[3], const double project[3], const double delta[3]) const;

  private:
    double center[3] = {0, 0, 0};
    double axis[3] = {0, 0, 1};
    double radius = 0;
    bool inside = true; ///< if particles should be confined inside this tube
};