/**
 * @file Constraint.hpp
 * @author Bryce Palmer (brycepalmer96@gmail.com)
 * @brief
 * @version 0.1
 * @date 8/23/2022
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef CONSTRAINT_HPP_
#define CONSTRAINT_HPP_

#include <vector>
#include <memory>
#include <limits>
#include <iostream>

/**
 * @brief abstract constraint object
 *
 * This parent constraint object (for now) controls how a constraint 
 * Each constraint has some number of unknown DOF, which must be solved for
 * 
 * Constraints can (currently) have the following types:
 *   id=0: collision              | 3 constrained DOF | prevents penetration and enforces tangency of colliding surfaces
 *   id=1: no penetration         | 1 constrained DOF | prevents penetration
 *   id=2: hookean spring         | 3 constrained DOF | resists relative translational motion between two points
 *   id=3: angular hookean spring | 3 constrained DOF | resists relative rotational motion between two vectors
 *   id=4: ball and socket        | 3 constrained DOF | prevents the separation of two points
 * 
 */
struct Constraint {
  private:
    int id = -1;     ///< identifier specifying the type of constraint this is 
    int numDOF = -1; ///< number of constrained degrees of freedom 
  public:
    virtual int getID() const {return id;}
    virtual int getNumDOF() const {return numDOF;}
    virtual void projectDOF(std::vector<double> &gammas) const = 0;
    virtual void projectValues(const std::vector<double> &gammas, 
                                     std::vector<double> &values) const = 0;
};

struct Collision: Constraint {
  private:
    int id = 0;     ///< identifier specifying the type of constraint this is 
    int numDOF = 3; ///< number of constrained degrees of freedom 
  public:
    int getID() const {return id;}
    int getNumDOF() const {return numDOF;}
    void projectDOF(std::vector<double> &gammas) const {
      // gammas[0]: no-penetration
      // gammas[1]: tangent constraint 1
      // gammas[2]: tanent constraint 2
      if (gammas[0] < 0.0) {
        gammas[0] = 0.0;
      }
    }
    void projectValues(const std::vector<double> &gammas, 
                             std::vector<double> &values) const {
      // values[0]: separation distance 
      //            unilateral >= 0
      // values[1]: change in angle between normal and tangent 1
      //            bilateral = 0
      // values[2]: change in angle between normal and tangent 1
      //            bilateral = 0
      const double eps = std::numeric_limits<double>::epsilon() * 100;
      if (gammas[0] < 0.0 + eps) {
        values[0] = std::min(values[0], 0.0);
      }
      if (values[0] < 0.0 + eps) {
        values[1] = 0.0;
        values[2] = 0.0;
      }
    }
};

struct NoPenetration: Constraint {
  private:
    int id = 1;     ///< identifier specifying the type of constraint this is 
    int numDOF = 1; ///< number of constrained degrees of freedom 
  public:
    int getID() const {return id;}
    int getNumDOF() const {return numDOF;}
    void projectDOF(std::vector<double> &gammas) const {
      // gammas[0]: no-penetration
      if (gammas[0] < 0.0) {
        gammas[0] = 0.0;
      }
    }
    void projectValues(const std::vector<double> &gammas, 
                             std::vector<double> &values) const {
      // values[0]: separation distance 
      //            unilateral >= 0
      const double eps = std::numeric_limits<double>::epsilon() * 100;
      if (gammas[0] < 0.0 + eps) {
        values[0] = std::min(values[0], 0.0);
      }
    }
};

struct Spring: Constraint {
  private:
    int id = 2;     ///< identifier specifying the type of constraint this is 
    int numDOF = 3; ///< number of constrained degrees of freedom 
  public:
    int getID() const {return id;}
    int getNumDOF() const {return numDOF;}
    void projectDOF(std::vector<double> &gammas) const {
      // gammas[0]: spring force in the x-direction
      // gammas[1]: spring force in the y-direction
      // gammas[2]: spring force in the z-direction
      // these are unconstrained
    }
    void projectValues(const std::vector<double> &gammas, 
                             std::vector<double> &values) const {
      // values[0]: violation of spring constraint in the x-direction
      //            bilateral = 0
      // values[1]: violation of spring constraint in the y-direction
      //            bilateral = 0
      // values[2]: violation of spring constraint in the z-direction
      //            bilateral = 0
    }
};

struct AngularSpring: Constraint {
  private:
    int id = 3;     ///< identifier specifying the type of constraint this is 
    int numDOF = 3; ///< number of constrained degrees of freedom 
  public:
    int getID() const {return id;}
    int getNumDOF() const {return numDOF;}
    void projectDOF(std::vector<double> &gammas) const {
      // gammas[0]: spring torque about the x-axis
      // gammas[1]: spring torque about the y-axis
      // gammas[2]: spring torque about the z-axis
      // these are unconstrained
    }
    void projectValues(const std::vector<double> &gammas, 
                             std::vector<double> &values) const {
      // values[0]: violation of spring constraint about the x-axis
      //            bilateral = 0
      // values[1]: violation of spring constraint about the y-axis
      //            bilateral = 0
      // values[2]: violation of spring constraint about the z-axis
      //            bilateral = 0
    }
};

struct Pivot: Constraint {
  private:
    int id = 4;     ///< identifier specifying the type of constraint this is 
    int numDOF = 3; ///< number of constrained degrees of freedom 
  public:
    int getID() const {return id;}
    int getNumDOF() const {return numDOF;}
    void projectDOF(std::vector<double> &gammas) const {
      // gammas[0]: constraint force in the x-direction
      // gammas[1]: constraint force in the y-direction
      // gammas[2]: constraint force in the z-direction
      // these are unconstrained
    }
    void projectValues(const std::vector<double> &gammas, 
                             std::vector<double> &values) const {
      // values[0]: separation distance in the x-direction
      //            bilateral = 0
      // values[1]: separation distance in the y-direction
      //            bilateral = 0
      // values[2]: separation distance in the z-direction
      //            bilateral = 0
    }
};

// Note, to maintain polymorphism between base and derived classes, we must use unique pointers
// To add new objects, use
//    std::unique_ptr<Collision> collisionCon = std::make_unique<Collision>();
//    constraintQue.push_back(std::move(collisionCon)); 
using ConstraintQue = std::vector<std::unique_ptr<Constraint>>; ///< a vect contains constraints collected by one thread
using ConstraintPool = std::vector<ConstraintQue>;              ///< a pool contains queues on different threads

#endif