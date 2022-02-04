## Definition

Each particle's total force, torque, velocity, and angular velocity are defined in the `Particle` class as these members:

```cpp
double vel[6] = {vx,vy,vz,wx,wy,wz};
double force[6] = {fx,fy,fz,tx,ty,tz};
```

The velocity and omega contain the following components:

```cpp
vel = velConU + velConB + velNonCon + velBrown
```

The force and torque contain the same components except the Brownian ones, because Brownian motion is calculated directly as displacements without computing forces.

```cpp
force = forceConU + forceConB + forceNonCon
```

Here, `ConU` is unilateral constraints (collisions), `ConB` is bilateral constraints (springs), `Brown` is Brownian motion, `NonCon` is all Non-Brownian components before constraint solver is supplied.

Each step contains the following steps:

1. `velBrown` is calculated.
2. `velNonCon` and `forceNonCon` are calculated.
3. `velBrown+velNonCon` is passed to the constraint solver, and `velConB` and `velConU` are calculated. `forceConB` and `forceConU` are simultaneously calculated by the constraint solver.
