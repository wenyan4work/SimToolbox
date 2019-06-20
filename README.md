[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f387b1d7c5534099a139ce707fe6de8e)](https://www.codacy.com/app/wenyan4work/SimToolbox?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=wenyan4work/SimToolbox&amp;utm_campaign=Badge_Grade)

# SimToolBox
A toolbox for object-tracking simulation. 
This toolbox contains a collection of useful tools for different tasks involved in a particle-tracking type simulation.

At the base level the code relies on `FDPS` and `Trilinos`.
The folder `FDPS` contains the headers to the `Framework for Developing Particle Simulator` publicly available at `https://github.com/FDPS/FDPS`.
The folder `Trilinos` contains an interface to the huge C++ project for distributed linear algebra and some other stuff.

The folder `Collision` contains the routines for LCP based collision resolution algorithms, supporting both OpenMP and MPI through `Trilinos`.

The folder `Protein`, `Sphere`, and `Sylinder` contains geometric primitives for both collision and Boundary Integral applications. Parallel IO is also supported for each geometric primitive through the XML-based vtk routines.

The folder `Util` contains some utilities to facilitate the application, including an interface to linear algebra `Eigen`, a parallel RNG interface to `TRNG`, a command line parser `cmdparser` (https://github.com/FlorianRappl/CmdParser), a timer, and a Gauss_Legendre table (https://github.com/sivaramambikasaran/Quadrature).

New functions will be continuously added to this toolbox to facilitate quick development of HPC simulation code.

Run the script `Gendoc.sh` to generate html document with doxygen in `doc/html/index.html`.