[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f387b1d7c5534099a139ce707fe6de8e)](https://www.codacy.com/app/wenyan4work/SimToolbox?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=wenyan4work/SimToolbox&amp;utm_campaign=Badge_Grade)

# SimToolBox
A toolbox for object-tracking simulation. 
This toolbox contains a collection of useful tools for different tasks involved in a particle-tracking type simulation.

At the base level the code relies on `Trilinos`.
The folder `Trilinos` contains an interface to [the huge C++ project](https://trilinos.github.io/) for distributed linear algebra and some other useful stuff.

The folder `ext` contains several header-only dependency libraries, including `spdlog`, `msgpack-c++`, and `toml11`.

The folder `Constraint` contains the routines for LCP based collision resolution algorithms, supporting both OpenMP and MPI through `Trilinos`.

The folder `MPI` contains mpi communication, particle domain decomposition, and near neighbor detection code. The near neighbor detection part relies on [`ArborX`](https://github.com/arborx/ArborX) and extended with periodic boundary conditions.
This library will be downloaded automatically as part of the building process.

The folder `Sylinder` contains a parallel `SylinderSystem` class for simulating rigid spherocylinders with overdamped dynamics. A geometric constraint optimization method is implemented to resolve collisions. Parallel IO is also supported for each geometric primitive through the XML-based vtk routines. This code is used in my publications:

1. Yan, Wen, Huan Zhang, and Michael J. Shelley. 2019. “Computing Collision Stress in Assemblies of Active Spherocylinders: Applications of a Fast and Generic Geometric Method.” The Journal of Chemical Physics 150(6): 064109.
 
The folder `Util` contains some utilities to facilitate the application, including an interface to linear algebra `Eigen`, a parallel RNG interface to `TRNG`, a timer, etc.

New functions will be continuously added to this toolbox to facilitate quick development of HPC simulation code.
Run the script `Gendoc.sh` to generate html document with doxygen in `doc/html/index.html`.

# Extra dependency libraries

You need to install `boost >=1.71`, `Trilinos >=13.0`, and `eigen` externally to compile this libraries. Other dependencies are already included in this codebase.

On a cluster, `boost` and `eigen` are usually already installed by the administrator.
To install `Trilinos`, you can either do it manually or perform automated installation using `spack`.  

Read `StepByStep.md` for detailed instructions.

**Note** Macs are peculiar and I do not have a mac to test this codebase. If you are using a mac, you will very likely runtime configuration and compilation errors. Please read the document `Deprecated/MacSetup.md` for some outdated but still relevant instructions.
