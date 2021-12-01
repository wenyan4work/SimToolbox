# Guide

This is a step-by-step document of setting up necessary environment, toolchain, and dependence libraries.
Depending on your operating system, you may need to slightly change the steps described in this document.
This guide walks you through the steps on Linux.

# Option 1: Manual installation

Installing `boost` and `eigen` are easy. You can simply copy the headers to some location and it's done.
To install Trilinos, you need `cmake>=3.17`, a working compiler with `c++11` support, a working mpi installation with `mpicxx`, `mpicc`, `mpirun`.
Use a released tag with version `>=13.0`, avoid the master or develop branches.
Then, use this script to run `cmake` and configure it.
Assuming `${HOME}/envs/t13/` is where you plan to install Trilinos.

```bash
cmake \
	-D CMAKE_INSTALL_PREFIX:FILEPATH="${HOME}/envs/t13/" \
	-D CMAKE_INSTALL_LIBDIR=lib \
	-D CMAKE_BUILD_TYPE:STRING="Release" \
	-D CMAKE_CXX_FLAGS:STRING="-O3 -march=native" \
	-D CMAKE_C_FLAGS:STRING="-O3 -march=native" \
	-D TPL_ENABLE_HWLOC:BOOL=OFF \
	-D Trilinos_HIDE_DEPRECATED_CODE:BOOL=ON \
	-D Trilinos_ENABLE_EXPLICIT_INSTANTIATION:BOOL=ON \
	-D Trilinos_ENABLE_THREAD_SAFE:BOOL=ON \
	-D Trilinos_ENABLE_Fortran:BOOL=OFF \
	-D Trilinos_ENABLE_EXAMPLES:BOOL=ON \
	-D Trilinos_ENABLE_TESTS:BOOL=ON \
	-D Trilinos_ENABLE_OpenMP:BOOL=ON \
	-D Trilinos_ENABLE_CXX11:BOOL=ON \
	-D Trilinos_ENABLE_Tpetra:BOOL=ON \
	-D Trilinos_ENABLE_Belos:BOOL=ON \
	-D Trilinos_ENABLE_Ifpack2:BOOL=ON \
	-D Trilinos_ENABLE_Zoltan2:BOOL=ON \
	-D TPL_ENABLE_HWLOC:BOOL=OFF \
	-D TPL_ENABLE_MPI:BOOL=ON \
	-D BUILD_SHARED_LIBS:BOOL=OFF \
	-D TPL_ENABLE_BLAS:BOOL=ON \
	-D TPL_ENABLE_LAPACK:BOOL=ON \
	-D BLAS_LIBRARY_DIRS:FILEPATH="$FLEXIBLAS_ROOT/lib64" \
	-D BLAS_LIBRARY_NAMES:STRING="flexiblas" \
	-D LAPACK_LIBRARY_DIRS:FILEPATH="$FLEXIBLAS_ROOT/lib64" \
	-D LAPACK_LIBRARY_NAMES:STRING="flexiblas" \
	$EXTRA_ARGS \
	$TRILINOS_PATH
```

This example uses FlexiBLAS to support runtime switching between different blas/lapack backends. If you do not need this feature, replace the BLAS/LAPACK lines with the following:

linking Intel mkl:

```bash
	-D TPL_ENABLE_MKL:BOOL=ON \
	-D MKL_INCLUDE_DIRS:STRING="$MKL_INCLUDE_DIRS" \
	-D MKL_LIBRARY_DIRS:FILEPATH="$MKL_LIB_DIRS" \
	-D MKL_LIBRARY_NAMES:STRING="mkl_rt" \
	-D BLAS_LIBRARY_DIRS:FILEPATH="$MKL_LIB_DIRS" \
	-D BLAS_LIBRARY_NAMES:STRING="mkl_rt" \
	-D LAPACK_LIBRARY_DIRS:FILEPATH="$MKL_LIB_DIRS" \
	-D LAPACK_LIBRARY_NAMES:STRING="mkl_rt" \
```

linking OpenBLAS:

```bash
	-D TPL_ENABLE_MKL:BOOL=OFF \
	-D BLA_STATIC:BOOL=ON \
	-D BLA_VENDOR:STRING="OpenBLAS" \
	-D TPL_BLAS_LIBRARIES:PATH="$OPENBLAS_BASE/lib/libopenblas.so" \
	-D TPL_LAPACK_LIBRARIES:PATH="$OPENBLAS_BASE/lib/libopenblas.so" \
```

After cmake successfully finishes without error, just run `make && make install`

# Option 2: Automated installation using spack.

**WARNING** if you want to try this on a cluster or on a managed workstation, ask your administrator before running spack.

**WARNING 2** spack recompiles everything so this option is automated but takes a lot of time (several hours with 8 cores).

`spack >= 0.17` is recommended.

## step 1, download and install spack

```bash
$ git clone -c feature.manyFiles=true https://github.com/spack/spack.git
```

add command line support of spack to your `.bashrc`:

```bash
. /path/to/your/spack/installation/share/spack/setup-env.sh
```

## step 2, install compiler

This step installs a compiler that is independent from your system default compiler, to avoid problems induces by ancient operating system default compilers.

```bash
spack install gcc@11.2.0 # this takes hours
spack load gcc@11.2.0 # this is immediate
spack compiler add # this add gcc@11.2.0 to usable compilers of spack
```

Then you can inspect the file `~/.spack/linux/compilers.yaml` to view all compilers found by spack.
For example, this shows that operating system installed `clang@10` and `gcc@9`, and we installed `gcc@11` by spack.

```yaml
$ cat ~/.spack/linux/compilers.yaml
compilers:
- compiler:
    spec: clang@10.0.0
    paths:
      cc: /usr/bin/clang
      cxx: /usr/bin/clang++
      f77: null
      fc: null
    flags: {}
    operating_system: ubuntu20.04
    target: x86_64
    modules: []
    environment: {}
    extra_rpaths: []
- compiler:
    spec: gcc@9.3.0
    paths:
      cc: /usr/bin/gcc
      cxx: /usr/bin/g++
      f77: /usr/bin/gfortran
      fc: /usr/bin/gfortran
    flags: {}
    operating_system: ubuntu20.04
    target: x86_64
    modules: []
    environment: {}
    extra_rpaths: []
- compiler:
    spec: gcc@11.2.0
    paths:
      cc: /home/wyan/software/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.3.0/gcc-11.2.0-sf5dve2qduhb7ukkr6prxiubaorcwseo/bin/gcc
      cxx: /home/wyan/software/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.3.0/gcc-11.2.0-sf5dve2qduhb7ukkr6prxiubaorcwseo/bin/g++
      f77: /home/wyan/software/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.3.0/gcc-11.2.0-sf5dve2qduhb7ukkr6prxiubaorcwseo/bin/gfortran
      fc: /home/wyan/software/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.3.0/gcc-11.2.0-sf5dve2qduhb7ukkr6prxiubaorcwseo/bin/gfortran
    flags: {}
    operating_system: ubuntu20.04
    target: x86_64
    modules: []
    environment: {}
    extra_rpaths: []
```

## step 3 (optional), specify compiler flags

If you want to enforce a flag for all spack builds, replace the `flags: {}` part with the flags you want.
For example:

```yaml
flags:
  cflags: "-march=native -O3"
  cxxflags: "-march=native -O3"
  fflags: "-march=native -O3"
```

## step 4, create a spack environment
This is similar to a conda environment. 
Go to a folder where you want your spack installation to be, for example, `~/env_spack`.
Go to that folder and create a file called `spack.yaml` with the following content:
```yaml
$ cat ~/env_spack/spack.yaml
# This is a Spack Environment file.
#
# It describes a set of packages to be installed, along with
# configuration settings.
spack:
  # add package specs to the `specs` list
  specs: [
      cmake,
      boost cxxstd=14,
      openmpi,
      openblas threads=openmp,
      eigen@3.4.0
      build_type=Release,
      trilinos@13.2.0+belos+ifpack2+mpi+openmp+rol+tpetra+zoltan+zoltan2+cuda+wrapper
      cuda_arch=75 build_type=Release cxxstd=14 ^openblas threads=openmp,
    ]
  view: /home/wyan/env_spack/view
  concretization: together
  packages:
    all:
      compiler: [gcc@11.2.0]
```

If you do not need cuda in trilinos, remove cuda-related definitions and use this:
```yaml
$ cat ~/env_spack/spack.yaml
# This is a Spack Environment file.
#
# It describes a set of packages to be installed, along with
# configuration settings.
spack:
  # add package specs to the `specs` list
  specs: [
      cmake,
      boost cxxstd=14,
      openmpi,
      openblas threads=openmp,
      eigen@3.4.0
      build_type=Release,
      trilinos@13.2.0+belos+ifpack2+mpi+openmp+rol+tpetra+zoltan+zoltan2 build_type=Release cxxstd=14 ^openblas threads=openmp,
    ]
  view: /home/wyan/env_spack/view
  concretization: together
  packages:
    all:
      compiler: [gcc@11.2.0]
```



Make sure that the location `view` is consistent with your spack environment folder.

Then, let spack figure out all dependencies and install the packages:
```bash
spack env activate -d /home/wyan/env_spack # this is where your spack.yaml file located
spack concretize # this should finish within minutes without errors.
spack install # this will take hours
```

This should finish smoothly. All packages will be installed and symlinked to the folder `/home/wyan/env_spack/view`, where you have a standard linux installtion folder structure, like `include`, `lib`, `share`, etc.

Everytime you need to use these software, the activate/deactivate process is similar to `conda activate`. 
To activate the dependencies:
```bash
spack env activate -d /home/wyan/env_spack # this is where your spack.yaml file located
```
This will automatically setup your `PATH`, `LD_LIBRARY_PATH`, etc.
Then you can compile and run SimToolbox.
After you finishes, deactivate this environment:
```bash
despacktivate
```

