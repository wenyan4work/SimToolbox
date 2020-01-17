# Step 1 MPI compiler
Use whatever mpi compiler comes with your system. 
Because mpi requires fast communicating hardware to work properly on a cluster, you should use the version offered by your administrator through the module system.
If you have multiple choices, use `OpenMPI` because it seems working better than `mpich`.

`Intel MPI` is developed by Intel based on `mpich`, but the recent versions of `Intel MPI` is not very stable. So use `OpenMPI` unless you run into any issues. 

If using `Intel MPI`, you may need to set these environment variables to use Intel compilers.
```bash
export MPICH_CC="icc"
export MPICH_CXX="icpc"
export MPICH_FC="ifort"
```
The `alltoall` algorithms in `Intel MPI` can be tuned by setting up environment variables. You can skip this. The default values usually work well.
```bash
export I_MPI_ADJUST_ALLTOALL=1 # 1=Bruck's, 2=isend/irecv, 3=pairwise exchange, 4=Plum's
export I_MPI_ADJUST_ALLTOALLV=1 # 1=isend/irecv, 2=Plum's
```

# Step 2 MKL
Before using MKL, source the script to setup the environment:
```bash
source /the/correct/path/to/mkl/bin mklvars.sh intel64 
```
Then you should see `MKLROOT` pointing to the correct absolute path:
```bash
~/ $ ls $MKLROOT
benchmarks  bin  examples  include  interfaces  lib  tools
```

`Intel MKL` has a layered design
- Interface Layer: Function interfaces. You can choose LP64 or ILP64. The default LP64 suffices.
- Core Layer: Core library functions.
- Threading Layer: TBB or OpenMP. Only a few functions are threaded by TBB. Choose OpenMP unless you have troubles with this. Using the OpenMP threading layer depends on a compatible OpenMP runtime library.

To correctly link to `Intel MKL`, start with Intel's link advisor:
`https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor`.

Note that the link line suggested by link advisor only resolves all symbols in the link stage. It doesn't guarantee that the executable runs successfully. Refer to MKL's reference manual, section 'Linking In Detail' for more information.

There are a lot of libraries corresponding to the three layers:
```bash
mkl/ $ ls $MKLROOT/lib/intel64/
locale
libmkl_blacs_intelmpi_ilp64.a
libmkl_blacs_intelmpi_lp64.a
libmkl_blacs_openmpi_ilp64.a
libmkl_blacs_openmpi_lp64.a
libmkl_blacs_sgimpt_ilp64.a
libmkl_blacs_sgimpt_lp64.a
libmkl_blas95_ilp64.a
libmkl_blas95_lp64.a
libmkl_cdft_core.a
libmkl_core.a
libmkl_gf_ilp64.a
libmkl_gf_lp64.a
libmkl_gnu_thread.a
libmkl_intel_ilp64.a
libmkl_intel_lp64.a
libmkl_intel_thread.a
libmkl_lapack95_ilp64.a
libmkl_lapack95_lp64.a
libmkl_scalapack_ilp64.a
libmkl_scalapack_lp64.a
libmkl_sequential.a
libmkl_tbb_thread.a
libmkl_avx2.so
libmkl_avx512_mic.so
libmkl_avx512.so
libmkl_avx.so
libmkl_blacs_intelmpi_ilp64.so
libmkl_blacs_intelmpi_lp64.so
libmkl_blacs_openmpi_ilp64.so
libmkl_blacs_openmpi_lp64.so
libmkl_blacs_sgimpt_ilp64.so
libmkl_blacs_sgimpt_lp64.so
libmkl_cdft_core.so
libmkl_core.so
libmkl_def.so
libmkl_gf_ilp64.so
libmkl_gf_lp64.so
libmkl_gnu_thread.so
libmkl_intel_ilp64.so
libmkl_intel_lp64.so
libmkl_intel_thread.so
libmkl_mc3.so
libmkl_mc.so
libmkl_rt.so
libmkl_scalapack_ilp64.so
libmkl_scalapack_lp64.so
libmkl_sequential.so
libmkl_tbb_thread.so
libmkl_vml_avx2.so
libmkl_vml_avx512_mic.so
libmkl_vml_avx512.so
libmkl_vml_avx.so
libmkl_vml_cmpt.so
libmkl_vml_def.so
libmkl_vml_mc2.so
libmkl_vml_mc3.so
libmkl_vml_mc.so
```

The `libmkl_rt.so` is a special `single dynamic library` where all three layers are bundled into this single dynamic library and choices can be tuned at runtime by changing environment variables.
**Note** You should almost always use the following combination. 

**Short summary of using `mkl_rt`**

If using `gcc`, set the environment variables to
```bash
export MKL_THREADING_LAYER=GNU
export MKL_INTERFACE_LAYER=LP64
```
then compile your code with `-fopenmp` flag in both compiling and linking stages to use GNU OpenMP runtime libgomp.

If using Intel compiler `icc`, set the environment variables to 
```bash
export MKL_THREADING_LAYER=INTEL
export MKL_INTERFACE_LAYER=LP64
```
then compile your code with `-qopenmp` flag in both compiling and linking stages to use Intel OpenMP runtime libiomp5.

In both cases, adding `$MKLROOT/lib/intel64` to your `LD_LIBRARY_PATH`.

**Short summary of using static linking**

Static linking requires the three layers of MKL to be explicitly linked. you can see clearly from the link line below:

`gcc` link line:
```bash
 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl
```

`icc` link line:
```bash
 -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl
```
# Step 3 Trilinos
First go to the Trilinos repo folder and then check out the current latest release 12-14 branch:
```bash
Trilinos $ git checkout trilinos-release-12-14-branch
Already on 'trilinos-release-12-14-branch'
Your branch is up to date with 'origin/trilinos-release-12-14-branch'.
```
Then create a build folder, `cd` to this folder, and then run the following cmake script to setup Trilinos compilation process.

Note:
- For debug build, use `CMAKE_BUILD_TYPE:STRING=Debug`
- Change `../Trilinos` to the relative path to the trilinos git repo folder.
- Change `/mnt/home/wyan/local` to the folder you want to install. `$HOME/local` is recommended.

If using `gcc`, compile Trilinos with the following script.

```bash
#!/bin/bash

# Set this to the root of your Trilinos source directory.
TRILINOS_PATH=../Trilinos

#
# You can invoke this shell script with additional command-line
# arguments.  They will be passed directly to CMake.
#
EXTRA_ARGS=$@

#
# Each invocation of CMake caches the values of build options in a
# CMakeCache.txt file.  If you run CMake again without deleting the
# CMakeCache.txt file, CMake won't notice any build options that have
# changed, because it found their original values in the cache file.
# Deleting the CMakeCache.txt file before invoking CMake will insure
# that CMake learns about any build options you may have changed.
# Experience will teach you when you may omit this step.
#
rm -f CMakeCache.txt

#
# Enable all primary stable Trilinos packages.
#
ccmake  \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="/mnt/home/wyan/local/" \
  -D CMAKE_BUILD_TYPE:STRING=Release \
  -D BUILD_SHARED_LIBS=OFF \
  -D CMAKE_C_FLAGS:STRING="-O3 -march=native" \
  -D CMAKE_CXX_FLAGS:STRING="-O3 -march=native" \
  -D OpenMP_C_FLAGS:STRING="-fopenmp" \
  -D OpenMP_CXX_FLAGS:STRING="-fopenmp" \
  -D Kokkos_ENABLE_HWLOC:BOOL=OFF \
  -D Kokkos_ENABLE_OPENMP:BOOL=ON \
  -D Trilinos_ENABLE_EXPLICIT_INSTANTIATION=ON \
  -D Trilinos_ENABLE_Fortran:BOOL=OFF \
  -D Trilinos_ENABLE_TESTS:BOOL=ON \
  -D Trilinos_ENABLE_EXAMPLES:BOOL=OFF \
  -D Trilinos_ENABLE_OpenMP:BOOL=ON \
  -D Trilinos_ENABLE_CXX11:BOOL=ON \
  -D Trilinos_ENABLE_Kokkos:BOOL=ON \
  -D Trilinos_ENABLE_Tpetra:BOOL=ON \
  -D Trilinos_ENABLE_Belos:BOOL=ON \
  -D Trilinos_ENABLE_Ifpack2:BOOL=ON \
  -D Trilinos_ENABLE_ROL:BOOL=ON \
  -D TPL_ENABLE_MPI:BOOL=ON \
  -D TPL_ENABLE_MKL:BOOL=ON \
  -D MKL_INCLUDE_DIRS:STRING="$MKLROOT/include" \
  -D MKL_LIBRARY_DIRS:FILEPATH="$MKLROOT/lib/intel64" \
  -D MKL_LIBRARY_NAMES:STRING="mkl_rt" \
  -D BLAS_LIBRARY_DIRS:FILEPATH="$MKLROOT/lib/intel64" \
  -D BLAS_LIBRARY_NAMES:STRING="mkl_rt" \
  -D LAPACK_LIBRARY_DIRS:FILEPATH="$MKLROOT/lib/intel64" \
  -D LAPACK_LIBRARY_NAMES:STRING="mkl_rt" \
  -D TpetraCore_Threaded_MKL:BOOL=ON \
  -D TPL_ENABLE_HWLOC:BOOL=OFF \
$EXTRA_ARGS \
$TRILINOS_PATH
```

If using `icc`, the script is mostly the same. Just change:
```bash
  -D CMAKE_C_FLAGS_RELEASE:STRING="-O3 -xHost" \
  -D CMAKE_CXX_FLAGS_RELEASE:STRING="-O3 -xHost" \
  -D OpenMP_C_FLAGS:STRING="-qopenmp" \
  -D OpenMP_CXX_FLAGS:STRING="-qopenmp" \
```

If running Trilinos on several machines with different architecture, you may need to change the C/CXX_FLAGS to adapt to your environment.
For example, this generates two versions with `avx2` and `avx512` support in a single binary library, using `icc`.
```bash
  -D CMAKE_C_FLAGS_RELEASE:STRING="-O3 -xcore-avx2 -axcore-avx512" \
  -D CMAKE_CXX_FLAGS_RELEASE:STRING="-O3 -xcore-avx2 -axcore-avx512" \
```
**Note** 
`-xcore-avx2 -axcore-avx512` doesn't guarantee the code becomes faster. It may actually slow down the code because the runtime branching and much larger binary size. Use the flags wisely based on your own code. In general, `-xcore-avx2` already gives decent performance, even on machines with `avx512` capability.

# Step 4 TRNG, Yaml-cpp and msgpack
Use the **SAME** compiler as you used to Trilinos to compile these libraries, to avoid potential ABI issues.

