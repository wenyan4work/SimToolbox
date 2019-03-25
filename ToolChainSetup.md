# Linux 
## MPI compiler
Use whatever mpi compiler comes with your system. `OpenMPI` seems working better than `mpich`. `Intel MPI` also works.

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

## MKL
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
## Trilinos
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
  -D CMAKE_C_FLAGS_RELEASE:STRING="-O3 -march=native" \
  -D CMAKE_CXX_FLAGS_RELEASE:STRING="-O3 -march=native" \
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

## TRNG, Yaml-cpp, msgpack, etc
Use the **SAME** compiler as you used to Trilinos to compile these libraries, to avoid potential ABI issues.


# Mac
I assume you use HomeBrew with default settings. We will setup everything with the llvm compiler from HomeBrew, and link compiled programs to bundled libraries instead of Mac's default c++ runtime.
## MPI compiler 
```bash
brew install llvm
brew install openmpi
```
Then, add the following to `~/.bash_profile` and relaunch the terminal:
```bash
# OpenMPI
export OMPI_MPICC=/usr/local/opt/llvm/bin/clang
export OMPI_MPICXX=/usr/local/opt/llvm/bin/clang++
export OMPI_CFLAGS="-I/usr/local/opt/llvm/include"
export OMPI_CXXFLAGS="-I/usr/local/opt/llvm/include"
export OMPI_LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
```
You should verify the installation:
```bash
~ $ mpicxx --version
clang version 8.0.0 (tags/RELEASE_800/final)
Target: x86_64-apple-darwin18.2.0
Thread model: posix
InstalledDir: /usr/local/opt/llvm/bin
```
You should see the clang version from HomeBrew instead of Mac's default clang.

You should also verify:
```bash
~ $ mpicc --showme
/usr/local/opt/llvm/bin/clang -I/usr/local/Cellar/open-mpi/4.0.0/include -I/usr/local/opt/llvm/include -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -lmpi
~ $ mpicxx --showme
/usr/local/opt/llvm/bin/clang++ -I/usr/local/Cellar/open-mpi/4.0.0/include -I/usr/local/opt/llvm/include -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -lmpi
```

## MKL
Download whatever the latest version from Intel and you should see:
```bash
~ $ echo $MKLROOT
/opt/intel/compilers_and_libraries_2019.2.184/mac/mkl
~ $ ls /opt/intel/lib/
libiomp5.a          libiomp5.dylib      libiomp5_db.dylib   libiompstubs5.a     libiompstubs5.dylib
```
Those dylibs are what we need. 
- **DO NOT** statically link openmp runtime library.
- **DO NOT** link to clang's libomp or gcc's libgomp when using MKL. Program crashes or results are wrong.

## OpenMP
By default, clang's OpenMP libraries are installed like this:
```bash
~ $ ls -ag /usr/local/opt/llvm/lib/libiomp5.dylib
lrwxr-xr-x  1 staff  12 Mar 13 04:53 /usr/local/opt/llvm/lib/libiomp5.dylib -> libomp.dylib
~ $ ls -ag /usr/local/opt/llvm/lib/libgomp.dylib
lrwxr-xr-x  1 staff  12 Mar 13 04:53 /usr/local/opt/llvm/lib/libgomp.dylib -> libomp.dylib
~ $ ls -ag /usr/local/opt/llvm/lib/libomp.dylib
-r--r--r--  1 staff  535328 Mar 21 13:51 /usr/local/opt/llvm/lib/libomp.dylib
```
**Note**
- `libiomp5` is Intel's OpenMP runtime library. 
- `libgomp` is GNU's OpenMP runtime library.
- `libomp` claims (partial) compatibility with `libgomp` through the definition of `__GOMP_` functions in `libgomp` 
- `libomp` claims binary compatibilty with `libiomp5`, since Intel is the major open-source contributor to `libomp`. `libomp` contains a bunch of `__KMP_` functions and `KMP_` environment variables, just like the `libiomp5` does. However, the details of such compatibility is not well documented.
  
There is an internal mechanism in the linking stage of clang to choose which OpenMP runtime library to link to:
- `-fopenmp` or `-fopenmp=libomp` links to clang's default `libomp`.
- `-fopenmp=libiomp5` links to `libiomp5`.
- `-fopenmp=libgomp` links to `libgomp`.

**However**, by default clang searches only its own bundled library directory for those runtime libraries. Therefore, `-fopenmp=libiomp5` and `-fopenmp=libgomp` both still link to `libomp` because they are directly sym-linked to `libomp`.

To force the linkage to the real `libiomp5`, first install Intel MKL, and then replace the sym-link to point to the correct full path to `libiomp5`:
```bash
/usr/local/opt/llvm/lib/libiomp5.dylib -> /opt/intel/lib/libiomp5.dylib
```

Then at compile time, use this:
```bash
CFLAGS = -fopenmp=libiomp5
LDFLAGS = -fopenmp=libiomp5 -L/opt/intel/lib -liomp5 -Wl,-rpath,/opt/intel/lib
```

The `-Wl,-rpath` is necessary on Mac. More about this in the next section.

## RPATH
Traditionally on Mac, the dynamic libraries can be searched through the environment variable `DYLD_LIBRARY_PATH`. However, in new versions of Mac because of the new SIP security system, this environment variable will be purged in many cases.
Here is the official description:
```url
https://developer.apple.com/library/archive/documentation/Security/Conceptual/System_Integrity_Protection_Guide/RuntimeProtections/RuntimeProtections.html#//apple_ref/doc/uid/TP40016462-CH3-SW1
```
```
Spawning children processes of processes restricted by System Integrity Protection, such as by launching a helper process in a bundle with NSTask or calling the exec(2) command, resets the Mach special ports of that child process. Any dynamic linker (dyld) environment variables, such as DYLD_LIBRARY_PATH, are purged when launching protected processes.
```

This means in the following important cases the executable will not be able to find dylibs through `DYLD_LIBRARY_PATH`:
- `make test` to launch test executable files
- launch an executable with a debugger

Now we must embed RPATH to every executable dependent on some dynamic libraries. For numerical computing programs, this usually involves two libraries:
```bash
testOMP $ otool -L ./test.X
./test.X:
	@rpath/libiomp5.dylib (compatibility version 5.0.0, current version 5.0.0)
	@rpath/libmkl_rt.dylib (compatibility version 0.0.0, current version 0.0.0)
	/usr/local/opt/open-mpi/lib/libmpi.40.dylib (compatibility version 61.0.0, current version 61.0.0)
	/usr/local/opt/llvm/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.200.5)
```
The `RPATH` for these two dylibs must be properly embed to the executable, by passing the link line as:
```bash
LIBRARIES = \
-L/opt/intel/lib -liomp5 -Wl,-rpath,/opt/intel/lib \
-L/opt/intel/mkl/lib -lmkl_rt -Wl,-rpath,/opt/intel/mkl/lib
```
**Note** Here MKL is linked with the Single Dynamic Library interface. If you link MKL manually to its static libraries, the rpath is not needed. 

To check the rpath is properly set:
```bash
testOMP $ otool -l ./test.X
./test.X:
Mach header

#  some long output

Load command 17
          cmd LC_RPATH
      cmdsize 32
         path /opt/intel/lib (offset 12)
Load command 18
          cmd LC_RPATH
      cmdsize 32
         path /opt/intel/mkl/lib (offset 12)
Load command 19
          cmd LC_RPATH
      cmdsize 40
         path /usr/local/opt/llvm/lib (offset 12)

# other output

```

**Note**
- If not using `libiomp5`, the link line and rpath at `/opt/intel/lib` can be removed.
- Even if not compiling MPI code, the `mpicc` and `mpicxx` compilers, configured as in the first section, is recommended because the include and link lines are already configured.


## MKL and `libomp`
If using the OpenMP threading layer (instead of the TBB layer) of MKL, officially Intel supports only `libiomp5` on Mac.
This is the reason for the `libiomp5` sym-link hack documented above.

Recently, with `llvm 8.0` and `Intel MKL 2019`, the runtime library `libomp` seems to work fine with MKL. So the sym-link hack seems not necessary anymore. However, if there are any abnormal program behavior, including crash or floating point errors, switching back to `libiomp5` is worth trying.  

## Trilinos
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
- Change `/Users/wyan/local` to the folder you want to install. `$HOME/local` is recommended. Avoid `/usr/local`, because it is difficult to cleanup if you want to upgrade or switch to the debug version

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
  -D CMAKE_INSTALL_PREFIX:FILEPATH="/Users/wyan/local/" \
  -D CMAKE_BUILD_TYPE:STRING=Release \
  -D BUILD_SHARED_LIBS=OFF \
  -D CMAKE_C_FLAGS_RELEASE:STRING="-O3 -march=native" \
  -D CMAKE_CXX_FLAGS_RELEASE:STRING="-O3 -march=native" \
  -D OpenMP_C_FLAGS:STRING="-fopenmp=libiomp5" \
  -D OpenMP_CXX_FLAGS:STRING="-fopenmp=libiomp5" \
  -D OpenMP_C_LIB_NAMES:STRING="iomp5" \
  -D OpenMP_CXX_LIB_NAMES:STRING="iomp5" \
  -D OpenMP_iomp5_LIBRARY="/opt/intel/lib/libiomp5.dylib" \
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
  -D MKL_LIBRARY_DIRS:FILEPATH="$MKLROOT/lib" \
  -D MKL_LIBRARY_NAMES:STRING="mkl_rt" \
  -D BLAS_LIBRARY_DIRS:FILEPATH="$MKLROOT/lib" \
  -D BLAS_LIBRARY_NAMES:STRING="mkl_rt" \
  -D LAPACK_LIBRARY_DIRS:FILEPATH="$MKLROOT/lib" \
  -D LAPACK_LIBRARY_NAMES:STRING="mkl_rt" \
  -D TPL_ENABLE_HWLOC:BOOL=OFF \
$EXTRA_ARGS \
$TRILINOS_PATH

```

## Other libraries
Using HomeBrew, other libraries are straightforward to setup:
```bash
brew install autoconf automake boost cmake eigen fftw libtrng msgpack yaml-cpp
```