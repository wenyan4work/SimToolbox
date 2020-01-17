# 0
This is a step-by-step document of setting up necessary environment, toolchain, and dependence libraries.
Depending on your operating system, you may need to slightly change the steps described in this document.
The steps described here are used to setup things on Ubuntu 18.04 of WSL.
If you do not know what WSL is, just use the same steps on Ubunto 18.04 Linux.

This automated `Environment` at `https://github.com/wenyan4work/Environment.git` will try to install the following software automatically for you. 
Use it with caution so your own files are not overwritted during the installation process. 

# 1 Compilers and building tools
```
sudo apt install build-essential automake cmake cmake-curses-gui
```
This will install `gcc`, `automake`, and `cmake`.
If you are on a cluster, make sure you have a working compiler supporting `c++14` and `OpenMP`, and make sure the version of `cmake` is >=3.10.

On Ubuntu 18.04, after the above step you should have working compilers and building tools.
However, the openmpi shipped with Ubuntu 18.04 has a bug that Trilinos/Zoltan does not work properly. You have to compile `openmpi` from source your self.
Download the latest version of `openmpi` from their official website.
Or you could use mpich by 
```
sudo apt install libmpich-dev
```

Check them out as this:
```
wyan@DESKTOP-U6JNAI0:~$ cmake --version
cmake version 3.10.2

CMake suite maintained and supported by Kitware (kitware.com/cmake).
wyan@DESKTOP-U6JNAI0:~$ g++ --version
g++ (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
For `openmpi`:
```
wyan@DESKTOP-U6JNAI0:~$ mpicc -showme
gcc -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
wyan@DESKTOP-U6JNAI0:~$ mpicxx -showme
g++ -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi
wyan@DESKTOP-U6JNAI0:~$ mpicxx --version
g++ (Ubuntu 7.4.0-1ubuntu1~18.04.1) 7.4.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

For `mpich`:
```
wyan@DESKTOP-U6JNAI0:~$ mpicc -v
mpicc for MPICH version 3.3a2
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-linux-gnu/7/lto-wrapper
OFFLOAD_TARGET_NAMES=nvptx-none
OFFLOAD_TARGET_DEFAULT=1
Target: x86_64-linux-gnu
Configured with: ../src/configure -v --with-pkgversion='Ubuntu 7.4.0-1ubuntu1~18.04.1' --with-bugurl=file:///usr/share/doc/gcc-7/README.Bugs --enable-languages=c,ada,c++,go,brig,d,fortran,objc,obj-c++ --prefix=/usr --with-gcc-major-version-only --program-suffix=-7 --program-prefix=x86_64-linux-gnu- --enable-shared --enable-linker-build-id --libexecdir=/usr/lib --without-included-gettext --enable-threads=posix --libdir=/usr/lib --enable-nls --with-sysroot=/ --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --with-default-libstdcxx-abi=new --enable-gnu-unique-object --disable-vtable-verify --enable-libmpx --enable-plugin --enable-default-pie --with-system-zlib --with-target-system-zlib --enable-objc-gc=auto --enable-multiarch --disable-werror --with-arch-32=i686 --with-abi=m64 --with-multilib-list=m32,m64,mx32 --enable-multilib --with-tune=generic --enable-offload-targets=nvptx-none --without-cuda-driver --enable-checking=release --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu
Thread model: posix
gcc version 7.4.0 (Ubuntu 7.4.0-1ubuntu1~18.04.1)
wyan@DESKTOP-U6JNAI0:~$ mpicxx -v
mpicxx for MPICH version 3.3a2
Using built-in specs.
COLLECT_GCC=g++
COLLECT_LTO_WRAPPER=/usr/lib/gcc/x86_64-linux-gnu/7/lto-wrapper
OFFLOAD_TARGET_NAMES=nvptx-none
OFFLOAD_TARGET_DEFAULT=1
Target: x86_64-linux-gnu
Configured with: ../src/configure -v --with-pkgversion='Ubuntu 7.4.0-1ubuntu1~18.04.1' --with-bugurl=file:///usr/share/doc/gcc-7/README.Bugs --enable-languages=c,ada,c++,go,brig,d,fortran,objc,obj-c++ --prefix=/usr --with-gcc-major-version-only --program-suffix=-7 --program-prefix=x86_64-linux-gnu- --enable-shared --enable-linker-build-id --libexecdir=/usr/lib --without-included-gettext --enable-threads=posix --libdir=/usr/lib --enable-nls --with-sysroot=/ --enable-clocale=gnu --enable-libstdcxx-debug --enable-libstdcxx-time=yes --with-default-libstdcxx-abi=new --enable-gnu-unique-object --disable-vtable-verify --enable-libmpx --enable-plugin --enable-default-pie --with-system-zlib --with-target-system-zlib --enable-objc-gc=auto --enable-multiarch --disable-werror --with-arch-32=i686 --with-abi=m64 --with-multilib-list=m32,m64,mx32 --enable-multilib --with-tune=generic --enable-offload-targets=nvptx-none --without-cuda-driver --enable-checking=release --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu
Thread model: posix
gcc version 7.4.0 (Ubuntu 7.4.0-1ubuntu1~18.04.1)
```

**Warning** 
After this point the root privilege should never be used. **Never** install anything outside your home folder.

# 2 Dependence libraries
Create two folders in your home folder:
```
mkdir software 
mkdir local
```

The folder `software` is used for source files and compilation files. 
The folder `local` is used for installation files. 

## 2.0 Intel MKL
First, download the MKL files from Intel website because this is the only option for fully threaded and optimized BLAS & LAPACK3. 
The Linux version works for both Linux and WSL.
The current latest version is 2019.4.243. Put the zip file into the folder `MKL`:
```
cd software
mkdir MKL 
cd MKL
tar xvf ./l_mkl_2019.4.243.tgz
cd ./l_mkl_2019.4.243/
bash ./install.sh
```
Then follow the installer's instructions to install it to the default location, which should be your `${HOME}/intel` folder.
Make sure you choose `ALL` components during the installation.

After installation, you should see those files:
```
wyan@DESKTOP-U6JNAI0:~/software/MKL/l_mkl_2019.4.243$ ls ~/intel/
bin                           compilers_and_libraries_2019.4.243  intel_sdp_products.db  parallel_studio_xe_2019        tbb
compilers_and_libraries       conda_channel                       lib                    parallel_studio_xe_2019.4.070
compilers_and_libraries_2019  documentation_2019                  mkl                    samples_2019
```

Modify the ~/.bashrc file to setup the necessary MKL environment:
```
echo "source ~/intel/bin/compilervars.sh intel64" >> ~/.bashrc
echo "export MKL_INTERFACE_LAYER=GNU,LP64" >> ~/.bashrc
echo "export MKL_THREADING_LAYER=GNU" >> ~/.bashrc
```
This is necessary because we are using gcc compilers with MKL. If you are using intel compilers, ignore the two lines with "MKL_INTERFACE_LAYER" and "MKL_THREADING_LAYER".
For details, read the MKL document:
```
https://software.intel.com/en-us/mkl-linux-developer-guide-dynamically-selecting-the-interface-and-threading-layer
```

After installation, open a new terminal to load the correct bash environments and verify:
```
wyan@DESKTOP-U6JNAI0:~$ env | grep MKL
MKL_INTERFACE_LAYER=GNU,LP64
MKLROOT=/home/wyan/intel/compilers_and_libraries_2019.4.243/linux/mkl
MKL_THREADING_LAYE=GNU
```

## 2.1 TRNG
As of late Aug 2019, TRNG has been updated to a cmake build system. This document is updated accordingly:

Create a folder and pull `trng`:
```
mkdir ~/software/TRNG && cd ~/software/TRNG
git clone https://github.com/rabauke/trng4.git
```

Create a cmake build script `~/software/TRNG/do-configure-TRNG4.sh` as the following:
```bash
#!/bin/bash

SOURCE_PATH=../trng4

EXTRA_ARGS=$@

rm -f CMakeCache.txt

cmake  \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="$HOME/local/" \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
  -D CMAKE_C_COMPILER:STRING="mpicc" \
  -D CMAKE_CXX_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
  -D CMAKE_C_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
$EXTRA_ARGS \
$SOURCE_PATH
```



Then configure with cmake and compile it:
``` bash
cd ~/software/TRNG/build
bash ../do-configure-TRNG4.sh
make
```
Note: if you use `make -jN` for parallel compiling, you may see errors because the cmake file is not fully correct.
The compiled library is not affected and works fine.
After compilation, test it:

```bash
wyan@ccblin014:~/software/TRNG/build$ ./examples/time 
                                            10^6 random numbers per second
generator                       [min,max] [0,1]     [0,1)     (0,1]     (0,1)     canonical
=============================================================================================
trng::lcg64                     845.2     845.711   316.958   408.941   409.32    409.28    
trng::lcg64_shift               777.371   333.497   334.774   409.131   409.32    409.85    
trng::mrg2                      189.657   183.35    182.796   179.986   184.584   178.563   
trng::mrg3                      164.974   158.232   156.608   157.488   156.409   157.042   
trng::mrg3s                     163.588   158.11    160.697   161.234   163.264   160.977   
trng::mrg4                      176.135   121.85    122.798   172.329   170.196   170.795   
trng::mrg5                      186.648   195.356   195.302   195.034   195.685   195.266   
trng::mrg5s                     113.785   116.477   116.741   103.928   112.579   114.281   
trng::yarn2                     155.607   153.163   152.159   153.28    153.4     152.746   
trng::yarn3                     142.407   140.198   140.157   140.062   140.066   140.169   
trng::yarn3s                    119.172   112.93    113.354   113.789   113.603   113.639   
trng::yarn4                     129.869   122.226   121.494   108.032   120.007   119.515   
trng::yarn5                     112.872   106.347   115.468   113.89    114.416   106.162   
trng::yarn5s                    90.579    91.135    91.0583   90.1768   91.1176   91.2266   
trng::mt19937                   373.126   250.743   250.395   289.822   289.322   289.717   
trng::mt19937_64                318.65    257.363   209.726   280.148   286.296   285.944   
trng::lagfib2xor_19937_64       1071.21   335.022   334.467   408.981   409.66    407.996   
trng::lagfib4xor_19937_64       699.371   333.417   333.344   409.39    407.986   409.52    
trng::lagfib2plus_19937_64      1074.5    334.594   335.109   408.991   409.68    409.57    
trng::lagfib4plus_19937_64      696.41    332.189   333.47    409.131   409.43    409.131   
std::minstd_rand0               260.698   256.227   256.121   253.647   253.8     253.463   
std::minstd_rand                260.257   255.906   255.973   253.781   253.835   253.67    
std::mt19937                    233.305   208.594   208.247   204.858   203.733   204.89    
std::mt19937_64                 233.845   202.604   203.136   201.134   200.973   201.826   
std::ranlux24_base              203.05    189.045   186.723   192.569   187.635   187.191   
std::ranlux48_base              246.3     237.54    227.475   233.93    234.725   235.374   
std::ranlux24                   20.1553   20.1358   19.7287   19.2087   20.2037   19.642    
std::ranlux48                   7.21006   6.94696   6.94541   6.95043   6.48851   6.67138   
std::knuth_b                    66.3966   64.3456   64.4135   64.355    64.4695   64.3562   
boost::minstd_rand              253.582   
boost::ecuyer1988               141.231   
boost::kreutzer1986             214.856   
boost::hellekalek1995           7.88665   
boost::mt11213b                 484.163   
boost::mt19937                  487.214   
boost::lagged_fibonacci607      173.818   
boost::lagged_fibonacci1279     173.279   
boost::lagged_fibonacci2281     173.875   
boost::lagged_fibonacci3217     173.209   
boost::lagged_fibonacci4423     173.489   
boost::lagged_fibonacci9689     172.297   
boost::lagged_fibonacci19937    173.444   
boost::lagged_fibonacci23209    172.138   
boost::lagged_fibonacci44497    172.737   
```
Note 1: if the boost headers are not found, you will not see the last several lines using boost::random. This does not affect the functionality of the compiled library.

Note 2: if you are using `boost`>=1.7.0, you may see errors during the cmake or compiling phase. In this case, add two more switchs during the cmake phase to disable boost. `boost` is only used in the testing routines and does not affect the functionality of the `trng` library. The script looks like this:

```bash
#!/bin/bash

SOURCE_PATH=../trng4

EXTRA_ARGS=$@

rm -f CMakeCache.txt

cmake  \
        -D Boost_NO_BOOST_CMAKE=ON \
        -D Boost_NO_SYSTEM_PATHS=ON \
        -D CMAKE_INSTALL_PREFIX:FILEPATH="$HOME/local/" \
        -D CMAKE_BUILD_TYPE:STRING="Release" \
        -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
        -D CMAKE_C_COMPILER:STRING="mpicc" \
        -D CMAKE_CXX_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
        -D CMAKE_C_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
$EXTRA_ARGS \
$SOURCE_PATH
```

Finally, install it.
```
make install
```

## 2.2 YamlCpp
This is a library used to parse `*.yaml` files in `c++`.

```
mkdir ~/software/YamlCpp && cd ~/software/YamlCpp
git clone https://github.com/jbeder/yaml-cpp.git
mkdir build
```
Use `cmake` to configure it. Create a file named `do-configure.sh` under the folder `YamlCpp`:
```
#!/bin/bash

SOURCE_PATH=../yaml-cpp

rm -f CMakeCache.txt

cmake  \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="${HOME}/local/" \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
  -D CMAKE_C_COMPILER:STRING="mpicc" \
  -D CMAKE_CXX_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
  -D CMAKE_C_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
$EXTRA_ARGS \
$SOURCE_PATH
```
Then go to the folder `build` you just created:
```
cd build
bash ../do-configure.sh
make -j4
make test
make install
```

## 2.3 Eigen3
This is a `c++` linear algebra template library. 
The library itself does not need to be compiled, but the tests require a long time to compile.
It is still recommended that you compile and run all tests.
```
mkdir ~/software/Eigen3
cd ~/software/Eigen3
git clone https://github.com/eigenteam/eigen-git-mirror.git
cd eigen-git-mirror/
git checkout 3.3.7
```
The version `3.3.7` is the latest version as of writing this document. You should use whatever latest release version.

Create a build folder and prepare a configure script, similar to YamlCpp:
```
mkdir ~/software/Eigen3/build
vim ~/software/Eigen3/do-configure.sh
```
The `do-configure.sh` file should be this:
```
#!/bin/bash

SOURCE_PATH=../eigen-git-mirror

rm -f CMakeCache.txt

cmake  \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="${HOME}/local/" \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D CMAKE_CXX_COMPILER:STRING="mpicxx" \
  -D CMAKE_C_COMPILER:STRING="mpicc" \
  -D CMAKE_CXX_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
  -D CMAKE_C_FLAGS:STRING="-O3 -march=native -DNDEBUG" \
$EXTRA_ARGS \
$SOURCE_PATH
```

Then:
```
cd ~/software/Eigen3/build
bash ../do-configure.sh
make check -j4
```
It usually takes about 1~2 hours to compile and run all tests.
After all tests are complete, you will see a summary:
```
99% tests passed, 1 tests failed out of 775

Label Time Summary:
Official       = 178.12 sec*proc (664 tests)
Unsupported    =  43.48 sec*proc (110 tests)

Total Test time (real) = 3086.25 sec

The following tests FAILED:
        666 - NonLinearOptimization (Child aborted)
Errors while running CTest
```
Typically a few (<10) tests fail, and you can safely ignore these fails.
If you see a large number of fails, check your compiler setup.

After tests, remember to install:
```
make install
```

## 2.4 Trilinos
```
mkdir ~/software/Trilinos && cd ~/software/Trilinos
git clone https://github.com/trilinos/Trilinos.git
```

This is a large repo. Depending on you internet and disk speed, it may take ~5 minutes to clone.
Then
```
cd Trilinos
git checkout trilinos-release-12-14-1
cd ~/software/Trilinos
vim do-configure.sh
```
Edit the file `do-configure.sh` as this:
```
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
cmake  \
  -D CMAKE_INSTALL_PREFIX:FILEPATH="$HOME/local/" \
  -D CMAKE_BUILD_TYPE:STRING="Release" \
  -D BUILD_SHARED_LIBS=OFF \
  -D Trilinos_HIDE_DEPRECATED_CODE=ON \
  -D CMAKE_C_FLAGS:STRING="-O3 -march=native" \
  -D CMAKE_CXX_FLAGS:STRING="-O3 -march=native" \
  -D OpenMP_C_FLAGS:STRING="-fopenmp" \
  -D OpenMP_CXX_FLAGS:STRING="-fopenmp" \
  -D Kokkos_ENABLE_HWLOC:BOOL=OFF \
  -D Kokkos_ENABLE_OpenMP:BOOL=ON \
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
Then, build trilinos:
```
mkdir build
cd build
bash ../do-configure.sh
make -j4
```
This may take 5~30 minutes, depending on the CPU and disk performance.
After compilation, run the tests:
```
export OMP_NUM_THREADS=2
make test
```
If you have 8 cores, use `OMP_NUM_THREADS=2`, if you have 12 cores, use `OMP_NUM_THREADS=3`, etc.

After all tests complete, you should see:
```
100% tests passed, 0 tests failed out of 294

Subproject Time Summary:
Belos      = 103.90 sec*proc (66 tests)
Ifpack2    =  39.21 sec*proc (38 tests)
Kokkos     =  53.82 sec*proc (27 tests)
Tpetra     = 155.44 sec*proc (163 tests)

Total Test time (real) = 153.46 sec
```
The total test time should be less than 400 sec. If it takes longer than that, check your OpenMP thread binding settings to make sure the threads are not conflicting with each other to compete for CPU cores.

You should make sure no error occurs. Then:
```
make install
```

## 2.5 PVFMM
Create a folder and clone the `new_BC` branch of pvfmm:
```
mkdir ~/software/PVFMM && cd ~/software/PVFMM
git clone https://github.com/wenyan4work/pvfmm.git
cd pvfmm
git checkout new_BC
./autogen.sh
bash ./lin_gcc_mkl_configure.sh
make -j4
make -j4 all-examples
```
Test the examples:
```
cd examples/bin
./fmm_pts -N 65536 -omp 8 -ker 3 | grep Error
```
The test should complete within a few seconds, and the output should look like:
```
Maximum Absolute Error [Output] :  4.19e-05
Maximum Relative Error [Output] :  1.54e-06
```
The errors should be on this order of magnitude.

After successful tests, install pvfmm:
```
cd ~/software/PVFMM/pvfmm
make install
```

Remember to add `PVFMM_DIR` to your `~/.bashrc`:
```
echo "export PVFMM_DIR=${HOME}/local/share/pvfmm" >> ~/.bashrc
```

Depending on your use case, you may need to install the precomputed PVFMM data files for various kernels. If you need, contact me or generate those files as described in the [PeriodicFMM repo](https://github.com/wenyan4work/PeriodicFMM.git) 

## 2.6 Boost

```
mkdir ~/software/Boost && cd ~/software/Boost
wget https://dl.bintray.com/boostorg/release/1.70.0/source/boost_1_70_0.tar.bz2
tar xvf ./boost_1_70_0.tar.bz2
cd boost_1_70_0
./bootstrap.sh --prefix=$HOME/local
./b2 install
```

# 3 Compile and test SimToolbox
Clone the SimToolbox repo somewhere on your machine:
```
git clone https://github.com/wenyan4work/SimToolbox.git
``` 
Modify the `do-cmake.sh` file to setup the correct PATH to the libraries:
```
#! /bin/bash

export USER_LOCAL=$HOME/local
export SYSTEM_LOCAL=/usr/local

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc \
  -D Eigen3_DIR="${USER_LOCAL}/share/eigen3/cmake" \
  -D TRNG_INCLUDE_DIR="${USER_LOCAL}/include" \
  -D TRNG_LIBRARY="${USER_LOCAL}/lib/libtrng4.a" \
  -D Trilinos_DIR="${USER_LOCAL}/lib/cmake/Trilinos" \
  -D yaml-cpp_DIR="${USER_LOCAL}/lib/cmake/yaml-cpp" \
../

```
For each of the libraries, use either `SYSTEM_LOCAL` or `USER_LOCAL`, or their actual locations. 
If you compiled those libraries as described in Part 2, they will be located in `USER_LOCAL`. 
Otherwise, use their actual locations (absolute path).

Create the `build` folder and run the `do-cmake.sh` script, build, and test:
```
cd SimToolbox
mkdir build
cd build
bash ../do-cmake.sh
make -j4
make test
```

