# 0
This is a step-by-step document of setting up necessary environment, toolchain, and dependence libraries.
Depending on your operating system, you may need to slightly change the steps described in this document.
The steps described here are used to setup things on Ubuntu 18.04 of WSL.
If you do not know what WSL is, just use the same steps on Ubunto 18.04 Linux.

# 1 Compilers and building tools
```
sudo apt install build-essential libopenmpi-dev automake cmake cmake-curses-gui
```
This will install `gcc`, `openmpi`, `automake`, and `cmake`.
If you are on a cluster, make sure you have a working compiler supporting `c++14` and `OpenMP`, and make sure the version of `cmake` is >=3.10.

On Ubuntu 18.04, after the above step you should have working compilers and building tools. 
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
```
mkdir ~/software/TRNG && cd ~/software/TRNG
git clone https://github.com/rabauke/trng4.git
cd trng4
```
Then, configure and compile it.
```
autoreconf -f -i
export CC=mpicc && export CXX=mpicxx && ./configure CFLAGS="-O3 -march=native -DNDEBUG -std=c99" CXXFLAGS="-O3 -march=native -DNDEBUG -std=c++14" --prefix=$HOME/local
make -j4
make -j4 examples
```
Then, test it:
```
wyan@DESKTOP-U6JNAI0:~/software/TRNG/trng4$ ./examples/time
                                            10^6 random numbers per second
generator                       [min,max] [0,1]     [0,1)     (0,1]     (0,1)     canonical
=============================================================================================
trng::lcg64                     1080.6    1071.21   462.195   477.209   471.324   481.124
trng::lcg64_shift               908.934   475.87    462.642   480.379   471.123   465.309
trng::mrg2                      213.274   212.235   212.445   203.269   210.377   201.611
trng::mrg3                      188.159   180.429   182.673   175.941   177.779   177.552
trng::mrg3s                     181.808   180.109   185.85    176.526   181.599   176.714
trng::mrg4                      188.61    182.898   176.156   177.238   187.223   185.776
trng::mrg5                      225.585   247.018   249.408   241.548   240.251   239.326
trng::mrg5s                     130.991   133.852   136.904   137.348   133.505   136.231
trng::yarn2                     232.117   219.744   217.169   215.563   212.912   218.127
trng::yarn3                     213.932   201.611   201.911   191.346   198.505   201.25
trng::yarn3s                    164.117   142.575   152.107   151.435   149.87    152.301
trng::yarn4                     147.333   140.544   145.279   141.569   144.293   144.109
trng::yarn5                     165.963   155.6     161.75    162.962   159.879   161.163
trng::yarn5s                    123.378   123.616   127.116   124.753   124.04    124.764
trng::mt19937                   425.019   340.391   334.782   323.41    343.712   338.837
trng::mt19937_64                403.833   332.11    217.764   201.212   207.68    206.098
trng::lagfib2xor_19937_64       1106.76   438.333   474.711   475.76    469.714   471.138
trng::lagfib4xor_19937_64       762.078   467.424   472.667   463.767   465.605   467.281
trng::lagfib2plus_19937_64      1064.55   467.694   472.784   471.64    471.947   471.309
trng::lagfib4plus_19937_64      810.224   464.741   458.871   407.708   474.605   457.047
std::minstd_rand0               292.688   293.823   301.396   288.715   291.437   286.839
std::minstd_rand                296.05    295.873   295.259   280.214   287.32    290.068
std::mt19937                    223.851   211.467   208.979   195.86    184.268   202.481
std::mt19937_64                 208.25    191.418   206.113   198.787   199.112   196.666
std::ranlux24_base              173.544   171.833   172.879   171.636   171.878   176.882
std::ranlux48_base              205.684   211.791   206.983   213.093   213.772   206.631
std::ranlux24                   17.3029   16.91     16.9856   16.9126   17.1392   17.3823
std::ranlux48                   5.88981   5.83369   5.77585   5.82133   5.82979   5.86228
std::knuth_b                    95.8879   95.3803   94.6033   93.8408   93.3119   92.975
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

