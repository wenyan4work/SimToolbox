# Warning of deprecation
This document was working for macOS Mojave with Homebrew. Due to the stupid and frequent compatibility-breaking changes of macOS, there is no support for macOS now. 
The code may still work, but you are on your own. 
Good luck.


# Mac
I assume you use HomeBrew with default settings. We will setup everything with the llvm compiler from HomeBrew, and link compiled programs to bundled libraries instead of Mac's default c++ runtime.

Since Mojave, the `/usr/include` folder is removed by Apple. The necessary system headers can be found:
```bash
~ $ xcrun --show-sdk-version
10.14.4
~ $ xcrun --show-sdk-path
/Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk
~ $ ls $(xcrun --show-sdk-path)/usr/
bin     include lib     libexec share
```

You will need to specify the `-isysroot` to `clang` from `HomeBrew`:
```bash
/usr/local/opt/llvm/bin/clang++ -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk
```

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
export OMPI_CFLAGS="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk -I/usr/local/opt/llvm/include"
export OMPI_CXXFLAGS="-isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk -I/usr/local/opt/llvm/include"
export OMPI_LDFLAGS="-L/usr/local/lib -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
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
/usr/local/opt/llvm/bin/clang -I/usr/local/Cellar/open-mpi/4.0.0/include -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk -I/usr/local/opt/llvm/include -L/usr/local/lib -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -lmpi
~ $ mpicxx --showme
/usr/local/opt/llvm/bin/clang++ -I/usr/local/Cellar/open-mpi/4.0.0/include -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.14.sdk -I/usr/local/opt/llvm/include -L/usr/local/lib -L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib -lmpi
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
- **DO NOT** link to gcc's libgomp when using MKL on mac, because there is no gomp threading layer of MKL on mac. Program crashes or results are wrong.

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

Then at compile and link time, use this:
```bash
CFLAGS = -fopenmp=libiomp5
LDFLAGS = -fopenmp=libiomp5 -L/opt/intel/lib -liomp5 -Wl,-rpath,/opt/intel/lib
```

The `-Wl,-rpath` is necessary on Mac. More about this in the next section.

### Remark in 2019
As of llvm 8 and Intel MKL 2019, libomp seems to be working fine with MKL. There is no need to play the above tricks about omp libs when using clang with MKL. You can simply add `-fopenmp` to both compiling and linking flags.

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
  -D Trilinos_HIDE_DEPRECATED_CODE=ON \
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
