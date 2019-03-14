SFTPATH:=/mnt/home/wyan/local

# inherit flags, includes and libs from Trilinos and pvfmm
include $(SFTPATH)/include/Makefile.export.Trilinos
include $(PVFMM_DIR)/MakeVariables

# internal includes
SIMTOOLBOX := ${CURDIR}/../../

# external libraries
TRNG = $(SFTPATH)/include/trng
EIGEN= $(SFTPATH)/include/eigen3
PVFMM= $(SFTPATH)/include/pvfmm
YAML= $(SFTPATH)/include/yaml-cpp

USERINCLUDE = -I$(TRNG)/include -I$(EIGEN) -I$(SIMTOOLBOX)
USERLIB_DIRS = -L$(SFTPATH)/lib
USERLIBS = -ltrng4 -lyaml-cpp

INCLUDE_DIRS = $(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS) $(USERINCLUDE)
LIBRARY_DIRS = $(Trilinos_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARY_DIRS) $(USERLIB_DIRS)
LIBRARIES = $(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES) $(USERLIBS)

CXX= mpicxx
LINK= $(CXX)

# optimized
CXXFLAGS= $(CXXFLAGS_PVFMM) 
LINKFLAGS= $(CXXFLAGS) $(LDLIBS_PVFMM) $(Trilinos_EXTRA_LD_FLAGS) #-lm -ldl

# remove some flags for debugging
# if Trilinos and pvfmm are compiled with ipo, removing this may cause linking failures

# debug
DEBUGMODE:= yes

ifeq ($(DEBUGMODE), yes)
CXXFLAGS:= $(subst -O3, ,$(CXXFLAGS))
LINKFLAGS:= $(subst -O3, ,$(LINKFLAGS))
CXXFLAGS := $(CXXFLAGS) -O0 -g
LINKFLAGS := $(LINKFLAGS) -O0 -g
else
CXXFLAGS:= $(CXXFLAGS) -DNDEBUG
LINKFLAGS:= $(LINKFLAGS) -DNDEBUG
endif

# almost always yes
WITHMPI ?= yes

CXXFLAGS += -DPARTICLE_SIMULATOR_MPI_PARALLEL
CXXFLAGS += -DPARTICLE_SIMULATOR_THREAD_PARALLEL

