include ../MakefileInc.mk

SRC =\
main.cpp \
../../Sphere/Sphere.cpp \
../../Sphere/Shexp.cpp \
../../Sphere/SphereXMLVTKReader.cpp \
../../Util/Base64.cpp
INC =\
../../Sphere/Sphere.hpp\
../../Sphere/Shexp.hpp\
../../Sphere/SphereXMLVTKReader.hpp \
../../Util/Buffer.hpp
         
# Definitions
EXE =   TestDump.X
OBJ :=   $(SRC:.cpp=.o)

INCLUDE_DIRS += -I$(HOME)/local/include/vtk-8.2
LIBRARY_DIRS += -L$(HOME)/local/lib64

# WARNING: DO NOT TOUCH THIS
LIBRARIES += \
-lvtkIOXML-8.2 \
-lvtkIOParallel-8.2 \
-lvtkIOParallelXML-8.2 \
-lvtkIOXMLParser-8.2 \
-lvtkIOCore-8.2 \
-lvtkCommonExecutionModel-8.2 \
-lvtkCommonDataModel-8.2 \
-lvtkCommonMisc-8.2 \
-lvtkCommonSystem-8.2 \
-lvtkCommonTransforms-8.2 \
-lvtkCommonMath-8.2 \
-lvtkCommonCore-8.2 \
-lvtkParallelCore-8.2 \
-lvtkFiltersCore-8.2 \
-lvtklibxml2-8.2 \
-lvtkzlib-8.2 \
-lvtklzma-8.2 \
-lvtklz4-8.2 \
-lvtkexpat-8.2 \
-lvtksys-8.2 \


# Link rule
$(EXE):	$(OBJ) 
	$(LINK) $(OBJ) -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)

# Compilation rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o


all: $(EXE)

clean: 
	rm -f ./$(OBJ)
	rm -f ./$(EXE)
