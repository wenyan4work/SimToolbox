include ../MakefileInc.mk

SRC =   main.cpp 
INC =   ../../Trilinos/ZDD.hpp
         
# Definitions

EXE =   TestZDD.X
OBJ :=   $(SRC:.cpp=.o)


# Link rule
$(EXE):	$(OBJ) 
	$(LINK) $(OBJ) -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)

# Compilation rules
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o


all: $(EXE)

clean: 
	rm ./$(OBJ)
	rm ./$(EXE)