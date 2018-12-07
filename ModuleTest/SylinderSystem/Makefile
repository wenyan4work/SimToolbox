include ../MakefileInc.mk

# set of libraries to correctly link to the target

INCLUDE_DIRS = $(Trilinos_INCLUDE_DIRS) $(Trilinos_TPL_INCLUDE_DIRS) $(USERINCLUDE) -I$(CURDIR) 
LIBRARY_DIRS = $(Trilinos_LIBRARY_DIRS) $(Trilinos_TPL_LIBRARY_DIRS) $(USERLIB_DIRS)
LIBRARIES = $(Trilinos_LIBRARIES) $(Trilinos_TPL_LIBRARIES) $(USERLIBS)

# System-specific settings
SHELL = /bin/sh
SYSLIB =	
SIZE =	size

SIMTOOLBOX:=../..
# Files
SRC = main.cpp \
	$(SIMTOOLBOX)/Collision/CollisionSolver.cpp \
	$(SIMTOOLBOX)/Collision/CPSolver.cpp \
	$(SIMTOOLBOX)/Sylinder/Sylinder.cpp \
	$(SIMTOOLBOX)/Sylinder/SylinderSystem.cpp \
	$(SIMTOOLBOX)/Sylinder/SylinderConfig.cpp \
	$(SIMTOOLBOX)/Trilinos/TpetraUtil.cpp \
	$(SIMTOOLBOX)/Util/Base64.cpp

# Definitions
EXE :=   TestSylinderSimulator.X
OBJ :=   $(SRC:.cpp=.o)

all: $(EXE)

# pull in dependency info for *existing* .o files
-include $(OBJ:.o=.d)

# Link rule
$(EXE):	$(OBJ)
	$(LINK) $(OBJ)  -o $(EXE) $(LINKFLAGS) $(SYSLIB) $(LIBRARY_DIRS) $(LIBRARIES)
	$(SIZE) $(EXE)


# use the trick from
# http://scottmcpeak.com/autodepend/autodepend.html
# handle header dependency automatically
# compile and generate dependency info;
# more complicated dependency computation, so all prereqs listed
# will also become command-less, prereq-less targets
#   sed:    strip the target (everything before colon)
#   sed:    remove any continuation backslashes
#   fmt -1: list words one per line
#   sed:    strip leading spaces
#   sed:    add trailing colons
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o
	$(CXX) -MM $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp > $*.d
	@cp -f $*.d $*.d.tmp
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp


# remove compilation products
clean: 
	rm -f ./$(OBJ)
	rm -f ./$(EXE)
	rm -f ./*.d
	rm -f ./*~
