# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/vnk/miniconda3/lib/python3.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/vnk/miniconda3/lib/python3.7/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build"

# Include any dependencies generated for this target.
include CMakeFiles/stroke.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/stroke.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/stroke.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stroke.dir/flags.make

CMakeFiles/stroke.dir/main.cpp.o: CMakeFiles/stroke.dir/flags.make
CMakeFiles/stroke.dir/main.cpp.o: ../main.cpp
CMakeFiles/stroke.dir/main.cpp.o: CMakeFiles/stroke.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stroke.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stroke.dir/main.cpp.o -MF CMakeFiles/stroke.dir/main.cpp.o.d -o CMakeFiles/stroke.dir/main.cpp.o -c "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/main.cpp"

CMakeFiles/stroke.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stroke.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/main.cpp" > CMakeFiles/stroke.dir/main.cpp.i

CMakeFiles/stroke.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stroke.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/main.cpp" -o CMakeFiles/stroke.dir/main.cpp.s

CMakeFiles/stroke.dir/ETL/ETL.cpp.o: CMakeFiles/stroke.dir/flags.make
CMakeFiles/stroke.dir/ETL/ETL.cpp.o: ../ETL/ETL.cpp
CMakeFiles/stroke.dir/ETL/ETL.cpp.o: CMakeFiles/stroke.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/stroke.dir/ETL/ETL.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stroke.dir/ETL/ETL.cpp.o -MF CMakeFiles/stroke.dir/ETL/ETL.cpp.o.d -o CMakeFiles/stroke.dir/ETL/ETL.cpp.o -c "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/ETL/ETL.cpp"

CMakeFiles/stroke.dir/ETL/ETL.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stroke.dir/ETL/ETL.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/ETL/ETL.cpp" > CMakeFiles/stroke.dir/ETL/ETL.cpp.i

CMakeFiles/stroke.dir/ETL/ETL.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stroke.dir/ETL/ETL.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/ETL/ETL.cpp" -o CMakeFiles/stroke.dir/ETL/ETL.cpp.s

CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o: CMakeFiles/stroke.dir/flags.make
CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o: ../LogisticRegression/LogisticRegression.cpp
CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o: CMakeFiles/stroke.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o -MF CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o.d -o CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o -c "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/LogisticRegression/LogisticRegression.cpp"

CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/LogisticRegression/LogisticRegression.cpp" > CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.i

CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/LogisticRegression/LogisticRegression.cpp" -o CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.s

# Object files for target stroke
stroke_OBJECTS = \
"CMakeFiles/stroke.dir/main.cpp.o" \
"CMakeFiles/stroke.dir/ETL/ETL.cpp.o" \
"CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o"

# External object files for target stroke
stroke_EXTERNAL_OBJECTS =

stroke: CMakeFiles/stroke.dir/main.cpp.o
stroke: CMakeFiles/stroke.dir/ETL/ETL.cpp.o
stroke: CMakeFiles/stroke.dir/LogisticRegression/LogisticRegression.cpp.o
stroke: CMakeFiles/stroke.dir/build.make
stroke: /usr/lib/x86_64-linux-gnu/libboost_system.so
stroke: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
stroke: CMakeFiles/stroke.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable stroke"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stroke.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stroke.dir/build: stroke
.PHONY : CMakeFiles/stroke.dir/build

CMakeFiles/stroke.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stroke.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stroke.dir/clean

CMakeFiles/stroke.dir/depend:
	cd "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp" "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp" "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build" "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build" "/mnt/c/Users/Modern 14/projects/AI-on-Cpp/Logistic-Regression-Cpp/build/CMakeFiles/stroke.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/stroke.dir/depend

