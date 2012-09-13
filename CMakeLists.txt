cmake_minimum_required (VERSION 2.6)

project (PyMatlab)

find_package(PythonLibs REQUIRED)
find_package(Matlab REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS} ${MATLAB_INCLUDE_DIR})

set(sources engine.c)
set(lib_name engine)
add_library(${lib_name} SHARED ${sources})

target_link_libraries(${lib_name} ${PYTHON_LIBRARIES} ${MATLAB_LIBRARIES})


