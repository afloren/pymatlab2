cmake_minimum_required (VERSION 2.6)

project (PyMatlab)

find_package(PythonLibs REQUIRED)
find_package(Matlab REQUIRED)
find_path(NUMPY_INCLUDE_DIR numpy/ndarrayobject.h) 
include_directories(${PYTHON_INCLUDE_DIRS} ${MATLAB_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})

set(sources engine.cpp)
set(lib_name engine)
add_library(${lib_name} SHARED ${sources})

target_link_libraries(${lib_name} ${PYTHON_LIBRARIES} ${MATLAB_LIBRARIES})


