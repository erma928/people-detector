cmake_minimum_required(VERSION 3.10)

project(people-detector)
set(CMAKE_CXX_STANDARD 11)
option(ENABLE_NEON "ENABLE_NEON" ON)
option(ENABLE_VFPV3 "ENABLE_VFPV3" ON)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# TCLAP library is required from pkgconfig
find_package(PkgConfig REQUIRED)
pkg_check_modules(TCLAP REQUIRED tclap)
if (TCLAP_FOUND)
    include_directories (${TCLAP_INCLUDE_DIRS})
else (TCLAP_FOUND)
    message (SEND_ERROR "Unable to locate TCLAP")
endif (TCLAP_FOUND)

file(GLOB SOURCES
        contour/*.cpp
        dnn/*.cpp)

add_executable(people-detector main.cpp ${SOURCES})
target_link_libraries( people-detector ${OpenCV_LIBS} )
