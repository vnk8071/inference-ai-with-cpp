cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(stroke)
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake")
list(APPEND CMAKE_PREFIX_PATH "${PWD}/eigen")
find_package(Eigen3 REQUIRED)

find_package(Boost COMPONENTS system filesystem REQUIRED)
if(NOT TARGET Boost::filesystem)
    add_library(Boost::filesystem IMPORTED INTERFACE)
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set( project_sources
   main.cpp
   ETL/ETL.cpp
   LogisticRegression/LogisticRegression.cpp
)

add_executable(stroke main.cpp ${project_sources})
target_link_libraries(stroke ${EIGEN3_LIBRARIES} ${Boost_LIBRARIES})
set_property(TARGET stroke PROPERTY CXX_STANDARD 14)
