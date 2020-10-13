cmake_minimum_required(VERSION 3.7)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(N4 N4.cxx)
target_link_libraries(N4 ${ITK_LIBRARIES})
