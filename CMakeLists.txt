cmake_minimum_required(VERSION 3.13)
project(cppDL VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(network)
add_executable(xor examples/xor.cpp)

target_include_directories(xor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(xor PRIVATE network)

