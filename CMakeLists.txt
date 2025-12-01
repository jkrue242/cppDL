cmake_minimum_required(VERSION 3.13)
project(cppDL VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

# Generate compile_commands.json for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# subdirectories
add_subdirectory(network)

# executable
add_executable(cppDL xor.cpp)

# link to network librarys
target_link_libraries(cppDL PRIVATE network)

