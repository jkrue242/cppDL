add_executable(xor xor.cpp)
target_include_directories(xor PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(xor PRIVATE network)