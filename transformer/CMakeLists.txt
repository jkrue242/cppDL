add_executable(transformer gpt.cpp)
target_include_directories(transformer PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(transformer PRIVATE network utils)