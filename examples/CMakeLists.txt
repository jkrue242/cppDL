find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)

# Build the Python extension module as a shared library
add_library(cpDLL MODULE xor.cpp)
target_include_directories(cpDLL PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cpDLL PRIVATE 
    network 
    utils
    Boost::python
    Python::Python
)

# Set the output name to match the module name
set_target_properties(cpDLL PROPERTIES
    PREFIX ""
    OUTPUT_NAME "cpDLL"
)

if(APPLE)
    set_target_properties(cpDLL PROPERTIES
        SUFFIX ".so"
    )
endif()

# Copy the module to the examples directory so Python can find it
set_target_properties(cpDLL PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)