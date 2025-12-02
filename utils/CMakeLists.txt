# Find curl and boost
find_package(CURL REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem serialization python)

add_library(utils STATIC
    dataset_util.cpp
)
target_link_libraries(utils PUBLIC CURL::libcurl 
    Boost::boost
    Boost::filesystem
    Boost::serialization
    Boost::python
    Python::Python
)
target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)