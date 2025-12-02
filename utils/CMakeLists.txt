# Find curl
find_package(CURL REQUIRED)

add_library(utils STATIC
    dataset_util.cpp
)
target_link_libraries(utils PUBLIC CURL::libcurl)
target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)