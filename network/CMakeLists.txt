add_subdirectory(interfaces)
add_subdirectory(layers)
add_subdirectory(losses)
add_library(network INTERFACE)

target_include_directories(
    network INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(network INTERFACE interfaces layers losses)
