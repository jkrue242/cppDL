add_library(interfaces INTERFACE)

# setup eigen library
set(EIGEN3_ROOT "/opt/homebrew/opt/eigen")
set(EIGEN3_INCLUDE_DIR "${EIGEN3_ROOT}/include/eigen3")
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}")
    set(EIGEN3_INCLUDE_DIR "${EIGEN3_ROOT}/include")
endif()


target_include_directories(
    interfaces INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)