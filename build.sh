#!/bin/bash -e

# build directory
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory..."
    mkdir -p "$BUILD_DIR"
fi

# configure and build
echo "Configuring CMake..."
cd "$BUILD_DIR"
cmake ..

echo "Building project..."
cmake --build . -j$(sysctl -n hw.ncpu)

echo "Build complete!"
echo ""

