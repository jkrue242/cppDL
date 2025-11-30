# cppDL

A deep learning library implemented in C++. This is still a work in progress.

## Currently Supported

- **Neural Network Layers**: Fully connected linear layers with configurable input/output sizes
- **Multi-layer Networks**: Sequential networks with multiple layers
- **Forward Pass**: Complete forward propagation through the network
- **Activation Functions**:
  - ReLU
  - Sigmoid
  - Softmax
  - Sigmoid derivative

## Tools/Frameworks

- **C++17**: Modern C++ standard
- **CMake 3.13+**: Build system
- **Eigen**: Linear algebra library for matrix operations

## Building

Use the provided build script:

```bash
./build.sh
```

This will:
1. Create a `build` directory if it doesn't exist
2. Configure CMake
3. Build the project using all available CPU cores

Alternatively, build manually:

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)
```

## Running

Use the provided run script:

```bash
./run.sh
```

Or run the executable directly:

```bash
./build/cppDL
```
