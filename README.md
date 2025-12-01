# cppDL

A simple C++ deep learning library for building and training neural networks. This is a work in progress.

## Requirements

- CMake 3.13 or higher
- C++17 compatible compiler
- Eigen library (install via Homebrew: `brew install eigen`)

## Building

### Using the build script

To build the project, use the provided build script:

```bash
./build.sh
```

This will create a `build/` directory, configure CMake, and compile the project.

## Running the Example

After building, you can run the XOR example:

```bash
./build/xor
```

This example trains a neural network to learn the XOR function and displays the training progress and final accuracy.

## Project Structure

```
cppDL/
├── network/          # Core neural network library
│   ├── network.hpp   # Network class
│   ├── layer.hpp     # Layer implementation
│   ├── functions.hpp # Activation functions
│   └── ...
├── examples/         # Example programs
│   └── xor.cpp       # XOR training example
└── build/            # Build directory (created during build)
```
