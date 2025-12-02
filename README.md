# cppDL

A simple C++ deep learning library for deep learning model development. I am working on this to improve my C++ and ML/DL fundamentals, and to get a better feel for what goes on under the hood with some of the popular deep learning models. Sadly I do not have a GPU so no GPU support for large scale training yet.

## Requirements

- CMake 3.13 or higher
- C++17 compatible compiler
- Eigen library (install via Homebrew: `brew install eigen`)
- Boost library (install via Homebrew: `brew install boost`)

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
