# CUDA Algorithms Showcase

This repository contains a CUDA project showcasing implementations of three fundamental algorithms: Matrix Multiplication, Bitonic Sort, and Odd-Even Sort. Each algorithm demonstrates the use of parallel computing concepts to improve performance on compatible NVIDIA GPUs.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- NVIDIA CUDA Toolkit (tested with version 12.4, but other versions may also work)
- An NVIDIA GPU that supports CUDA
- Visual Studio (for Windows users) or a similar IDE that supports CUDA development

### Installing

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Jvogie/CUDA-Algos.git
    ```
2. Navigate to the cloned directory:
    ```bash
    cd CUDA-Algos
    ```
3. Open the project in Visual Studio or your preferred IDE that supports CUDA development.
4. Build the project to ensure everything compiles correctly.
5. Run the executable to see the algorithms in action.

## Running the tests

To verify the correctness of the algorithms, observe the output of the program after running. The output will display:
- The result of matrix multiplication (the first element of the resulting matrix).
- The first element in the sorted array for both Bitonic and Odd-Even sorts, indicating the arrays were sorted in ascending order.


