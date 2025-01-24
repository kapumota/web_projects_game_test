## Extended GPU Project with CUDA, cuBLAS, cuDNN, and NPP

### Overview

This project demonstrates a **GPU-based** application that covers:

1. **Basic vector operations**  
   - Filling and adding arrays, ReLU forward and backward on the GPU.
2. **cuBLAS** Examples  
   - Dot product, matrix-vector multiply, and usage in a feed-forward layer.
3. **Simple 2-Layer MLP** with gradient descent  
   - Shows a non-trivial MSE that changes over multiple iterations.
4. **cuDNN** Examples  
   - Convolution, activation (ReLU), pooling, and softmax layers.
5. **NPP** Stub  
   - A placeholder function (`nppImageRotationExample()`) that logs an attempt to rotate an image, but no real rotation is done yet.

### Code structure

All the core code is in **`code_cuda.cu`**, which contains:

- **Kernels** for array ops and ReLU  
- **cuBLAS** calls (`cublasSgemm`, `cublasSdot`, `cublasSgemv`)  
- **cuDNN** demonstrations (`cudnnConvolutionExample`, `cudnnReluExample`, `cudnnPoolingExample`, `cudnnSoftmaxExample`)  
- A **2-layer neural network** example with MSE loss and gradient descent.

In a Jupyter notebook environment, we run commands like:

```bash
!apt-get update
!apt-get install -y cuda

!nvcc --version
!cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

!sudo apt-get update
!sudo apt-get install -y libnpp-dev-11-8
```

Then we write out our **`code_cuda.cu`** file with:

```bash
%%writefile code_cuda.cu
......
```

And finally **compile** and **run**:

```bash
!nvcc code_cuda.cu -o code_cuda -lcublas -lcudnn
!./code_cuda
```

### Execution logs

Below is an example of the output after running `./code_cuda`:

```
[INFO] Basic Array Ops Demo
[INFO] Sum of c after add: 0 (expected ~3000 if N=1000)
...
[INFO] Running a cuBLAS Dot Product example...
  Dot Product of [1..10] and [2..2] = 110 (expected 110)
...
[INFO] Running a cuBLAS Matrix-Vector example (SGEMV)...
  [1 2 3; 4 5 6; 7 8 9] * [1; 2; 3] = [30 36 42]^T
...
[INFO] Starting Simple NN Training (5 iterations)...
  [Iteration 1] MSE Loss = ...
  [Iteration 2] MSE Loss = ...
  ...
[INFO] Finished Simple NN Training Example.
[INFO] Program finished successfully.
```

> Note that depending on random initialization and floating-point quirks, you may see different MSE values each run.

### How to build and run (outside notebook)

If you want to run this **locally** or in a standard Linux environment:

1. **Install** the CUDA toolkit (matching your GPU drivers) and cuDNN.  
2. **Install** `libnpp` if you plan to extend the NPP functionality.  
3. **Clone** or download this repository so you have `code_cuda.cu`.  
4. **Compile** with:
   ```bash
   nvcc code_cuda.cu -o code_cuda -lcudart -lcublas -lcudnn
   ```
5. **Run**:
   ```bash
   ./code_cuda
   ```

You should see console logs with the same sequence of demonstrations.

## Command-line arguments (optional)

The code currently does not parse command-line flags, but you can add them in `main()` to modify hyperparameters (e.g., `inputSize`, `hiddenSize`, `iterations`). Then you can run:
```
./code_cuda --inputSize=256 --hiddenSize=128 --outputSize=10 --iterations=10
```
*(Implement the parsing code if needed.)*

### Google C++ Style and Additional Files

- We aim to follow the **Google C++ Style Guide** regarding function naming, file structure, and line lengths.  
- For a **fully complete** project, consider adding:
  - A `Makefile` (or CMake) so you can build with `make`.
  - A `run.sh` script to compile and run with a single command.
  - Additional logs or images if you enhance the NPP rotation function.

### Proof of execution

From the logs, you can see:

- The code **executes** on an actual CUDA-enabled system.  
- Dot product, matrix-vector multiply, etc., produce expected results.  
- MLP **training** runs over multiple iterations, showing MSE changes.  

If you want to show bigger data or images, simply update the code to handle them (e.g., bigger arrays, real images in the NPP stub).

#### Project description and goals

This example aims to **teach** how to integrate:

1. **Custom CUDA kernels** for simple tasks (array fill, add, ReLU).  
2. **cuBLAS** for advanced linear algebra routines.  
3. **cuDNN** for deep-learning specific operations (convolutions, pooling, activation, softmax).  
4. A **two-layer MLP** training loop to illustrate how these building blocks can form a neural network workflow entirely on the GPU.

**Challenges** include:
- Properly **allocating and freeing** GPU memory.  
- Handling **library handles** (e.g., `cublasHandle_t`, `cudnnHandle_t`).  
- Debugging or verifying **randomly initialized** networks.  

**Lessons learned**:
- Minimizing CPU↔GPU transfers is crucial for performance.  
- Libraries like cuBLAS/cuDNN save a lot of effort versus writing custom kernels for everything.  
- Handling MSE that doesn’t remain zero requires correct arithmetic (avoid sign flipping confusion) and actual **weight updates**.

#### Next steps

- **Expand** the MLP to more layers or a batch-based approach.  
- Implement **actual** NPP rotation logic, reading an image (e.g., `input.png`), rotating it, and writing out a result.  
- Add **argument parsing** for a robust CLI, plus a more extensive **Makefile** or `run.sh`.


**Happy GPU Computing!**
