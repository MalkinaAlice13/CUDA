#!/usr/bin/env python

from time import time

import numpy
import pycuda.autoinit
from pycuda import compiler, gpuarray

kernel_code_template = """
__global__ void matrix_mul_gpu(float *a, float *b, float *c)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float c_element = 0;

    // Each thread loads one row of A and one column of B.
    for (int i = 0; i < %(SIZE)s; i++)
    {
        float a_element = a[ty * %(SIZE)s + i];
        float b_element = b[i * %(SIZE)s + tx];

        c_element += a_element * b_element;
    }

    // Write the matrix to device memory, each thread writes one element.
    c[ty * %(SIZE)s + tx] = c_element;
}
"""


def multiply_cpu(A, B, size):
    result = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        for j in range(size):
            for k in range(size):
                result[i][j] += A[i][k] * B[k][j]

    return result


def main():
    # Init.
    size = 64

    print(f"Size: {size}x{size}")

    A = [[1 for _ in range(size)] for _ in range(size)]
    B = [[2 for _ in range(size)] for _ in range(size)]

    # ------------------------------------------------------------------
    # CPU:

    t1 = time()
    C = multiply_cpu(A, B, size)
    t2 = time()

    cpu_time = t2 - t1

    print("CPU:", "{:.6f}".format(round(cpu_time, 6)))

    # ------------------------------------------------------------------
    # GPU:

    # Copy matrices to GPU.
    A = gpuarray.to_gpu(numpy.asarray(A).astype(numpy.float32))
    B = gpuarray.to_gpu(numpy.asarray(B).astype(numpy.float32))

    # Create new empty matrix on GPU.
    C = gpuarray.empty((size, size), numpy.float32)

    # Create dict of parameters.
    kernel_code = kernel_code_template % {"SIZE": size}

    # Compile C++ code.
    mod = compiler.SourceModule(kernel_code)

    # Get function.
    multiply_gpu = mod.get_function("matrix_mul_gpu")

    t1 = time()
    multiply_gpu(A, B, C, block=(16, 16, 1))
    t2 = time()

    gpu_time = t2 - t1

    print("GPU:", "{:.6f}".format(round(gpu_time, 6)))

    # ------------------------------------------------------------------

    print("CPU / GPU:", "{:.6f}".format(round(cpu_time / gpu_time, 6)))


if __name__ == "__main__":
    main()
