# CUDA-Stencil-Optimisation
Various more or less instructive performance optimisations for stencil computation in CUDA.

Naive implementation uses 2D grid striding technique to traverse the domain with threads. Data dependency is handled through seperation into small kernels and blocking memory copying with cudaMemcpy.

Vectorize makes use of the fact that the 2 arrays P and Q are always accessed in pairs so they may as well exist next to eachother in memory in an RGB sense.

Cooperative removes all **cudaKernelLaunch** overhead associated with launching many small kernels, by converting the kernels into __device__ functions and launching a larger __global__ kernel which combines all steps and performs N iterations. **cudaLaunchCooperativeKernel** is invoked here to allow for GPU side grid-wide synchronization.

Async overlaps all host - device copying using a second buffer to store output on the device which can then be copied from asynchronously using **cudaMemcpyAsync** and host pinned memory using **cudaMallocHost**. 

Invert utilises the temporal locality of PQ and dPQdt arrays in L2 cache by inverting the integration loop: The derivative loop updates the last elements PQ at the end thus this is the first data we should make use of in the integration loop.

Cache uses a manual 2D cache-tiling approach to copying data from DRAM into L1 cache thus ensuring reuse of data between threads of the same thread-block. This optimisation works well for larger grids, when the arrays no longer fit in L2 cache.

Pipeline uses asynchronous device side memory copying operations thus reducing register pressure, it also makes use of the fact that we can precalculate some parts of the PDEs that only depend on the threads own center value.

Double Buffer uses a second buffer to store a copy of the PQ array and pointer swapping to deal with the data dependency between calculating the time-derivative and integrating the array on which it depends. This is by far the largest leap in performance.
