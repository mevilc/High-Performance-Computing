## HPC -- CPEG 655

**Lab 1**:  C programs optimized to maximize L2 and TLB cache misses. Misses measured with PAPI (C API) hardware counter. Misses/hits generated by array memory accesses.

**Lab 2**:
- C programs optimized to maximize and minimize L2 cache misses. Struct arranged as a struct-of-array (SoA). Misses measured with PAPI (C API) hardware counter. Misses/hits generated by array memory accesses.
- Tracked node, edge, and path profile of a workload and measured memory heirarchy performance of all paths and most frequently executed path.

**Lab 3**:  C++ program to build a binary tree. Lock/unlock operations used to control access to the tree methods using std::mutex. Coarse vs fine lock granulity explored. Multi-threading with pthreads and std::threads used to run a workload in parallel.

**Lab 4**:  CUDA program to perform parallelized matrix multiplications using:
- one thread per product element
- one thread block per product tile (cache tiled method)
- one thread per product tile
