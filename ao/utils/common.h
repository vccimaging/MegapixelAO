#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int StrideAlignment = 32;

// A GPU timer
struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
  	}

  	~GpuTimer()
  	{
    	cudaEventDestroy(start);
    	cudaEventDestroy(stop);
  	}

	void Start()
  	{
    	cudaEventRecord(start, 0);
  	}

  	void Stop()
  	{
    	cudaEventRecord(stop, 0);
  	}

  	float Elapsed()
  	{
    	float elapsed;
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsed, start, stop);
    	return elapsed;
  	}
  	
  	void Free()
  	{
    	cudaEventDestroy(start);
    	cudaEventDestroy(stop);
  	}
};

// Align up n to the nearest multiple of m
inline int iAlignUp(int n, int m = StrideAlignment)
{
    int mod = n % m;

    if (mod)
        return n + m - mod;
    else
        return n;
}

// round up n/m
inline int iDivUp(int n, int m)
{
    return (n + m - 1) / m;
}

// swap two values
template<typename T>
inline void Swap(T &a, T &b)
{
    T t = a;
    a = b;
    b = t;
}

// Wrap to [-0.5 0.5], then add 0.5 to [0 1] for final phase show
__host__
__device__
inline float wrap(float x)
{
    return x - floor(x + 0.5f) + 0.5f;
}
#endif
