#include <cuda_runtime.h>
#include <curand.h>

#include <iostream>
#include <sstream>
#include <vector>

using namespace std;

#define CUPRINT(x, ...) \
  { printf("\33[33m(CUDA) " x "\n\33[0m", ##__VA_ARGS__); }

#define CUDA_CHECK(err)                                                          \
  do {                                                                           \
    cudaError_t err_ = (err);                                                    \
    if (err_ != cudaSuccess) {                                                   \
      std::stringstream ss;                                                      \
      ss << "CUDA error " << int(err_) << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(ss.str());                                        \
    }                                                                            \
  } while (false)

#define CURAND_CHECK(err)                                                          \
  do {                                                                             \
    curandStatus_t err_ = (err);                                                   \
    if (err_ != CURAND_STATUS_SUCCESS) {                                           \
      std::stringstream ss;                                                        \
      ss << "cuRAND error " << int(err_) << " at " << __FILE__ << ":" << __LINE__; \
      throw std::runtime_error(ss.str());                                          \
    }                                                                              \
  } while (false)

template <unsigned int blockSize>
__device__ void warpReduce(volatile int* sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
}

int main() {
  const size_t num = 10000;
  float mean = 1.0f, std_dev = 10.0f;
  // generate random array
  vector<float> data;
  float* data_cu;
  {
    CUDA_CHECK(cudaMalloc(&data_cu, sizeof(float) * num));
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937));
    CURAND_CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 12345));
    CURAND_CHECK(curandGenerateNormal(gen, data_cu, num, mean, std_dev));
    data.resize(num);
    CUDA_CHECK(cudaMemcpy(data.data(), data_cu, sizeof(float) * num, cudaMemcpyDeviceToHost));
  }
  float sum = 0;
  for (auto v : data) {
    sum += v;
  }
  cout << num << " random numbers generated, sum is " << sum << "\n";
}