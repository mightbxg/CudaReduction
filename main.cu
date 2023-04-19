#include <cuda_runtime.h>
#include <curand.h>

#include <cassert>
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

template <unsigned int kBlockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  if (kBlockSize >= 64) sdata[tid] += sdata[tid + 32];
  if (kBlockSize >= 32) sdata[tid] += sdata[tid + 16];
  if (kBlockSize >= 16) sdata[tid] += sdata[tid + 8];
  if (kBlockSize >= 8) sdata[tid] += sdata[tid + 4];
  if (kBlockSize >= 4) sdata[tid] += sdata[tid + 2];
  if (kBlockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int kBlockSize>
__global__ void reduce(const float *g_idata, float *g_odata, size_t n) {
  extern __shared__ float sdata[];
  const unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (kBlockSize * 2) + tid;
  float my_sum = i < n ? g_idata[i] : 0.0f;
  if (i + kBlockSize < n) my_sum += g_idata[i + kBlockSize];
  sdata[tid] = my_sum;
  __syncthreads();

  if (kBlockSize >= 1024) {
    if (tid < 512) sdata[tid] += sdata[tid + 512];
    __syncthreads();
  }
  if (kBlockSize >= 512) {
    if (tid < 256) sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (kBlockSize >= 256) {
    if (tid < 128) sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (kBlockSize >= 128) {
    if (tid < 64) sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }
  if (tid < 32) warpReduce<kBlockSize>(sdata, tid);
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned kThreadNum>
float reduce_cu(const float *g_idata, int num) {
  const unsigned int blk_num = (num + kThreadNum * 2 - 1) / (kThreadNum * 2);
  if (blk_num > 32) return 0.0f;
  size_t smem_size = sizeof(float) * kThreadNum;
  float *tmp;
  CUDA_CHECK(cudaMalloc(&tmp, sizeof(float) * blk_num));
  reduce<kThreadNum><<<blk_num, kThreadNum, smem_size>>>(g_idata, tmp, num);
  // final reduction
  float *res_cu;
  CUDA_CHECK(cudaMalloc(&res_cu, sizeof(float)));
  reduce<32><<<1, 32, sizeof(float) * 64>>>(tmp, res_cu, blk_num);
  float res;
  CUDA_CHECK(cudaMemcpy(&res, res_cu, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(tmp));
  CUDA_CHECK(cudaFree(res_cu));
  return res;
}

int main() {
  const size_t num = 10000;
  float mean = 1.0f, std_dev = 10.0f;
  // generate random array
  vector<float> data;
  float *data_cu;
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

  // get device info
  cudaDeviceProp prop;
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  cout << "device info: name[" << prop.name << "] max_thd_num[" << prop.maxThreadsPerBlock << "]\n";

  // sum up in gpu
  auto sum_cu = reduce_cu<1024>(data_cu, num);
  cout << "sum from cuda: " << sum_cu << "\n";
}