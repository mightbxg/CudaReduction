cmake_minimum_required(VERSION 3.10)

project(CudaReduction)

set(CUDA_TOOLKIT_ROOT_DIR "/opt/compiler/cuda")
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
find_package(CUDA REQUIRED)

cuda_add_executable(t main.cu)
target_link_libraries(t ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcurand.so)
