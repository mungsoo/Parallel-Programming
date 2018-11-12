// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//__device__ int blockCounter = 0;

__global__ void prefixFixup(float *input, float *aux, int len)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int start = blockIdx.x * 2 * BLOCK_SIZE + tid;
  if(bid > 0)
  {
    if(start < len)
      input[start] += aux[bid - 1];
    if(start + BLOCK_SIZE < len)
      input[start + BLOCK_SIZE] += aux[bid - 1];
  }
}
__global__ void scan(float *input, float *output, int len, float *aux) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  //__shared__ int sbid;
  
  // Load the input into shared memory
  __shared__ float array[2 * BLOCK_SIZE];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int start = bid * BLOCK_SIZE * 2 + tid;
  if(start < len)
    array[tid] = input[start];
  else
    array[tid] = 0;
  if(start + BLOCK_SIZE < len)
    array[tid + BLOCK_SIZE] = input[start + BLOCK_SIZE];
  else
    array[tid + BLOCK_SIZE] = 0;
  
  // Reduction phase
  int stride = 1;
  while(stride < 2 * BLOCK_SIZE)
  {
    __syncthreads();
    int index = (tid + 1) * stride * 2 - 1;
    if(index < 2 * BLOCK_SIZE)
      array[index] += array[index - stride];
    stride *= 2;
  }
  
  // Post scan phase
  stride = BLOCK_SIZE / 2;
  while(stride > 0)
  {
    __syncthreads();
    int index = (tid + 1) * 2 * stride - 1;
    if(index + stride < 2 * BLOCK_SIZE)
      array[index + stride] += array[index];
    stride /= 2;
  }
  
  __syncthreads();
  // Directly write output
  if(start < len)
    output[start] = array[tid];
  if(start + BLOCK_SIZE < len)
    output[start + BLOCK_SIZE] = array[tid + BLOCK_SIZE];
  
  // Need current block sum out
  if(aux != NULL && tid == 0)
    aux[bid] = array[2 * BLOCK_SIZE - 1];

  //if(threadIdx.x == 0)
  //{
  //  sbid = atomicAdd(&blockCounter, 1);
    //sbid = atomicAdd(*blockCounter, 1);
  //  output[sbid] = output[sbid] + 1;
  //} 
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list  
  //float *flag;
  //float *preSum;
  //int blockCounter = 0;
  float *aux;
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  int numBlocks = ceil(1.0 * numElements / 2 / BLOCK_SIZE);
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&aux, numBlocks * sizeof(float)));
  //wbCheck(cudaMalloc((void **)&flag, numBlocks * sizeof(float)));
  //wbCheck(cudaMalloc((void **)&preSum, numBlocks * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(aux, 0, numBlocks * sizeof(float)));
  //wbCheck(cudaMemset(flag, 0, numBlocks * sizeof(float)));
  //wbCheck(cudaMemset(flag, 1, 1));
  //wbCheck(cudaMemset(preSum, 0, numBlocks * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 blockSize(BLOCK_SIZE, 1, 1);
  dim3 gridSize(numBlocks, 1, 1);
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  
  scan<<<gridSize, blockSize>>>(deviceInput, deviceOutput, numElements, aux);
  cudaDeviceSynchronize();
  scan<<<1, BLOCK_SIZE>>>(aux, aux, numBlocks, NULL);
  cudaDeviceSynchronize();
  prefixFixup<<<gridSize, blockSize>>>(deviceOutput, aux, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");
  wbLog(TRACE, "output", deviceOutput);
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(aux);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
