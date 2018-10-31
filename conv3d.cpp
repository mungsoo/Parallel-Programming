#include <wb.h>
#define KERNEL_WIDTH 3
#define TILE_WIDTH 8
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  //Shared memory
  __shared__ float inputTile[TILE_WIDTH + KERNEL_WIDTH - 1][TILE_WIDTH + KERNEL_WIDTH - 1][TILE_WIDTH + KERNEL_WIDTH - 1];
  //Compute indexes
  int col_o = bx * TILE_WIDTH + tx;
  int row_o = by * TILE_WIDTH + ty;
  int hei_o = bz * TILE_WIDTH + tz;
  int row_i = row_o - int(KERNEL_WIDTH / 2.0);
  int col_i = col_o - int(KERNEL_WIDTH / 2.0);
  int hei_i = hei_o - int(KERNEL_WIDTH / 2.0);
  float result = 0;
  //Load all the essential elements
  if(hei_i >= 0 && col_i >= 0 && row_i >= 0 && hei_i < z_size && row_i < y_size && col_i < x_size)
      inputTile[tz][ty][tx] = input[hei_i * (x_size * y_size) + row_i * x_size + col_i];
  else
      inputTile[tz][ty][tx] = 0.0;
  
  __syncthreads();
  //Compute result
  if(tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH)
  {
      for(int i = 0;i < KERNEL_WIDTH;i++)
          for(int j = 0;j < KERNEL_WIDTH;j++)
              for(int k = 0;k < KERNEL_WIDTH;k++)
                  result += deviceKernel[i][j][k] * inputTile[i + tz][j + ty][k + tx];
  
  
    if(hei_o < z_size && row_o < y_size && col_o < x_size)
        output[hei_o * (y_size * x_size) + row_o * x_size + col_o] = result;
  }    
      
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength, matLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  matLength = inputLength - 3;
  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **)&deviceInput, sizeof(float) * matLength);
  cudaMalloc((void **)&deviceOutput, sizeof(float) * matLength);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], sizeof(float) * matLength, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, sizeof(float) * KERNEL_WIDTH * KERNEL_WIDTH * KERNEL_WIDTH);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 blockSize(TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1, TILE_WIDTH + KERNEL_WIDTH - 1);
  dim3 gridSize(ceil(x_size*1.0/TILE_WIDTH), ceil(y_size*1.0/TILE_WIDTH), ceil(z_size*1.0/TILE_WIDTH));
  //@@ Launch the GPU kernel here
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");
  conv3d<<<gridSize, blockSize>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  
  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, sizeof(float) * matLength, cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}