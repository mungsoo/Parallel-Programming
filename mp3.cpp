#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
//#define TILE_WIDTH 32
// Compute C = A * B
#define SHARE_MEM_PER_THREAD 8
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  extern __shared__ float array[];
  int blockWidth = blockDim.x;
  float *Mds = array;
  float *Nds = array + blockWidth*blockWidth;
  //__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  //__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  //int blockWidth = TILE_WIDTH;
  
  int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;
  int Row = by * blockWidth + ty;
  int Col = bx * blockWidth + tx;
  float PValue = 0;
  for(int ph = 0;ph < ceil(numAColumns / 1.0 / blockWidth);ph++)
  {
      if(ph * blockWidth + tx < numAColumns && Row < numARows)
          Mds[ty * blockWidth + tx] = A[numAColumns * Row + ph * blockWidth + tx];
          //Mds[ty][tx] = A[numAColumns * Row + ph * blockWidth + tx];

      else
          Mds[ty * blockWidth + tx] = 0;
          //Mds[ty][tx] = 0;
      if(ph * blockWidth + ty < numBRows && Col < numBColumns)
          Nds[ty * blockWidth + tx] = B[numBColumns * (ph * blockWidth + ty) + Col];
          //Nds[ty][tx] = B[numBColumns * (ph * blockWidth + ty) + Col];
      else
          Nds[ty * blockWidth + tx] = 0;
          //Nds[ty][tx] = 0;
      
      __syncthreads();
      
      for(int i = 0;i < blockWidth;i++)
          PValue += Mds[ty * blockWidth + i] * Nds[i * blockWidth + tx];
          //PValue += Mds[ty][i] * Nds[i][tx];
      __syncthreads();
  }  
  if(Row < numCRows && Col < numCColumns)
       C[Row * numCColumns + Col] = PValue;
       
      
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  int matALength, matBLength, matCLength;
   
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  cudaDeviceProp deviceProp;

  cudaGetDeviceProperties(&deviceProp, 0);
  int blockWidth;
  if(deviceProp.sharedMemPerMultiprocessor / SHARE_MEM_PER_THREAD >= deviceProp.maxThreadsPerMultiProcessor)
      blockWidth = int(sqrt(deviceProp.maxThreadsPerBlock));
  else if(deviceProp.sharedMemPerMultiprocessor / SHARE_MEM_PER_THREAD <= deviceProp.maxThreadsPerBlock)
      blockWidth = int(sqrt(deviceProp.sharedMemPerMultiprocessor / SHARE_MEM_PER_THREAD));
  else
  {
      int maxBlockNum = deviceProp.sharedMemPerMultiprocessor / SHARE_MEM_PER_THREAD / deviceProp.maxThreadsPerBlock + 1;
      blockWidth = int(sqrt(deviceProp.sharedMemPerMultiprocessor / SHARE_MEM_PER_THREAD / maxBlockNum));
  }
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  matALength = numARows * numAColumns;
  matBLength = numBRows * numBColumns;
  matCLength = numCRows * numCColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(sizeof(float) * matCLength);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void **)&deviceA, sizeof(float) * matALength));
  wbCheck(cudaMalloc((void **)&deviceB, sizeof(float) * matBLength));
  wbCheck(cudaMalloc((void **)&deviceC, sizeof(float) * matCLength));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, sizeof(float) * matALength, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, sizeof(float) * matBLength, cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numCColumns*1.0/blockWidth), ceil(numCRows*1.0/blockWidth), 1);
  dim3 dimBlock(blockWidth, blockWidth, 1);
  // dim3 dimGrid(ceil(numCColumns*1.0/TILE_WIDTH), ceil(numCRows*1.0/TILE_WIDTH), 1);
  // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  size_t arraySize = sizeof(float)*2*blockWidth*blockWidth;
  matrixMultiplyShared<<<dimGrid, dimBlock, arraySize>>>(deviceA, deviceB, deviceC, 
                                                numARows, numAColumns, 
                                                numBRows, numBColumns, 
                                                numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, sizeof(float) * matCLength, cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
