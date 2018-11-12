// Histogram Equalization

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

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512

__global__ void fromFloatToUnsigned(float *input, unsigned char *output, int len)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if(index < len)
    output[index] = (unsigned char) (255 * input[index]);
  //if (index == 0) printf("input float: %f, input unsigned: %d\n", input[0], output[0]);
}

__global__ void fromRGBToGrayScale(unsigned char *input, unsigned char *output, int len)
{
  
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < len)
  {
    unsigned char r = input[3 * index];
    unsigned char g = input[3 * index + 1];
    unsigned char b = input[3 * index + 2];
    output[index] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
    //if(index == 0){ printf("gray scale: %u, r %u, g %u b %u\n", output[0], r, g, b);
                   //printf("gray scale: %u\n", output[1]);}
  }
  
}


__global__ void histogram(unsigned char *input, int *hist, int len)
{
  __shared__ unsigned int sHist[HISTOGRAM_LENGTH];
  int tid = threadIdx.x;
  int index = tid + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  // Notice, here use tid < HISTOGRAM_LENGTH
  // Since each block need to maintain its own sHist.
  if(tid < HISTOGRAM_LENGTH)
    sHist[tid] = 0;
  
  __syncthreads();
  while(index < len)
  {
    atomicAdd(&(sHist[input[index]]), 1);
    index += stride;
  }
  __syncthreads();
  if(tid < HISTOGRAM_LENGTH)
    atomicAdd(&(hist[tid]), sHist[tid]);

  // Why this unoptimized approach even better than the approach above in the dataset?
  //  while(index < len)
  //{
  //  atomicAdd(&(hist[input[index]]), 1);
  //  index += stride;
  //}
  

  //__syncthreads();
  //if(index == stride) {
  //  int sum = 0;
  //  for(int i = 0; i < 256;i++){
  //    printf("histogram: %d\n", hist[i]);
  //    sum += hist[i];}
  //printf("histogram 137 : %d\n", hist[137]);
  //printf("total: %d\n", sum);
  //  }
}

__global__ void getCDF(int *input, float *output, int total) {
  // Only accept input array of element num no more than BLOCK_SIZE.
  // Use Kogge Stone.
  __shared__ float array1[BLOCK_SIZE];
  __shared__ float array2[BLOCK_SIZE];
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int len = HISTOGRAM_LENGTH;
  int index = bid + tid;
  float *tmp;
  float *src = array1, *dst = array2;
  
  // Load data
  if(index < len)
    src[index] = input[index];
  else
    src[index] = 0;
  
  // Kogge Stone scan
  
  int stride = 1;
  while(stride < len)
  {
    __syncthreads();
    if(index >= stride)
      dst[index] = src[index] + src[index - stride];
    tmp = dst;
    dst = src;
    src = tmp;
    stride *= 2;
  }
  
  __syncthreads();
  
  // Write to output
  if(index < len)
    output[index] = src[index] / total;
  
  //__syncthreads();
  //if(index == 0)
  //  printf("cdf 0: %f\n, cdf n: %f\n", output[0], output[len-1]);
}

__global__ void equalization(unsigned char *input, float *output, float *cdf, int cdfmin, int len)
{
    
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < len)
    {
        unsigned char val = input[index];
        //output[index] = (float) ((unsigned char)(min(max(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0.0), 255.0)) / 255.0); 
        output[index] = (float) (min(max(255*(cdf[val] - cdfmin)/(1.0 - cdfmin), 0.0), 255.0) / 255.0); 
    }
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  
  int imageLength;
  int imageSize;
  
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  
  const char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  
  unsigned char *deviceInputRGBData;
  unsigned char *deviceInputGrayScaleData;
  int *hist;
  float *cdf;
  float cdfmin;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  
  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  
  imageLength = imageWidth * imageHeight * imageChannels;
  imageSize = imageWidth * imageHeight;
  
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);//get image data 
  hostOutputImageData = wbImage_getData(outputImage); 
  
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "SIZE: ", imageLength);
  
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imageLength * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceInputRGBData, imageLength * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&deviceInputGrayScaleData, imageSize * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **)&hist, HISTOGRAM_LENGTH * sizeof(int)));
  wbCheck(cudaMalloc((void **)&cdf, HISTOGRAM_LENGTH * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imageLength * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing hist memory.");
  wbCheck(cudaMemset(hist, 0, HISTOGRAM_LENGTH * sizeof(int)));
  wbTime_stop(GPU, "Clearing hist memory.");
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageLength * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(GPU, "Convert from float to unsigned.");
  fromFloatToUnsigned<<<ceil(1.0 * imageLength / BLOCK_SIZE), BLOCK_SIZE>>>(deviceInputImageData, 
                     deviceInputRGBData, imageLength);
  wbTime_stop(GPU, "Convert from float to unsigned.");
  cudaDeviceSynchronize();
  
  wbTime_start(GPU, "Convert from RGB to gray scale");
  fromRGBToGrayScale<<<ceil(1.0 * imageSize / BLOCK_SIZE), BLOCK_SIZE>>>(deviceInputRGBData, 
                     deviceInputGrayScaleData, imageSize);
  wbTime_stop(GPU, "Convert from RGB to gray scale");
  cudaDeviceSynchronize();
  
  wbTime_start(GPU, "Compute histogram");
  histogram<<<ceil(1.0 * imageSize / BLOCK_SIZE), BLOCK_SIZE>>>(deviceInputGrayScaleData, hist, imageSize);
  wbTime_stop(GPU, "Compute histogram");
  cudaDeviceSynchronize();  
  
  wbTime_start(GPU, "Compute cdf");
  getCDF<<<1, BLOCK_SIZE>>>(hist, cdf, imageSize);
  wbTime_stop(GPU, "Compute cdf");
  cudaDeviceSynchronize();  
  
  // Get the minimal cdf
  wbCheck(cudaMemcpy(&cdfmin, cdf, 1 * sizeof(float), cudaMemcpyDeviceToHost));
  
  wbTime_start(GPU, "Equalization");
  equalization<<<ceil(1.0 * imageLength / BLOCK_SIZE), BLOCK_SIZE>>>(deviceInputRGBData, 
                     deviceOutputImageData, cdf, cdfmin, imageLength);
  wbTime_stop(GPU, "Equalization");
  cudaDeviceSynchronize();
  
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageLength * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  
  
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceInputImageData);
  cudaFree(deviceInputGrayScaleData);
  cudaFree(hist);
  cudaFree(deviceInputRGBData);
  cudaFree(cdf);
  cudaFree(deviceOutputImageData);
  
  return 0;
}
