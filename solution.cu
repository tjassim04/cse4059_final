#include <gputk.h>

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define FILTER_DIM 5
#define FILTER_RADIUS FILTER_DIM / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + FILTER_DIM - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))

//@@ INSERT CODE HERE

__constant__ float deviceMaskData[FILTER_DIM*FILTER_DIM];

__global__ void convolution(float *imageData, float *outputData, int channels, int width, int height) {

    extern __shared__ float tile[];

    int startRow = blockIdx.y * TILE_WIDTH - FILTER_RADIUS;
    int startCol = blockIdx.x * TILE_WIDTH - FILTER_RADIUS;

    const int tileWidth = w;
    const int tileHeight = w;

    for (int r = threadIdx.y; r < tileHeight; r += blockDim.y) {
        for (int c = threadIdx.x; c < tileWidth; c += blockDim.x) {
            int globalRow = startRow + r;
            int globalCol = startCol + c;

            bool inside = (globalRow >= 0 && globalRow < height && globalCol >= 0 && globalCol < width);

            int tileBaseIdx = (r * tileWidth + c) * channels;
            for (int ch = 0; ch < channels; ++ch) {
                float val = 0.0f;
                if (inside) {
                    int globalIdx = (globalRow * width + globalCol) * channels + ch;
                    val = imageData[globalIdx];
                }
                tile[tileBaseIdx + ch] = val;
            }
        }
    }

    __syncthreads();

    int outRow = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int outCol = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (outRow < height && outCol < width) {
        for (int ch = 0; ch < channels; ++ch) {
            float sum = 0.0f;

            #pragma unroll
            for (int fRow = 0; fRow < FILTER_DIM; ++fRow) {
                #pragma unroll
                for (int fCol = 0; fCol < FILTER_DIM; ++fCol) {
                    int tileRow = threadIdx.y + fRow;
                    int tileCol = threadIdx.x + fCol;
                    int tileIdx = (tileRow * tileWidth + tileCol) * channels + ch;

                    float maskVal = deviceMaskData[fRow * FILTER_DIM + fCol];
                    sum += maskVal * tile[tileIdx];
                }
            }

            int outIdx = (outRow * width + outCol) * channels + ch;
            outputData[outIdx] = clamp(sum);
        }
    }
}

int main(int argc, char *argv[]) {
  gpuTKArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  gpuTKImage_t inputImage;
  gpuTKImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
//  float *deviceMaskData;

  arg = gpuTKArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = gpuTKArg_getInputFile(arg, 0);
  inputMaskFile  = gpuTKArg_getInputFile(arg, 1);

  inputImage   = gpuTKImport(inputImageFile);
  hostMaskData = (float *)gpuTKImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == 5);    /* mask height is fixed to 5 in this mp */
  assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

  imageWidth    = gpuTKImage_getWidth(inputImage);
  imageHeight   = gpuTKImage_getHeight(inputImage);
  imageChannels = gpuTKImage_getChannels(inputImage);

  outputImage = gpuTKImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = gpuTKImage_getData(inputImage);
  hostOutputImageData = gpuTKImage_getData(outputImage);

  gpuTKTime_start(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE

  cudaMalloc((void**) &deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
 // cudaMalloc((void**) &deviceMaskData, maskRows*maskColumns*sizeof(float));
  cudaMalloc((void**) &deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));

  gpuTKTime_stop(GPU, "Doing GPU memory allocation");

  gpuTKTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE

  cudaMemcpy(deviceInputImageData, hostInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceMaskData, hostMaskData, maskRows*maskColumns*sizeof(float));


  gpuTKTime_stop(Copy, "Copying data to the GPU");

  gpuTKTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE

  int smSize = w * w * imageChannels * sizeof(float);

  dim3 dimBlock(16,16);
  dim3 dimGrid((imageWidth+dimBlock.x - 1)/dimBlock.x, (imageHeight + dimBlock.y -1)/dimBlock.y);

  convolution<<<dimGrid, dimBlock, smSize>>>(deviceInputImageData,  deviceOutputImageData, imageChannels, imageWidth, imageHeight);

  gpuTKTime_stop(Compute, "Doing the computation on the GPU");

  gpuTKTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  gpuTKTime_stop(Copy, "Copying data from the GPU");

  gpuTKTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  gpuTKSolution(arg, outputImage);

  //@@ Insert code here

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  free(hostMaskData);
  gpuTKImage_delete(outputImage);
  gpuTKImage_delete(inputImage);

  return 0;
}
