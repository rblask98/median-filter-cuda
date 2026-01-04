#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

#define WINDOW_SIZE 3
#define BLOCK 4
#define SIZE_OF_FILTER (WINDOW_SIZE * WINDOW_SIZE)

__host__ void memoryInitCUDA(unsigned char* in_img_data, unsigned char* out_img_data,
                             int imageHeight, int imageWidth, int Channels,
                             long* time, bool isSharedMemory);

__host__ void medianFilterCPU(unsigned char* inputImageKernel, unsigned char* outputImageKernel,
                              int imageHeight, int imageWidth);

__global__ void medianFilterCUDA(unsigned char* inputImageKernel, unsigned char* outputImageKernel,
                                 int imageHeight, int imageWidth);

__global__ void medianFilterCUDAShared(unsigned char* inputImageKernel, unsigned char* outputImageKernel,
                                       int imageHeight, int imageWidth);

int main() {
    Mat img = imread("FESB1CV.bmp", IMREAD_GRAYSCALE);

    Mat resCPU = Mat::zeros(img.size(), CV_8UC1);
    Mat resGPU = Mat::zeros(img.size(), CV_8UC1);
    Mat resGPUShared = Mat::zeros(img.size(), CV_8UC1);

    long timeCPU, timeGPU, timeGPUShared;

    cout << "LOADED IMAGE INFO:\n";
    cout << "Image Height: " << img.rows
         << ", Image Width: " << img.cols
         << ", Image Channels: " << img.channels() << "\n\n";

    cout << "USED: " << WINDOW_SIZE << "x" << WINDOW_SIZE << " median filter\n\n";
    cout << "MEASURED TIME:\n\n";

    auto timeStart = high_resolution_clock::now();
    medianFilterCPU(img.data, resCPU.data, img.rows, img.cols);
    auto timeStop = high_resolution_clock::now();

    timeCPU = duration_cast<microseconds>(timeStop - timeStart).count();
    cout << "CPU time: " << timeCPU << " us = " << (float)timeCPU / 1000 << " ms\n";

    memoryInitCUDA(img.data, resGPU.data, img.rows, img.cols, img.channels(), &timeGPU, false);
    cout << "GPU time: " << timeGPU << " us = " << (float)timeGPU / 1000 << " ms\n";

    memoryInitCUDA(img.data, resGPUShared.data, img.rows, img.cols, img.channels(), &timeGPUShared, true);
    cout << "GPU - Shared Memory time: " << timeGPUShared << " us = " << (float)timeGPUShared / 1000 << " ms\n\n";

    imwrite("Filtered_Image_CPU.bmp", resCPU);
    imwrite("Filtered_Image_GPU.bmp", resGPU);
    imwrite("Filtered_Image_GPU_Shared.bmp", resGPUShared);

    cout << "Input image was successfully filtered by the Median filter!\n";
    system("pause");

    return 0;
}

__host__ void memoryInitCUDA(unsigned char* in_img_data, unsigned char* out_img_data,
                             int imageHeight, int imageWidth, int Channels,
                             long* time, bool isSharedMemory)
{
    unsigned char* dev_in = nullptr;
    unsigned char* dev_out = nullptr;

    dim3 dimBlock(BLOCK, BLOCK);
    dim3 dimGrid((int)ceil((float)imageWidth / BLOCK),
                 (int)ceil((float)imageHeight / BLOCK));

    cudaMalloc((void**)&dev_in, (long long)(imageHeight * imageWidth * Channels));
    cudaMalloc((void**)&dev_out, (long long)(imageHeight * imageWidth * Channels));

    cudaMemcpy(dev_in, in_img_data, (long long)(imageHeight * imageWidth * Channels), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_out, out_img_data, (long long)(imageHeight * imageWidth * Channels), cudaMemcpyHostToDevice);

    auto timeStartInternal = high_resolution_clock::now();

    if (!isSharedMemory)
        medianFilterCUDA<<<dimGrid, dimBlock>>>(dev_in, dev_out, imageHeight, imageWidth);
    else
        medianFilterCUDAShared<<<dimGrid, dimBlock>>>(dev_in, dev_out, imageHeight, imageWidth);

    auto timeStopInternal = high_resolution_clock::now();
    *time = duration_cast<microseconds>(timeStopInternal - timeStartInternal).count();

    cudaMemcpy(in_img_data, dev_in, (long long)(imageHeight * imageWidth * Channels), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_img_data, dev_out, (long long)(imageHeight * imageWidth * Channels), cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_out);
}

__host__ void medianFilterCPU(unsigned char* inputImageKernel, unsigned char* outputImageKernel,
                              int imageHeight, int imageWidth)
{
    unsigned char filterVector[SIZE_OF_FILTER] = {0};

    for (int row = 0; row < imageHeight; row++) {
        for (int col = 0; col < imageWidth; col++) {

            if (row == 0 || col == 0 || row == imageHeight - 1 || col == imageWidth - 1) {
                outputImageKernel[row * imageWidth + col] = 0;
                continue;
            }

            for (int x = 0; x < WINDOW_SIZE; x++)
                for (int y = 0; y < WINDOW_SIZE; y++)
                    filterVector[x * WINDOW_SIZE + y] =
                        inputImageKernel[(row + x - 1) * imageWidth + (col + y - 1)];

            for (int i = 0; i < SIZE_OF_FILTER; i++)
                for (int j = i + 1; j < SIZE_OF_FILTER; j++)
                    if (filterVector[i] > filterVector[j])
                        swap(filterVector[i], filterVector[j]);

            outputImageKernel[row * imageWidth + col] = filterVector[SIZE_OF_FILTER / 2];
        }
    }
}

__global__ void medianFilterCUDA(unsigned char* inputImageKernel, unsigned char* outputImageKernel,
                                 int imageHeight, int imageWidth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned char filterVector[SIZE_OF_FILTER] = {0};

    if (row == 0 || col == 0 || row == imageHeight - 1 || col == imageWidth - 1) {
        outputImageKernel[row * imageWidth + col] = 0;
        return;
    }

    for (int x = 0; x < WINDOW_SIZE; x++)
        for (int y = 0; y < WINDOW_SIZE; y++)
            filterVector[x * WINDOW_SIZE + y] =
                inputImageKernel[(row + x - 1) * imageWidth + (col + y - 1)];

    for (int i = 0; i < SIZE_OF_FILTER; i++)
        for (int j = i + 1; j < SIZE_OF_FILTER; j++)
            if (filterVector[i] > filterVector[j])
                swap(filterVector[i], filterVector[j]);

    outputImageKernel[row * imageWidth + col] = filterVector[SIZE_OF_FILTER / 2];
}

__global__ void medianFilterCUDAShared(unsigned char* inputImageKernel,
                                       unsigned char* outputImageKernel,
                                       int imageHeight, int imageWidth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned char sharedmem[(WINDOW_SIZE + 2)][(WINDOW_SIZE + 2)];

    bool is_x_left   = (threadIdx.x == 0);
    bool is_x_right  = (threadIdx.x == WINDOW_SIZE - 1);
    bool is_y_top    = (threadIdx.y == 0);
    bool is_y_bottom = (threadIdx.y == WINDOW_SIZE - 1);

    if (is_x_left)
        sharedmem[threadIdx.x][threadIdx.y + 1] = 0;
    else if (is_x_right)
        sharedmem[threadIdx.x + 2][threadIdx.y + 1] = 0;

    if (is_y_top) {
        sharedmem[threadIdx.x + 1][threadIdx.y] = 0;

        if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y] = 0;
        else if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y] = 0;
    }
    else if (is_y_bottom) {
        sharedmem[threadIdx.x + 1][threadIdx.y + 2] = 0;

        if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y + 2] = 0;
        else if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y + 2] = 0;
    }

    sharedmem[threadIdx.x + 1][threadIdx.y + 1] =
        inputImageKernel[row * imageWidth + col];

    if (is_x_left && (col > 0))
        sharedmem[threadIdx.x][threadIdx.y + 1] =
            inputImageKernel[row * imageWidth + (col - 1)];
    else if (is_x_right && (col < imageWidth - 1))
        sharedmem[threadIdx.x + 2][threadIdx.y + 1] =
            inputImageKernel[row * imageWidth + (col + 1)];

    if (is_y_top && (row > 0)) {
        sharedmem[threadIdx.x + 1][threadIdx.y] =
            inputImageKernel[(row - 1) * imageWidth + col];

        if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y] =
                inputImageKernel[(row - 1) * imageWidth + (col - 1)];
        else if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y] =
                inputImageKernel[(row - 1) * imageWidth + (col + 1)];
    }
    else if (is_y_bottom && (row < imageHeight - 1)) {
        sharedmem[threadIdx.x + 1][threadIdx.y + 2] =
            inputImageKernel[(row + 1) * imageWidth + col];

        if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y + 2] =
                inputImageKernel[(row + 1) * imageWidth + (col + 1)];
        else if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y + 2] =
                inputImageKernel[(row + 1) * imageWidth + (col - 1)];
    }

    __syncthreads();

    if (SIZE_OF_FILTER == 9)
    {
        unsigned char filterVector[9] = {
            sharedmem[threadIdx.x][threadIdx.y],
            sharedmem[threadIdx.x + 1][threadIdx.y],
            sharedmem[threadIdx.x + 2][threadIdx.y],

            sharedmem[threadIdx.x][threadIdx.y + 1],
            sharedmem[threadIdx.x + 1][threadIdx.y + 1],
            sharedmem[threadIdx.x + 2][threadIdx.y + 1],

            sharedmem[threadIdx.x][threadIdx.y + 2],
            sharedmem[threadIdx.x + 1][threadIdx.y + 2],
            sharedmem[threadIdx.x + 2][threadIdx.y + 2]
        };

        for (int i = 0; i < SIZE_OF_FILTER; i++) {
            for (int j = i + 1; j < SIZE_OF_FILTER; j++) {
                if (filterVector[i] > filterVector[j]) {
                    char tmp = filterVector[i];
                    filterVector[i] = filterVector[j];
                    filterVector[j] = tmp;
                }
            }
        }

        outputImageKernel[row * imageWidth + col] =
            filterVector[SIZE_OF_FILTER / 2];
    }
    else if (SIZE_OF_FILTER == 16)
    {
        unsigned char filterVector[16] = {
            sharedmem[threadIdx.x][threadIdx.y],
            sharedmem[threadIdx.x + 1][threadIdx.y],
            sharedmem[threadIdx.x + 2][threadIdx.y],
            sharedmem[threadIdx.x + 3][threadIdx.y],

            sharedmem[threadIdx.x][threadIdx.y + 1],
            sharedmem[threadIdx.x + 1][threadIdx.y + 1],
            sharedmem[threadIdx.x + 2][threadIdx.y + 1],
            sharedmem[threadIdx.x + 3][threadIdx.y + 1],

            sharedmem[threadIdx.x][threadIdx.y + 2],
            sharedmem[threadIdx.x + 1][threadIdx.y + 2],
            sharedmem[threadIdx.x + 2][threadIdx.y + 2],
            sharedmem[threadIdx.x + 3][threadIdx.y + 2],

            sharedmem[threadIdx.x][threadIdx.y + 3],
            sharedmem[threadIdx.x + 1][threadIdx.y + 3],
            sharedmem[threadIdx.x + 2][threadIdx.y + 3],
            sharedmem[threadIdx.x + 3][threadIdx.y + 3]
        };

        for (int i = 0; i < SIZE_OF_FILTER; i++) {
            for (int j = i + 1; j < SIZE_OF_FILTER; j++) {
                if (filterVector[i] > filterVector[j]) {
                    char tmp = filterVector[i];
                    filterVector[i] = filterVector[j];
                    filterVector[j] = tmp;
                }
            }
        }

        outputImageKernel[row * imageWidth + col] =
            filterVector[SIZE_OF_FILTER / 2];
    }
    else if (SIZE_OF_FILTER == 25)
    {
        unsigned char filterVector[25] = {
            sharedmem[threadIdx.x][threadIdx.y],
            sharedmem[threadIdx.x + 1][threadIdx.y],
            sharedmem[threadIdx.x + 2][threadIdx.y],
            sharedmem[threadIdx.x + 3][threadIdx.y],
            sharedmem[threadIdx.x + 4][threadIdx.y],

            sharedmem[threadIdx.x][threadIdx.y + 1],
            sharedmem[threadIdx.x + 1][threadIdx.y + 1],
            sharedmem[threadIdx.x + 2][threadIdx.y + 1],
            sharedmem[threadIdx.x + 3][threadIdx.y + 1],
            sharedmem[threadIdx.x + 4][threadIdx.y + 1],

            sharedmem[threadIdx.x][threadIdx.y + 2],
            sharedmem[threadIdx.x + 1][threadIdx.y + 2],
            sharedmem[threadIdx.x + 2][threadIdx.y + 2],
            sharedmem[threadIdx.x + 3][threadIdx.y + 2],
            sharedmem[threadIdx.x + 4][threadIdx.y + 2],

            sharedmem[threadIdx.x][threadIdx.y + 3],
            sharedmem[threadIdx.x + 1][threadIdx.y + 3],
            sharedmem[threadIdx.x + 2][threadIdx.y + 3],
            sharedmem[threadIdx.x + 3][threadIdx.y + 3],
            sharedmem[threadIdx.x + 4][threadIdx.y + 3],

            sharedmem[threadIdx.x][threadIdx.y + 4],
            sharedmem[threadIdx.x + 1][threadIdx.y + 4],
            sharedmem[threadIdx.x + 2][threadIdx.y + 4],
            sharedmem[threadIdx.x + 3][threadIdx.y + 4],
            sharedmem[threadIdx.x + 4][threadIdx.y + 4]
        };

        for (int i = 0; i < SIZE_OF_FILTER; i++) {
            for (int j = i + 1; j < SIZE_OF_FILTER; j++) {
                if (filterVector[i] > filterVector[j]) {
                    char tmp = filterVector[i];
                    filterVector[i] = filterVector[j];
                    filterVector[j] = tmp;
                }
            }
        }

        outputImageKernel[row * imageWidth + col] =
            filterVector[SIZE_OF_FILTER / 2];
    }
    else if (SIZE_OF_FILTER == 49)
    {
        unsigned char filterVector[49] = {
            sharedmem[threadIdx.x][threadIdx.y],
            sharedmem[threadIdx.x + 1][threadIdx.y],
            sharedmem[threadIdx.x + 2][threadIdx.y],
            sharedmem[threadIdx.x + 3][threadIdx.y],
            sharedmem[threadIdx.x + 4][threadIdx.y],
            sharedmem[threadIdx.x + 5][threadIdx.y],
            sharedmem[threadIdx.x + 6][threadIdx.y],

            sharedmem[threadIdx.x][threadIdx.y + 1],
            sharedmem[threadIdx.x + 1][threadIdx.y + 1],
            sharedmem[threadIdx.x + 2][threadIdx.y + 1],
            sharedmem[threadIdx.x + 3][threadIdx.y + 1],
            sharedmem[threadIdx.x + 4][threadIdx.y + 1],
            sharedmem[threadIdx.x + 5][threadIdx.y + 1],
            sharedmem[threadIdx.x + 6][threadIdx.y + 1],

            sharedmem[threadIdx.x][threadIdx.y + 2],
            sharedmem[threadIdx.x + 1][threadIdx.y + 2],
            sharedmem[threadIdx.x + 2][threadIdx.y + 2],
            sharedmem[threadIdx.x + 3][threadIdx.y + 2],
            sharedmem[threadIdx.x + 4][threadIdx.y + 2],
            sharedmem[threadIdx.x + 5][threadIdx.y + 2],
            sharedmem[threadIdx.x + 6][threadIdx.y + 2],

            sharedmem[threadIdx.x][threadIdx.y + 3],
            sharedmem[threadIdx.x + 1][threadIdx.y + 3],
            sharedmem[threadIdx.x + 2][threadIdx.y + 3],
            sharedmem[threadIdx.x + 3][threadIdx.y + 3],
            sharedmem[threadIdx.x + 4][threadIdx.y + 3],
            sharedmem[threadIdx.x + 5][threadIdx.y + 3],
            sharedmem[threadIdx.x + 6][threadIdx.y + 3],

            sharedmem[threadIdx.x][threadIdx.y + 4],
            sharedmem[threadIdx.x + 1][threadIdx.y + 4],
            sharedmem[threadIdx.x + 2][threadIdx.y + 4],
            sharedmem[threadIdx.x + 3][threadIdx.y + 4],
            sharedmem[threadIdx.x + 4][threadIdx.y + 4],
            sharedmem[threadIdx.x + 5][threadIdx.y + 4],
            sharedmem[threadIdx.x + 6][threadIdx.y + 4],

            sharedmem[threadIdx.x][threadIdx.y + 5],
            sharedmem[threadIdx.x + 1][threadIdx.y + 5],
            sharedmem[threadIdx.x + 2][threadIdx.y + 5],
            sharedmem[threadIdx.x + 3][threadIdx.y + 5],
            sharedmem[threadIdx.x + 4][threadIdx.y + 5],
            sharedmem[threadIdx.x + 5][threadIdx.y + 5],
            sharedmem[threadIdx.x + 6][threadIdx.y + 5],

            sharedmem[threadIdx.x][threadIdx.y + 6],
            sharedmem[threadIdx.x + 1][threadIdx.y + 6],
            sharedmem[threadIdx.x + 2][threadIdx.y + 6],
            sharedmem[threadIdx.x + 3][threadIdx.y + 6],
            sharedmem[threadIdx.x + 4][threadIdx.y + 6],
            sharedmem[threadIdx.x + 5][threadIdx.y + 6],
            sharedmem[threadIdx.x + 6][threadIdx.y + 6]
        };

        for (int i = 0; i < SIZE_OF_FILTER; i++) {
            for (int j = i + 1; j < SIZE_OF_FILTER; j++) {
                if (filterVector[i] > filterVector[j]) {
                    char tmp = filterVector[i];
                    filterVector[i] = filterVector[j];
                    filterVector[j] = tmp;
                }
            }
        }

        outputImageKernel[row * imageWidth + col] =
            filterVector[SIZE_OF_FILTER / 2];
    }
    else if (SIZE_OF_FILTER == 81)
    {
        unsigned char filterVector[81] = {
            sharedmem[threadIdx.x][threadIdx.y],
            sharedmem[threadIdx.x + 1][threadIdx.y],
            sharedmem[threadIdx.x + 2][threadIdx.y],
            sharedmem[threadIdx.x + 3][threadIdx.y],
            sharedmem[threadIdx.x + 4][threadIdx.y],
            sharedmem[threadIdx.x + 5][threadIdx.y],
            sharedmem[threadIdx.x + 6][threadIdx.y],
            sharedmem[threadIdx.x + 7][threadIdx.y],
            sharedmem[threadIdx.x + 8][threadIdx.y],

            sharedmem[threadIdx.x][threadIdx.y + 1],
            sharedmem[threadIdx.x + 1][threadIdx.y + 1],
            sharedmem[threadIdx.x + 2][threadIdx.y + 1],
            sharedmem[threadIdx.x + 3][threadIdx.y + 1],
            sharedmem[threadIdx.x + 4][threadIdx.y + 1],
            sharedmem[threadIdx.x + 5][threadIdx.y + 1],
            sharedmem[threadIdx.x + 6][threadIdx.y + 1],
            sharedmem[threadIdx.x + 7][threadIdx.y + 1],
            sharedmem[threadIdx.x + 8][threadIdx.y + 1],

            sharedmem[threadIdx.x][threadIdx.y + 2],
            sharedmem[threadIdx.x + 1][threadIdx.y + 2],
            sharedmem[threadIdx.x + 2][threadIdx.y + 2],
            sharedmem[threadIdx.x + 3][threadIdx.y + 2],
            sharedmem[threadIdx.x + 4][threadIdx.y + 2],
            sharedmem[threadIdx.x + 5][threadIdx.y + 2],
            sharedmem[threadIdx.x + 6][threadIdx.y + 2],
            sharedmem[threadIdx.x + 7][threadIdx.y + 2],
            sharedmem[threadIdx.x + 8][threadIdx.y + 2],

            sharedmem[threadIdx.x][threadIdx.y + 3],
            sharedmem[threadIdx.x + 1][threadIdx.y + 3],
            sharedmem[threadIdx.x + 2][threadIdx.y + 3],
            sharedmem[threadIdx.x + 3][threadIdx.y + 3],
            sharedmem[threadIdx.x + 4][threadIdx.y + 3],
            sharedmem[threadIdx.x + 5][threadIdx.y + 3],
            sharedmem[threadIdx.x + 6][threadIdx.y + 3],
            sharedmem[threadIdx.x + 7][threadIdx.y + 3],
            sharedmem[threadIdx.x + 8][threadIdx.y + 3],

            sharedmem[threadIdx.x][threadIdx.y + 4],
            sharedmem[threadIdx.x + 1][threadIdx.y + 4],
            sharedmem[threadIdx.x + 2][threadIdx.y + 4],
            sharedmem[threadIdx.x + 3][threadIdx.y + 4],
            sharedmem[threadIdx.x + 4][threadIdx.y + 4],
            sharedmem[threadIdx.x + 5][threadIdx.y + 4],
            sharedmem[threadIdx.x + 6][threadIdx.y + 4],
            sharedmem[threadIdx.x + 7][threadIdx.y + 4],
            sharedmem[threadIdx.x + 8][threadIdx.y + 4],

            sharedmem[threadIdx.x][threadIdx.y + 5],
            sharedmem[threadIdx.x + 1][threadIdx.y + 5],
            sharedmem[threadIdx.x + 2][threadIdx.y + 5],
            sharedmem[threadIdx.x + 3][threadIdx.y + 5],
            sharedmem[threadIdx.x + 4][threadIdx.y + 5],
            sharedmem[threadIdx.x + 5][threadIdx.y + 5],
            sharedmem[threadIdx.x + 6][threadIdx.y + 5],
            sharedmem[threadIdx.x + 7][threadIdx.y + 5],
            sharedmem[threadIdx.x + 8][threadIdx.y + 5],

            sharedmem[threadIdx.x][threadIdx.y + 6],
            sharedmem[threadIdx.x + 1][threadIdx.y + 6],
            sharedmem[threadIdx.x + 2][threadIdx.y + 6],
            sharedmem[threadIdx.x + 3][threadIdx.y + 6],
            sharedmem[threadIdx.x + 4][threadIdx.y + 6],
            sharedmem[threadIdx.x + 5][threadIdx.y + 6],
            sharedmem[threadIdx.x + 6][threadIdx.y + 6],
            sharedmem[threadIdx.x + 7][threadIdx.y + 6],
            sharedmem[threadIdx.x + 8][threadIdx.y + 6],

            sharedmem[threadIdx.x][threadIdx.y + 7],
            sharedmem[threadIdx.x + 1][threadIdx.y + 7],
            sharedmem[threadIdx.x + 2][threadIdx.y + 7],
            sharedmem[threadIdx.x + 3][threadIdx.y + 7],
            sharedmem[threadIdx.x + 4][threadIdx.y + 7],
            sharedmem[threadIdx.x + 5][threadIdx.y + 7],
            sharedmem[threadIdx.x + 6][threadIdx.y + 7],
            sharedmem[threadIdx.x + 7][threadIdx.y + 7],
            sharedmem[threadIdx.x + 8][threadIdx.y + 7],

            sharedmem[threadIdx.x][threadIdx.y + 8],
            sharedmem[threadIdx.x + 1][threadIdx.y + 8],
            sharedmem[threadIdx.x + 2][threadIdx.y + 8],
            sharedmem[threadIdx.x + 3][threadIdx.y + 8],
            sharedmem[threadIdx.x + 4][threadIdx.y + 8],
            sharedmem[threadIdx.x + 5][threadIdx.y + 8],
            sharedmem[threadIdx.x + 6][threadIdx.y + 8],
            sharedmem[threadIdx.x + 7][threadIdx.y + 8],
            sharedmem[threadIdx.x + 8][threadIdx.y + 8]
        };

        for (int i = 0; i < SIZE_OF_FILTER; i++) {
            for (int j = i + 1; j < SIZE_OF_FILTER; j++) {
                if (filterVector[i] > filterVector[j]) {
                    char tmp = filterVector[i];
                    filterVector[i] = filterVector[j];
                    filterVector[j] = tmp;
                }
            }
        }

        outputImageKernel[row * imageWidth + col] =
            filterVector[SIZE_OF_FILTER / 2];
    }
    else {
        // no-op
    }
}
