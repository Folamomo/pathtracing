//
// Created by igor on 26.03.2021.
//

#include "Screen.cuh"
#include "ConvMask.cuh"

std::deque<Screen::ScreenImp> Screen::implementations;

template <typename From, typename To>
__global__ void cast_kernel(From* source, To* destination, size_t n){
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= n ) return;
    destination[x] = (To)source[x];
}



Screen::Screen(unsigned int x, unsigned int y): sizeX(x), sizeY(y) {
    implementations.emplace_back(x, y);
    image = implementations.back().image;
    d_image = implementations.back().d_image;
    d_imageC = implementations.back().d_imageC;
}

void Screen::copy(cudaMemcpyKind dir) {
    cudaError_t error = cudaSuccess;
    if (dir == cudaMemcpyHostToDevice){
         error = cudaMemcpy(d_imageC, image, sizeX * sizeY * sizeof(ColorC), dir);
    } else if (dir == cudaMemcpyDeviceToHost) {
        error = cudaMemcpy(image, d_imageC, sizeX * sizeY * sizeof(ColorC), dir);
    }
    if (error) throw std::runtime_error(cudaGetErrorString(error));
}



void Screen::save(const char *path) const {
    /* PPM header format:
     *
     * P6
     * X Y 255
     *
     * P6 is a magic number denoting file format (24 bit color, binary)
     * X, Y are image dimensions
     * 255 is max value of every color (independent from word length)
     */
    FILE *f = fopen(path, "w");
    fprintf(f, "P6\n%d %d 255\n", sizeX, sizeY);
    fclose(f);

    //open file again for binary data
    f = fopen(path, "ab");
    fwrite(image, sizeof(char), sizeX * sizeY * sizeof(ColorF), f);
    fclose(f);
}

__device__ ColorF* Screen::operator[](size_t row) {
    return &d_image[row * sizeX];
}


void Screen::removeAll() {
    implementations.clear();
}

void Screen::cast() {
    cast_kernel<<<(sizeX * sizeY + 512 - 1)/512, 512>>>(d_image, d_imageC, sizeX * sizeY);
}

void Screen::copyAndSave(const char *path) {
    cast();
    copy(cudaMemcpyDeviceToHost);
    save(path);
}


Screen::ScreenImp::ScreenImp(unsigned int sizeX, unsigned int sizeY) : sizeX(sizeX), sizeY(sizeY) {
    image = (ColorC*)malloc(sizeX * sizeY * sizeof(ColorC));
    if (image==nullptr) throw std::runtime_error("Malloc failed for image");
    d_image = nullptr;
    cudaError_t error = cudaMalloc(&d_image, sizeX * sizeY * sizeof(ColorF));
    if (error) throw std::runtime_error(cudaGetErrorString(error));
    if (d_image== nullptr) throw std::runtime_error("cudaMalloc is broken");
    d_imageC = nullptr;
    error = cudaMalloc(&d_imageC, sizeX * sizeY * sizeof(ColorC));
    if (error) throw std::runtime_error(cudaGetErrorString(error));
    if (d_image== nullptr) throw std::runtime_error("cudaMalloc is broken");
}

Screen::ScreenImp::~ScreenImp() {
    free(image);
    cudaFree(d_image);
    cudaFree(d_imageC);
}
