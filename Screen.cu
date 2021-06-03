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


__global__ void convolution(Screen screen, ConvMask mask){
    __shared__ ColorF old[BLOCK_X * BLOCK_Y];
    uint x = threadIdx.x - 1 + blockIdx.x * (blockDim.x - 2);
    uint y = threadIdx.y - 1 + blockIdx.y * (blockDim.y - 2);

    if (x >= screen.sizeX || y >= screen.sizeY) return;

    x = x < 0 ? 0 : x >= screen.sizeX ? screen.sizeX - 1 : x;
    y = y < 0 ? 0 : y >= screen.sizeY ? screen.sizeY - 1 : y;
    unsigned int shared_pos = threadIdx.x + threadIdx.y * BLOCK_X;
    old[shared_pos] = screen.d_image[y * screen.sizeX + x];

    if (threadIdx.x == 0 || threadIdx.y == 0 || threadIdx.x == BLOCK_X - 1 || threadIdx.y == BLOCK_Y - 1 ) return;

    __syncthreads();

    ColorF result = old[shared_pos - 1 - BLOCK_X] * mask[0][0] +
                    old[shared_pos     - BLOCK_X] * mask[0][1] +
                    old[shared_pos + 1 - BLOCK_X] * mask[0][2] +
                    old[shared_pos - 1          ] * mask[1][0] +
                    old[shared_pos              ] * mask[1][1] +
                    old[shared_pos + 1          ] * mask[1][2] +
                    old[shared_pos - 1 + BLOCK_X] * mask[2][0] +
                    old[shared_pos     + BLOCK_X] * mask[2][1] +
                    old[shared_pos + 1 + BLOCK_X] * mask[2][2];


    screen.d_imageC[y * screen.sizeX + x] = {(unsigned char)result.r, (unsigned char)result.g, (unsigned char)result.b};

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

__device__ __host__ Vector3 Screen::getRandomRayInPixel(unsigned int x, unsigned int y) {
    return Vector3();
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
