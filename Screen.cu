//
// Created by igor on 26.03.2021.
//

#include "Screen.cuh"

Screen::Screen(unsigned int x, unsigned int y) : sizeX(x), sizeY(y) {
    image = (Color*)malloc(sizeX*sizeY*sizeof(Color));
    if (image==nullptr) throw std::runtime_error("Malloc failed for image");
    d_image = nullptr;
    cudaError_t error = cudaMalloc(&d_image, sizeX * sizeY * sizeof(Color));
    if (error) throw std::runtime_error(cudaGetErrorString(error));
    if (d_image== nullptr) throw std::runtime_error("cudaMalloc is broken");
}



void Screen::copy(cudaMemcpyKind dir) {
    cudaError_t error = cudaMemcpy(image, d_image, sizeX * sizeY * sizeof(Color), dir);
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
    fwrite(image, sizeof(char), sizeX * sizeY * sizeof(Color), f);
    fclose(f);
}

Screen::~Screen() {
    free(image);
    cudaFree(d_image);
}

Screen::ScreenRef Screen::makeRef() {
    return {*this};
}
