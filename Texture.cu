//
// Created by igor on 03.06.2021.
//

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include "Texture.cuh"

Texture::Texture(const char *path) {
    std::ifstream inp(path, std::ios::in | std::ios::binary);
    if (inp.is_open()) {
        std::string line;
        std::getline(inp, line);
        if (line != "P6") {
            std::cout << "Error. Unrecognized file format." << std::endl;
            return;
        }
        std::getline(inp, line);
        while (line[0] == '#') {
            std::getline(inp, line);
        }
        std::stringstream dimensions(line);

        try {
            dimensions >> x;
            dimensions >> y;
        } catch (std::exception &e) {
            std::cout << "Header file format error. " << e.what() << std::endl;
            return;
        }

        std::getline(inp, line);
        std::stringstream max_val(line);
        try {
            max_val >> max;
        } catch (std::exception &e) {
            std::cout << "Header file format error. " << e.what() << std::endl;
            return;
        }

        uint size = x*y;

        img = (ColorF*)malloc(sizeof(ColorF) * size);

        char aux;
        for (unsigned int i = 0; i < size; ++i) {
            inp.read(&aux, 1);
            img[i].r = (float)(unsigned char) aux;
            inp.read(&aux, 1);
            img[i].g = (float)(unsigned char) aux;
            inp.read(&aux, 1);
            img[i].b = (float)(unsigned char) aux;
        }
    } else {
        std::cout << "Error. Unable to open " << path << std::endl;
    }
    inp.close();
}

Texture::~Texture() {
//    free(img);
}
