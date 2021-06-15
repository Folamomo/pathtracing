//
// Created by igor on 17.05.2021.
//

#define BLOCK_X 32
#define BLOCK_Y 32
#define EPSILON 0.0000001
#define BATCH_SIZE 1024
#define raysPerPixel 100
#include "Renderer.cuh"
#include <curand_kernel.h>
#include <curand.h>


__global__ void memoryDumpIndices(unsigned int* from, unsigned int* to){
    while(from < to){
        printf("%u ", *(from++));
        printf("%u ", *(from++));
        printf("%u\n", *(from++));
    }
}

__global__ void memoryDumpPositions(Vector3* from, Vector3* to){
    while(from < to){
        printf("%f %f %f\n", from->x, from->y, from->z);
        ++from;
    }
}

__global__ void memoryDumpMaterials(Material* from, Material* to){
    while(from < to){
        printf("%f %f %f\n", from->emit_color.r, from->emit_color.g, from->emit_color.b);
        ++from;
    }
}

__global__ void memoryDumpMaterialIndices(unsigned int* from, unsigned int* to){
    while(from < to){
        printf("%u\n", *(from++));
    }
}



__device__ Vector3 directionFromXY(const Renderer::RendererConstants& constants, int x, int y, curandState* state){
    return (constants.topLeft +
    (x + curand_uniform (state)) * constants.pixelDx +
    (y + curand_uniform (state)) * constants.pixelDy)
    .normalize();
}

__device__ int cast_ray(const Vector3 direction,
                         const Vector3 origin,
                         const Vector3* as,
                         const Vector3* edges1,
                         const Vector3* edges2,
                         const unsigned int triangles_n,
                         float* closest_distance){
    *closest_distance = 10000000.0f;
    int closest_id = -1;
    for (int t = 0; t < triangles_n; ++t) {
        Vector3 a = as[t];
        Vector3 edge1 = edges1[t];
        Vector3 edge2 = edges2[t];
        Vector3 h = direction.cross(edge2);
        float a_ = edge1.dot(h);
        // The ray is parallel to this triangle.
        if (a_ > -EPSILON && a_ < EPSILON) continue;
        float f = 1.0f / a_;
        Vector3 s = origin - a;
        float u = f * s.dot(h);
        Vector3 q = s.cross(edge1);
        float v = f * direction.dot(q);
        // The ray intercepts the plane outside the triangle;
        if (v < 0.0 || u + v > 1.0 || u < 0.0 || u > 1.0) continue;
        // At this stage we can compute t to find out where the intersection point is on the line.
        float distance = f * edge2.dot(q);
        if (distance > EPSILON && distance < *closest_distance) {
            closest_id = t;
            *closest_distance = distance;
        }
    }
    return closest_id;
}

__global__ void random_init(curandState *dev_random){
    unsigned int id = threadIdx.x + threadIdx.y * blockDim.x;
    curand_init(id, id, 0, &dev_random[id]);
}

__global__ void recalculateVertices(Vector3* vertices, Matrix4 transform, unsigned int n){
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n) vertices[id] = transform * vertices[id];
}

__global__ void recalculateIndices(unsigned int* indices, unsigned int* material_index,
                                   unsigned int material, unsigned int vertices_offset, unsigned int n){
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n) {
        indices[id * 3] += vertices_offset;
        indices[id * 3 + 1] += vertices_offset;
        indices[id * 3 + 2] += vertices_offset;
        material_index[id] = material;
    }
}

__global__  void calculatePixel(
        Renderer::RendererConstants constants
        ){
    unsigned int x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int y = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int id = threadIdx.x + threadIdx.y * blockDim.x;


    extern __shared__ Vector3 triangles[];
    Vector3* as = &triangles[0];
    Vector3* edges1 = &triangles[constants.n_triangles];
    Vector3* edges2 = &triangles[constants.n_triangles * 2];

    for(unsigned int triangle_id = id; triangle_id < constants.n_triangles; triangle_id += blockDim.x * blockDim.y){
        as[triangle_id] = constants.vertices[constants.indices[triangle_id * 3]];
        edges1[triangle_id] = constants.vertices[constants.indices[triangle_id * 3 + 1]] - as[triangle_id];
        edges2[triangle_id] = constants.vertices[constants.indices[triangle_id * 3 + 2]] - as[triangle_id];
    }
    __syncthreads();
    ColorF final_color{};
    for(int i = 0; i < raysPerPixel; ++i) {
        float closest_distance;
        Vector3 ray = directionFromXY(constants, x, y, &constants.dev_random[id]);
        int closest_id = cast_ray(ray, constants.origin, as, edges1, edges2, constants.n_triangles, &closest_distance);
        if (closest_id >= 0) {
            const Material& material = constants.materials[constants.material_index[closest_id]];
            //emission color
            final_color += material.emit_color * material.emit;
            //specular color
            Vector3 normal = edges1[closest_id].cross(edges2[closest_id]).normalize();
            Vector3 reflection_dir = -2.0f * ray.dot(normal) * normal + ray;
            Vector3 reflection_origin = constants.origin + closest_distance * ray;
            int reflection_id = cast_ray(reflection_dir, reflection_origin, as, edges1, edges2, constants.n_triangles, &closest_distance);
            if (reflection_id >= 0) {
                const Material &ref_material = constants.materials[constants.material_index[reflection_id]];
                final_color += ref_material.emit_color * material.specular * ref_material.emit;
            } else {
                final_color += ColorF{0, 0, 255.0f * (0.5f + asin(reflection_dir.y) / 3.1415f)} * material.specular;
            }
        } else {
          final_color +=  ColorF{0, 0, 255.0f * (0.5f + asin(ray.y)/3.1415f)};

        }
    }
    constants.d_image[y * constants.sizeX + x] = final_color * (1.0f/raysPerPixel);
}


void Renderer::render() {
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess){
        if (error) throw std::runtime_error(cudaGetErrorString(error));
    }
    calculatePixel<<<dim3((constants.sizeX + BLOCK_X - 1)/BLOCK_X,
                          (constants.sizeY + BLOCK_Y - 1)/BLOCK_Y),
                          dim3(BLOCK_X, BLOCK_Y), constants.n_triangles * sizeof(Vector3) * 3>>>
            (constants);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess){
        if (error) throw std::runtime_error(cudaGetErrorString(error));
    }
}

Renderer::Renderer() {
    // Initiate  random number generator states
    cudaMalloc((void**)&constants.dev_random, BLOCK_X * BLOCK_Y * sizeof(curandState));
    random_init<<<1, dim3(BLOCK_X, BLOCK_Y)>>>(constants.dev_random);
}

Renderer::~Renderer() {

    // free resources from old scene
    cudaFree(constants.materials);
    cudaFree(constants.indices);
    cudaFree(constants.vertices);
    cudaFree(constants.material_index);

    // free dev_random
    cudaFree(&constants.dev_random);
}

void Renderer::uploadScene(Scene &scene) {

    cudaError error = cudaGetLastError();
    if (error != cudaSuccess){
        if (error) throw std::runtime_error(cudaGetErrorString(error));
    }

    //Count how much memory is required
    constants.n_triangles = 0;
    unsigned int n_vertices = 0;
    unsigned int n_materials = 0;
    for (const Mesh& mesh: scene.meshes){
        constants.n_triangles += mesh.indices.size() / 3;
        n_vertices += mesh.positions.size();
        n_materials++;
    }


    //Allocate memory for all meshes
    cudaMalloc(&constants.indices, constants.n_triangles * 3 * sizeof(unsigned int));
    cudaMalloc(&constants.vertices, n_vertices * sizeof(Vector3));
    cudaMalloc(&constants.material_index, constants.n_triangles * sizeof(unsigned int));
    cudaMalloc(&constants.materials, n_materials * sizeof(Material));

    //Recalculate meshes on GPU
    unsigned int material_offset = 0;
    unsigned int vertex_offset = 0;
    unsigned int index_offset = 0;
    error = cudaGetLastError();
    if (error != cudaSuccess){
        if (error) throw std::runtime_error(cudaGetErrorString(error));
    }
    for (const Mesh& mesh: scene.meshes){
        //copy data
        cudaMemcpy(constants.vertices + vertex_offset, mesh.positions.data(), mesh.positions.size() * sizeof(Vector3), cudaMemcpyHostToDevice);
        cudaMemcpy(constants.indices + index_offset, mesh.indices.data(), mesh.indices.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(constants.materials + material_offset, &mesh.material, sizeof(Material), cudaMemcpyHostToDevice);
        //call recalculation kernels
        recalculateVertices<<<(mesh.positions.size() - 1 + BLOCK_X) / BLOCK_X, BLOCK_X>>>(constants.vertices + vertex_offset, mesh.transform, mesh.positions.size());
        recalculateIndices<<<(mesh.indices.size()/3 - 1 + BLOCK_X) / BLOCK_X, BLOCK_X>>>(constants.indices + index_offset, constants.material_index + index_offset / 3,
                                                                                         material_offset, vertex_offset, mesh.indices.size());
        material_offset++;
        vertex_offset += mesh.positions.size();
        index_offset +=  mesh.indices.size();
    }
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess){
        if (error) throw std::runtime_error(cudaGetErrorString(error));
    }

    constants.n_triangles = index_offset / 3;

    memoryDumpIndices<<<1, 1>>>(constants.indices, constants.indices + index_offset);
    memoryDumpPositions<<<1, 1>>>( constants.vertices, constants.vertices + vertex_offset);
    memoryDumpMaterials<<<1, 1>>>(constants.materials, constants.materials + material_offset);
    memoryDumpMaterialIndices<<<1, 1>>>(constants.material_index, constants.material_index + index_offset / 3);
}

void Renderer::uploadCamera(Camera &camera) {
    constants.topLeft = camera.topLeft;
    constants.pixelDx = camera.pixelDx;
    constants.pixelDy = camera.pixelDy;
    constants.origin  = camera.origin;
}

void Renderer::uploadScreen(Screen &screen) {
    constants.sizeX = screen.sizeX;
    constants.sizeY = screen.sizeY;
    //No allocations here since Screen class manages its own memory
    constants.d_image = screen.d_image;
}
