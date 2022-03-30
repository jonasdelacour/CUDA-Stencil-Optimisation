#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <numeric>
#include <cassert>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define INLINE __device__ __forceinline__
typedef float real_t;
typedef float2 real2_t;

namespace cg = cooperative_groups;

/** Helper function for printing out device failures such that silent GPU failures dont go unnoticed. **/
void printLastCudaError(std::string message = ""){
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        std::cout << "\n" << message << " :\t";
        std::cout << cudaGetErrorString(error);
        printf("\n");
    }
}

class Sim_Configuration {
public:
    int iter = 5000;  // Number of iterations
    int data_period = 100;  // how often to save coordinate to file
    int size = 512;
    int block_x_size = 32;
    int block_y_size = 8;
    int gridx = 16;
    int gridy = 0;
    std::string filename = "cuda.data";   // name of the output file with history

    Sim_Configuration(std::vector <std::string> argument){
        for (long unsigned int i = 1; i<argument.size() ; i += 2){
            std::string arg = argument[i];
            if(arg=="-h"){ // Write help
                std::cout << "./par --iter <number of iterations> --dt <time step>"
                          << "--dx <x grid size> --dy <y grid size>"
                          << "--fperiod <iterations between each save> --out <name of output file>\n";
                exit(0);
            } else if (i == argument.size() - 1) {
                throw std::invalid_argument("The last argument (" + arg +") must have a value");
            } else if(arg=="--iter"){
                if ((iter = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("iter most be a positive integer (e.g. -iter 1000)");
            } else if (arg=="--size"){
                if ((size = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("size most be a positive integer (e.g. --size 512)");
            } else if (arg=="--blockx"){
                if ((block_x_size = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("block-y-size most be a positive integer (e.g. --block-size 32)");
            } else if (arg=="--blocky"){
                if ((block_y_size = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("block-y-size most be a positive integer (e.g. --block-size 32)");
            } else if (arg=="--gridx"){
                if ((gridx = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("gridx most be a positive integer (e.g. --gridx 10)");
            } else if (arg=="--gridy"){
                if ((gridy = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("gridy most be a positive integer (e.g. --gridy 10)");
            } else if(arg=="--fperiod"){
                if ((data_period = std::stoi(argument[i+1])) < 0) 
                    throw std::invalid_argument("fperiod most be a positive integer (e.g. -fperiod 100)");
            } else if(arg=="--out"){
                filename = argument[i+1];
            } else{
                std::cout << "---> error: the argument type is not recognized \n";
            }
        }
    }
};



/** Representation of a Chemicals world including ghost lines, which is a "1-cell padding" of rows and columns
 *  around the world. These ghost lines is a technique to implement periodic boundary conditions. */
class Chemicals {
public:
    size_t NX, NY; // The shape of the Chemicals world including ghost lines.
    real_t D_p, D_q, C, K; 
    real_t dx, dy, dt;
    real_t* P; real_t* Q;

    std::vector<void*> h_buffers;
    real_t* d_Q; real_t* d_P;
    real_t* d_dQdt; real_t* d_dPdt;
    std::vector<void**> d_buffers = {(void**)&d_P, (void**)&d_Q, (void**)&d_dQdt, (void**)&d_dPdt}; 
    Chemicals(size_t NX, size_t NY, real_t K = 9.0, real_t C = 4.5, real_t D_p = 1.0, real_t D_q = 8.0) : 
       NX(NX), NY(NY), K(K), C(C), D_p(D_p), D_q(D_q)
    {   
        P = (real_t*)calloc(NX*NY, sizeof(real_t));
        Q = (real_t*)calloc(NX*NY, sizeof(real_t));

        for (size_t i = NY/4; i < NY - NY/4 ; ++i) 
        for (size_t j = NX/4; j < NX - NX/4 ; ++j) {
            Q[i*NX + j] =  (K / C) + 0.2;
            P[i*NX + j] =  C + 0.1;
        }

        //Largest approximate timestep determined empirically from function fitting.
        this->dt = 0.02 * std::pow(real_t(std::min(NX,NY))/40.0,-2);
        this->dx = 40/real_t(NX); this-> dy = 40.0/real_t(NY);
        h_buffers = {P,Q};
        for (int i = 0; i<d_buffers.size(); i++)        cudaMalloc(d_buffers[i], sizeof(real_t)*NX*NY);
        for (int i = 0; i < h_buffers.size(); i++)   cudaMemcpy(*d_buffers[i], h_buffers[i], sizeof(real_t)*NX*NY, cudaMemcpyHostToDevice);
    }

    void copyIn(){
        for (int i = 0; i<h_buffers.size(); i++) cudaMemcpy(*d_buffers[i], h_buffers[i], sizeof(real_t)*NX*NY, cudaMemcpyHostToDevice);
    }
    
    void copyOut(){
        for (int i = 0; i<h_buffers.size(); i++) cudaMemcpy(h_buffers[i], *d_buffers[i], sizeof(real_t)*NX*NY, cudaMemcpyDeviceToHost);
    }

    ~Chemicals(){
        for (int i = 0; i<d_buffers.size(); i++) cudaFree(*d_buffers[i]);
    }
};

void to_file(const Chemicals& c, real_t* history, const size_t frames, const std::string &filename){
    std::ofstream file(filename);
    file.write((const char*)(history), sizeof(real_t)*c.NX*c.NY*frames);
}

__global__
void exchange_horizontal(Chemicals c){
    real_t* P = c.d_P; real_t* Q = c.d_Q;
    for (size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y)*blockDim.x*gridDim.x ; tid < c.NX; tid += gridDim.x*blockDim.x*gridDim.y*blockDim.y)
    {
        P[tid] = P[tid + 2 * c.NX];                              //Top ghost cells = Top cells - 1
        Q[tid] = Q[tid + 2 * c.NX];                              //Top ghost cells = Top cells - 1
        P[tid + c.NX * (c.NY - 1)] = P[tid + c.NX * (c.NY - 3)]; //Bottom ghost cells = Bottom cells + 1
        Q[tid + c.NX * (c.NY - 1)] = Q[tid + c.NX * (c.NY - 3)]; //Bottom ghost cells = Bottom cells + 1
    }
}

__global__
void exchange_vertical(Chemicals c){
    real_t* P = c.d_P; real_t* Q = c.d_Q;
    for (size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y)*blockDim.x*gridDim.x ; tid < c.NY; tid += gridDim.x*blockDim.x*gridDim.y*blockDim.y)
    {    
        P[tid*c.NX] = P[tid*c.NX + 2];                          //Left ghost cells = Left cells + 1
        Q[tid*c.NX] = Q[tid*c.NX + 2];                          //Left ghost cells = Left cells + 1
        P[tid*c.NX + c.NX - 1] = P[tid*c.NX + c.NX - 3];        //Right ghost cells = Right cells - 1
        Q[tid*c.NX + c.NX - 1] = Q[tid*c.NX + c.NX - 3];        //Right ghost cells = Right cells - 1
    }
}

INLINE
real_t dPdt(Chemicals& c, const size_t tid){
    real_t* P = c.d_P; real_t* Q = c.d_Q;
    return c.D_p * ( P[tid - 1] + P[tid + 1] + P[tid + c.NX] + P[tid - c.NX] - real_t(4.0)*P[tid]) / (c.dx * c.dy) + c.C + P[tid]*P[tid]*Q[tid] - (c.K + real_t(1.0)) * P[tid];
}

INLINE
real_t dQdt(Chemicals& c, const size_t tid){
    real_t* P = c.d_P; real_t* Q = c.d_Q;
    return c.D_q * ( Q[tid - 1] + Q[tid + 1] + Q[tid + c.NX] + Q[tid - c.NX] - real_t(4.0)*Q[tid]) / (c.dx * c.dy) + P[tid] * c.K - P[tid]*P[tid]*Q[tid];
}

__global__
void derivative(Chemicals c){
    auto dimx = gridDim.x*blockDim.x;
    auto dimy = gridDim.y*blockDim.y;
    for (auto gtidy = threadIdx.y + blockIdx.y * blockDim.y + 1; gtidy < c.NY -1; gtidy += dimy)
    for (auto gtidx = threadIdx.x + blockIdx.x * blockDim.x + 1; gtidx < c.NX -1; gtidx += dimx)
    { 
        auto tid = gtidx + gtidy*c.NX;
        c.d_dQdt[tid] = dQdt(c, tid);
        c.d_dPdt[tid] = dPdt(c, tid);
    }
}
__global__
void integrate(Chemicals c){
    real_t* P = c.d_P; real_t* Q = c.d_Q;
    auto dimx = gridDim.x*blockDim.x;
    auto dimy = gridDim.y*blockDim.y;
    for (auto gtidy = threadIdx.y + blockIdx.y * blockDim.y + 1; gtidy < c.NY -1; gtidy += dimy)
    for (auto gtidx = threadIdx.x + blockIdx.x * blockDim.x + 1; gtidx < c.NX -1; gtidx += dimx)
    { 
        auto tid = gtidx + gtidy*c.NX;
        Q[tid] += c.dt * c.d_dQdt[tid];
        P[tid] += c.dt * c.d_dPdt[tid];
    }
}

void step(const dim3& grid, const dim3& block, void** args){
    cudaLaunchKernel((void*)exchange_horizontal,grid, block, args); 
    cudaLaunchKernel((void*)exchange_vertical,  grid, block, args); 
    cudaLaunchKernel((void*)derivative,         grid, block, args);
    cudaLaunchKernel((void*)integrate,          grid, block, args);
}

/** Simulation of Chemicals
 *
 * @param num_of_iterations  The number of time steps to simulate
 * @param size               The x and y domain size
 * @param output_filename    The filename of the written Chemicals history
 */
void simulate(const Sim_Configuration &config) {
    using namespace std::chrono_literals;
    // We pad the world with ghost lines (two in each dimension)
    Chemicals c(config.size, config.size);

    double checksum = 0.0;
    real_t* chemicals_history = (real_t*)calloc(c.NX*c.NY*(config.iter/config.data_period + 1), sizeof(real_t));
    
    /** CUDA things **/
    c.copyIn();

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);

    int cu_nblocks;
    int cu_blocksize = config.block_x_size * config.block_y_size; //properties.maxThreadsPerBlock; //TODO: Could play around with different block sizes.
    void* kernelargs[] = {(void*)&c,(void*)&config.data_period};
    cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&cu_nblocks,(void*)integrate,cu_blocksize,0);

    dim3 grid(config.gridx,1,1);
    grid.y = config.gridy>0 ? config.gridy : floor(cu_nblocks*properties.multiProcessorCount/config.gridx);
    dim3 block(config.block_x_size,config.block_y_size,1);
    printf("( %d, %d, %d ) x ( %d, %d, %d )", grid.x, grid.y, grid.z, block.x, block.y, block.z); std::cout << std::endl;

    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < int(config.iter / config.data_period); i++){
        for (size_t j = 0; j < config.data_period; j++) step(grid, block, kernelargs);
        cudaMemcpy(&chemicals_history[i * (c.NX*c.NY)],c.d_P,sizeof(real_t)*c.NX*c.NY, cudaMemcpyDeviceToHost);
    }

    for (size_t j = 0; j < config.iter % config.data_period; j++) step(grid, block, kernelargs);
    cudaMemcpy(&chemicals_history[(config.iter/config.data_period) * (c.NX*c.NY)],c.d_P,sizeof(real_t)*c.NX*c.NY, cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    printLastCudaError("Something went wrong: ");
    /** End of CUDA **/

    /** If you want to check the output: **/
    to_file(c, chemicals_history , config.iter / config.data_period + 1, config.filename);
    c.copyOut();

    checksum += std::accumulate(c.P, c.P + c.NX*c.NY, 0.0);
    std::cout << "checksum: " << checksum << std::endl;
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << std::endl;
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    auto config = Sim_Configuration({argv, argv+argc});
    simulate(config);
    return 0;
}
