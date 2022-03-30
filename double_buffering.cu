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
#include <cuda_pipeline_primitives.h>
#include <cuda/pipeline>

#define INLINE __device__ __forceinline__ 

typedef float real_t;
typedef float2 real2_t;
typedef float4 real4_t;

namespace cg = cooperative_groups;

INLINE float2 operator*(const float2& a, const float s)  { return {a.x*s, a.y*s};  }
INLINE float2 operator*(const float s, const float2& a)  { return a * s;  }
INLINE float2 operator+(const float2& a, const float2& b){ return {a.x+b.x, a.y+b.y};  }
INLINE void operator+=(float2& a, const float2& b) {a = a + b;}
INLINE bool operator==(const float2& a, const float2& b) {return a.x == b.x && a.y == b.y;}

INLINE float4 operator*(const float s, const float4& a)  { return {a.x*s, a.y*s, a.z*s, a.w*s};  }
INLINE float4 operator+(const float4& a, const float4& b){ return {a.x+b.x, a.y+b.y, a.z + b.z, a.w + b.w};  }
INLINE void operator+=(float4& a, const float4& b){a = a + b;  }

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
    real_t dx, dy, dt, dxdy;

    real_t* P; real_t* Q; real2_t* PQ; real2_t* PQ_Copy;

    std::vector<void*> h_buffers;

    real_t* d_P;
    real_t* d_Q;
    real2_t* d_PQ;   //RGB esque array for storing P and Q next to eachother in the same array.
    real2_t* d_PQ_Copy; // ----------------------------------- || ------------------------------------

    std::vector<std::pair<void**,size_t>> d_buffers = {{(void**)&d_P, sizeof(real_t)},{(void**)&d_Q, sizeof(real_t)}, {(void**)&d_PQ,sizeof(real2_t)}, {(void**)&d_PQ_Copy, sizeof(real2_t)}}; 
    Chemicals(size_t NX, size_t NY, real_t K = 9.0, real_t C = 4.5, real_t D_p = 1.0, real_t D_q = 8.0) : 
        NX(NX), NY(NY), K(K), C(C), D_p(D_p), D_q(D_q)
    {   
        P = (real_t*)calloc(NX*NY, sizeof(real_t));
        Q = (real_t*)calloc(NX*NY, sizeof(real_t));
        PQ = (real2_t*)calloc(NX*NY, sizeof(real2_t));
        PQ_Copy = (real2_t*)calloc(NX*NY, sizeof(real2_t));

        for (size_t i = 0; i < NX*NY; i++) {P[i] = real_t(0.0); Q[i] = real_t(0.0); PQ[i] = {real_t(0.0),real_t(0.0)};}

        for (size_t i = NY/4; i < NY - NY/4 ; ++i) 
        for (size_t j = NX/4; j < NX - NX/4 ; ++j) {
            Q[i*NX + j] = (K / C) + 0.2;
            P[i*NX + j] = C + 0.1;
            PQ[i*NX + j] = {C + real_t(0.1), (K / C) + real_t(0.2)};
        }

        //Largest approximate timestep determined empirically from function fitting.
        this->dt = 0.02 * std::pow(real_t(std::min(NX,NY))/40.0,-2);
        this->dx = 40/real_t(NX); this-> dy = 40.0/real_t(NY); dxdy = dx*dy;
        h_buffers = {P, Q, PQ, PQ_Copy};
        
        for (int i = 0; i<d_buffers.size(); i++) cudaMalloc(d_buffers[i].first, d_buffers[i].second*NX*NY);
    }

    void copyIn(){
        for (int i = 0; i<h_buffers.size(); i++) cudaMemcpy(*d_buffers[i].first, h_buffers[i], d_buffers[i].second*NX*NY, cudaMemcpyHostToDevice);
    }
    
    void copyOut(){
        for (int i = 0; i<h_buffers.size(); i++) cudaMemcpy(h_buffers[i], *d_buffers[i].first, d_buffers[i].second*NX*NY, cudaMemcpyDeviceToHost);
    }

    ~Chemicals(){
        for (int i = 0; i<d_buffers.size(); i++) cudaFree(*d_buffers[i].first);
        for (int i = 0; i<h_buffers.size(); i++) free(h_buffers[i]);
    }
};

void to_file(const Chemicals& c, real_t* history, const size_t frames, const std::string &filename){
    std::ofstream file(filename);
    file.write((const char*)(history), sizeof(real_t)*c.NX*c.NY*frames);
}

INLINE
unsigned int cidx(int id_y, int id_x){
    return threadIdx.x + 1 + id_x + (threadIdx.y +1 + id_y)* (blockDim.x+2);
}

INLINE
void exchange_horizontal(Chemicals& c){
    real2_t* PQ = c.d_PQ;
    for (size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y)*blockDim.x*gridDim.x ; tid < c.NX; tid += gridDim.x*blockDim.x*gridDim.y*blockDim.y)
    {
        PQ[tid] = PQ[tid + 2 * c.NX];                               //Top ghost cells = Top cells - 1
        PQ[tid + c.NX * (c.NY - 1)] = PQ[tid + c.NX * (c.NY - 3)];  //Bottom ghost cells = Bottom cells + 1
    }
}

INLINE
void exchange_vertical(Chemicals& c){
    real2_t* PQ = c.d_PQ;
    for (size_t tid = (threadIdx.x + blockIdx.x * blockDim.x) + (threadIdx.y + blockIdx.y * blockDim.y)*blockDim.x*gridDim.x ; tid < c.NY; tid += gridDim.x*blockDim.x*gridDim.y*blockDim.y)
    {   
        PQ[tid*c.NX] = PQ[tid*c.NX + 2];                            //Left ghost cells = Left cells + 1
        PQ[tid*c.NX + c.NX - 1] = PQ[tid*c.NX + c.NX - 3];          //Right ghost cells = Right cells - 1
    }
}

INLINE
void derivative(Chemicals& c, real2_t* s_PQ){
    auto ylim = c.NY + (blockDim.y - c.NY % blockDim.y);
    auto xlim = c.NX + (blockDim.x - c.NX % blockDim.x);
    for (auto gtidy = threadIdx.y + blockIdx.y * blockDim.y; gtidy < ylim; gtidy += gridDim.y*blockDim.y)
    for (auto gtidx = threadIdx.x + blockIdx.x * blockDim.x; gtidx < xlim; gtidx += gridDim.x*blockDim.x)
    {   
        real2_t PQ_center;
        real_t P2Q;
        auto tid = gtidx + gtidy*c.NX;
        bool valid_range = gtidx < c.NX - 1 && gtidy < c.NY -1 && gtidx > 0 && gtidy > 0;
        auto up = cidx(-1,0); auto down = cidx(1,0); auto right = cidx(0,1); auto left = cidx(0,-1); auto center = cidx(0,0);
        
        //Update interior cache cells.
        if (gtidx < c.NX && gtidy < c.NY){
            __pipeline_memcpy_async(&s_PQ[center],&c.d_PQ[tid], sizeof(real2_t));
        }

        //Update Cache halo cells.
        if (valid_range)
        {
            
            if (threadIdx.x == 0)              {__pipeline_memcpy_async( &s_PQ[left], &c.d_PQ[tid - 1], sizeof(real2_t));}
            if (threadIdx.x == blockDim.x -1)  {__pipeline_memcpy_async( &s_PQ[right], &c.d_PQ[tid + 1], sizeof(real2_t));}
            if (threadIdx.y == blockDim.y -1)  {__pipeline_memcpy_async( &s_PQ[down], &c.d_PQ[tid + c.NX], sizeof(real2_t));}
            if (threadIdx.y == 0)              {__pipeline_memcpy_async( &s_PQ[up], &c.d_PQ[tid - c.NX], sizeof(real2_t));}
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        if (valid_range){
            PQ_center = s_PQ[center];
            P2Q = PQ_center.x * PQ_center.x * PQ_center.y;
        }

        __syncthreads();
        if (valid_range){  
            real_t dPdt = c.dt * (c.D_p * ( s_PQ[left].x + s_PQ[right].x + s_PQ[up].x + s_PQ[down].x - real_t(4.0)*PQ_center.x) / (c.dxdy) + c.C + P2Q - (c.K + real_t(1.0)) * PQ_center.x);
            real_t dQdt = c.dt * (c.D_q * ( s_PQ[left].y + s_PQ[right].y + s_PQ[up].y + s_PQ[down].y - real_t(4.0)*PQ_center.y) / (c.dxdy) + PQ_center.x * c.K - P2Q);
            c.d_PQ_Copy[tid] = PQ_center + (real2_t){dPdt, dQdt};
        }
        __syncthreads();
    }
    cg::sync(cg::this_grid());

    //Swap pointers
    real2_t* temp = c.d_PQ;
    c.d_PQ = c.d_PQ_Copy;
    c.d_PQ_Copy = temp;
}

__global__
void kernel_simulate(Chemicals c, int iterations){
    extern __shared__ real2_t s_PQ[];
    for (int i = 0; i < iterations; i++)
    {   
        exchange_vertical(c);
        exchange_horizontal(c);
        cg::sync(cg::this_grid());
        derivative(c,s_PQ);;
    }
}

__global__
void prepare_buffer(Chemicals c){
    for (auto gtidy = threadIdx.y + blockIdx.y * blockDim.y; gtidy < c.NY; gtidy += gridDim.y*blockDim.y)
    for (auto gtidx = threadIdx.x + blockIdx.x * blockDim.x; gtidx < c.NX; gtidx += gridDim.x*blockDim.x)
    { 
        auto tid = gtidy * c.NX + gtidx;
        c.d_P[tid] = c.d_PQ[tid].x;
        c.d_Q[tid] = c.d_PQ[tid].y;
    }
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

    std::vector <std::vector<real_t>> water_history;
    double checksum = 0.0;
    real_t* chemicals_history; cudaMallocHost(&chemicals_history, sizeof(real_t)*c.NX*c.NY* (config.iter/config.data_period + 1));
    c.copyIn();
    cudaStream_t memcpy_stream, main_stream;
    cudaStreamCreateWithFlags(&memcpy_stream,cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&main_stream,cudaStreamNonBlocking);

    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties,0);
    int cu_nblocks;
    dim3 block(config.block_x_size,config.block_y_size,1);
    size_t smemsize = sizeof(real2_t)*(block.x + 2)*(block.y + 2);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor((int*)&cu_nblocks,(void*)kernel_simulate,block.x*block.y,smemsize);
    dim3 grid(config.gridx,1,1);
    grid.y = config.gridy>0 ? config.gridy : floor(cu_nblocks*properties.multiProcessorCount/config.gridx);
    printf("( %d, %d, %d ) x ( %d, %d, %d )", grid.x, grid.y, grid.z, block.x, block.y, block.z); std::cout << std::endl;

    void* kernelargs[] = {(void*)&c,(void*)&config.data_period}; void* buffer_args[] = {(void*)&c};

    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < int(config.iter / config.data_period); i++){
        cudaLaunchCooperativeKernel((void*)kernel_simulate, grid, block, kernelargs,smemsize, main_stream);
        cudaStreamSynchronize(main_stream); cudaStreamSynchronize(memcpy_stream);
        cudaLaunchKernel((void*)prepare_buffer, grid, block,buffer_args, 0, memcpy_stream);
        cudaMemcpyAsync(&chemicals_history[i * (c.NX*c.NY)],c.d_P,sizeof(real_t)*c.NX*c.NY, cudaMemcpyDeviceToHost, memcpy_stream);
    }
    cudaStreamSynchronize(memcpy_stream); cudaStreamSynchronize(main_stream);
    printLastCudaError("Main loop failed: ");

    size_t remaining_iterations = config.iter % config.data_period;
    void* kernelargsremainder[] = {(void*)&c, (void*)&remaining_iterations};
    cudaLaunchCooperativeKernel((void*)kernel_simulate, grid, block, kernelargsremainder, smemsize, main_stream);
    cudaLaunchKernel((void*)prepare_buffer, grid, block,buffer_args, 0, main_stream);
    cudaStreamSynchronize(main_stream);
    cudaMemcpy(&chemicals_history[(config.iter/config.data_period) * (c.NX*c.NY)],c.d_P,sizeof(real_t)*c.NX*c.NY, cudaMemcpyDeviceToHost);
    auto end = std::chrono::steady_clock::now();
    printLastCudaError("Something went wrong: ");
    
    /** If you want to check the output: **/
    to_file(c, chemicals_history , config.iter / config.data_period + 1, config.filename);
    c.copyOut();


    checksum += std::accumulate(c.P, c.P + c.NX*c.NY, 0.0);
    std::cout << "checksum: " << checksum << std::endl;
    std::cout << "elapsed time: " << (end - begin).count() / 1000000000.0 << " sec" << std::endl;
    cudaFreeHost(chemicals_history);
}

/** Main function that parses the command line and start the simulation */
int main(int argc, char **argv) {
    auto config = Sim_Configuration({argv, argv+argc});
    simulate(config);
    return 0;
}
