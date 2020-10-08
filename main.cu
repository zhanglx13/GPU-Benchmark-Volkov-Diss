#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include "repeat.h"


/*
 * alpha=0, used to determine #iterations
 */
//#define PURE_ARITH

/*
 * #iter is set, now it ready to run the throughput experiment
 */
// #define EXPERIMENT
/*
 * blocksize: 128 threads, i.e. 4 warps per block
 */
#ifndef BS
#define BS 128
#endif

/*
 * Parameter iteration. Default value is set to 1000
 */
#ifndef ITER
#define ITER 1000
#endif

/*
 * Parameter alpha, which is the arithmetic intensity.
 * default is set to 32
 */
#ifndef ALPHA
#define ALPHA 32
#endif

/*
 * Shared memory (#integers) per block, which is used to control
 * the occupancy.
 * The #blocks limited by shared memory is computed as
 *   sharedMemPerBlock / (SMEM*4)
 * The default SMEM is set to 2048. On TITAN, the default max #blocks
 * is set to be 48 * 1024 / (2048*4) = 6 blocks.
 */
#ifndef SMEM
#define SMEM 2048
#endif

/*
 * Here we make alpha a compile-time parameter
 * Check the following post to see why a second level of indirection
 * is required:
 * https://wiki.sei.cmu.edu/confluence/display/c/PRE05-C.+Understand+macro+replacement+when+concatenating+tokens+or+performing+stringification
 */
#define conc(a,b) conc_again(a,b)
#define conc_again(a,b) a##b

/*
 * Obtain the number of SMs
 * https://forums.developer.nvidia.com/t/how-does-clock-work/110849/5?u=lxzhang
 */
int mpc(){
   cudaDeviceProp prop;
   cudaGetDeviceProperties(&prop, 0);
   return prop.multiProcessorCount;
}

int get_shared_mem()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return (int)prop.sharedMemPerBlock;
}

/* E.D. Riedijk */
/**
 * Obtained from Robert
 * https://devtalk.nvidia.com/default/topic/1070285/cuda-programming-and-performance/how-does-clock-work/post/5422889/#5422889
 */

static __device__ __inline__ int get_smid(void) {

    int ret;

    asm volatile("mov.u32 %0, %smid;" : "=r"(ret) );

    return ret;

}

/*******************************************************************************
 * Pure arith kernel
 *
 * No memory instruction involved in the instruction mix
 * No shared memory is used

 * This kernel is used to determine the number of iterations.
 * According to volkov's diss, this kernel has alpha=\inf and the single
 * warp latency should be about 10⁶ cycles.
 *
 * When ALPHA=32, ITER should be 4096 to make the latency of a single
 * warp reach 10⁶ cycles
 ******************************************************************************/
#ifdef PURE_ARITH

__global__ void pure_arith (
    int *output,
    clock_t *tStart,
    clock_t *tEnd,
    int * sm_id,
    float a,
    float b)
{
    b=b-1; // make sure b is 0
    int pid = threadIdx.x + blockIdx.x*blockDim.x;
    int warpid = pid / 32;
    int laneid = pid % 32;
    clock_t start, end;

    start = clock();
#pragma unroll 16
    for (int i = 0; i < ITER; i++) {
        conc(repeat, ALPHA)(a=a+b;);
    }
    end=clock();

    if (laneid == 0){
        tStart[warpid] = start;
        tEnd[warpid] = end;
        output[warpid] = __float_as_int(a);
        if (threadIdx.x == 0)
            sm_id[blockIdx.x] = get_smid();
    }
}
#endif

/*******************************************************************************
 * Experiment kernel
 *
 * Use pointer chasing to perform global memory load in each group
 * Shared memory is used to control the occupancy
 ******************************************************************************/
#ifdef EXPERIMENT

__global__ void chase_pointers (
    uintptr_t *ptr_array,
    int *output,
    clock_t *tStart,
    clock_t *tEnd,
    int * sm_id,
    float b)
{
    __shared__ int svalue[SMEM];
    b=b-1; // make sure b is 0
    float a;
    int pid = threadIdx.x + blockIdx.x*blockDim.x;
    int warpid = pid / 32;
    int laneid = pid % 32;

    clock_t start, end;
    union
    {
        uintptr_t * addr;
        struct {
            int low;
            int high;
        } myInt;
    } myUnion;
    /*
     * Starting index of each thread is given by
     *   threadID + blockID * space
     * Since each block of threads access consecutive ITER*BS elements,
     * space = ITER * BS
     */
    int startIdx = threadIdx.x + blockIdx.x * ITER * BS;
    myUnion.addr = &ptr_array[startIdx];
    start = clock();
#pragma unroll 16
    for (int i = 0; i < ITER; i++) {
        myUnion.addr = (uintptr_t *)(*myUnion.addr);
        a=__int_as_float(myUnion.myInt.low);
        conc(repeat, ALPHA)(a=a+b;);
        myUnion.myInt.low = __float_as_int(a);
    }
    end=clock();

    if (laneid == 0){
        tStart[warpid] = start;
        tEnd[warpid] = end;
        output[warpid] = myUnion.myInt.low;
        if (threadIdx.x == 0){
            sm_id[blockIdx.x] = get_smid();
        }
    }
    svalue[blockIdx.x] = laneid;
}
#endif



int main (int argc, char *argv[])
{
    cudaError_t error_id;
    /*
     * Array size = 512 MB = 64 M elements of 8-byte uintptr_t
     * The extra BS*8 byte is made for the last block's last iteration
     */
    size_t arraySize = 512*1024*1024 + BS*8;
    int arrayLen = 64*1024*1024;
    uintptr_t *ptr_array = (uintptr_t*)malloc(arraySize);
    uintptr_t *ptr_array_d = 0;
    cudaMalloc ((void **)&ptr_array_d, arraySize);
    /*
     * The array is initialized so that
     * array[i] = &array[i+N], where N is blocksize (BS)
     *
     */
    for (int i = 0; i < arrayLen; i++){
        ptr_array[i] = (uintptr_t)&ptr_array_d[i+BS];
    }
    /*
     * use 8-bit LFSR to initialize array
     * From njuffa: https://forums.developer.nvidia.com/t/how-to-eliminate-address-computation-instructions-in-the-sass-code/156083/4?u=lxzhang
     */
    // int mask, state, new_state;
    // mask = 0x8e;
    // state = 1;
    // do {
    //     new_state = (state & 1) ? ((state >> 1) ^ mask) : (state >> 1);
    //     ptr_array [state] = (uintptr_t)&ptr_array_d [new_state];
    //     state = new_state;
    // } while (state != 1);
    // ptr_array[0] = (uintptr_t)&ptr_array_d[1];

    cudaMemcpy (ptr_array_d, ptr_array, arraySize, cudaMemcpyHostToDevice);

#ifdef EXPERIMENT
    /*
     * When memory instructions are involved, #blocks is determined by the array, iter
     * and BS.
     */
    int blocks = arrayLen/BS/ITER;
#endif
#ifdef PURE_ARITH
    /*
     * When no memory instruction is involved, the #blocks is chosen to be
     * 100 * #active blocks
     * And it is assumed the only limit of #active blocks on each SM is 16
     */
    int blocks = 100*16*mpc();
#endif
    int threads = BS;
    int warps = max(1, blocks*threads/32);
    int warpsPerBlock = max(1, threads/32);

    printf("blocks: %d, threads: %d, warps: %d, warpsPerBlock: %d\n",
           blocks, threads, warps, warpsPerBlock);
    printf("sharedMemPerBlock: %d bytes\n", get_shared_mem());
    printf("#blocks limited by smem: %d\n", get_shared_mem()/(SMEM*4));
    printf("Occupancy limited by smem: %d warps\n",BS/32 * get_shared_mem()/(SMEM*4));

    /*
     * Initialize output array, one element for each warp
     */
    int *output = (int*)malloc(sizeof(int)*warps);
    int *output_d;
    cudaMalloc((void**)&output_d, sizeof(int)*warps);
    /*
     * Initialize array to record timestamps
     * One warp has one start and one end timestamps
     */
    clock_t *tStart = (clock_t*)malloc(sizeof(clock_t)*warps);
    clock_t *tEnd = (clock_t*)malloc(sizeof(clock_t)*warps);
    clock_t *tStart_d, *tEnd_d;
    cudaMalloc((void**)&tStart_d, sizeof(clock_t)*warps);
    cudaMalloc((void**)&tEnd_d, sizeof(clock_t)*warps);
    /*
     * Initialize array to record sm_id for each warp
     * Note that warps from the same block share the same sm_id
     */
    int * sm_id = (int*)malloc(sizeof(int)*blocks);
    int * sm_id_d;
    cudaMalloc((void**)&sm_id_d, sizeof(int)*blocks);


#ifdef EXPERIMENT
    chase_pointers<<<blocks,threads>>>(ptr_array_d, output_d, tStart_d, tEnd_d, sm_id_d, 1);
    chase_pointers<<<blocks,threads>>>(ptr_array_d, output_d, tStart_d, tEnd_d, sm_id_d, 1);
    chase_pointers<<<blocks,threads>>>(ptr_array_d, output_d, tStart_d, tEnd_d, sm_id_d, 1);
    chase_pointers<<<blocks,threads>>>(ptr_array_d, output_d, tStart_d, tEnd_d, sm_id_d, 1);
#endif

#ifdef PURE_ARITH
    pure_arith <<<blocks, threads>>> (output_d, tStart_d, tEnd_d, sm_id_d, 1.0, 1.0);
    pure_arith <<<blocks, threads>>> (output_d, tStart_d, tEnd_d, sm_id_d, 1.0, 1.0);
    pure_arith <<<blocks, threads>>> (output_d, tStart_d, tEnd_d, sm_id_d, 1.0, 1.0);
    pure_arith <<<blocks, threads>>> (output_d, tStart_d, tEnd_d, sm_id_d, 1.0, 1.0);
    pure_arith <<<blocks, threads>>> (output_d, tStart_d, tEnd_d, sm_id_d, 1.0, 1.0);
#endif

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        printf("Kernel launch error:  %s\n", cudaGetErrorString(error_id));
    }

    cudaMemcpy(output, output_d, sizeof(int)*warps, cudaMemcpyDeviceToHost);
    cudaMemcpy(sm_id, sm_id_d, sizeof(int)*blocks, cudaMemcpyDeviceToHost);
    cudaMemcpy(tStart, tStart_d, sizeof(clock_t)*warps, cudaMemcpyDeviceToHost);
    cudaMemcpy(tEnd, tEnd_d, sizeof(clock_t)*warps, cudaMemcpyDeviceToHost);
    /*
     * clock_t is long int, which is signed 8-byte data type
     */
    int writeToFile = 0;
    if (argc == 2){
        writeToFile = atoi(argv[1]);
    }
    /*
     * Write to file
     * Only write the tStart, tEnd and sm_id of each warp
     */
    if (writeToFile){
        char filename[32];
        sprintf(filename, "result_%d_%d_%d.txt", ITER, ALPHA, BS);
        FILE *fptr = fopen(filename, "w");
        fprintf(fptr, "warpid\tsmid\tstart\t\tend\n");
        for (int w = 0; w < warps; w++){
            fprintf(fptr, "%5d\t%2d\t%ld\t%ld\n",
                    w, sm_id[w/warpsPerBlock], tStart[w], tEnd[w]);
        }
        fclose(fptr);
    }

    /* clean up */
    cudaFree (ptr_array_d);
    cudaFree (output_d);
    cudaFree (tStart_d);
    cudaFree (tEnd_d);
    cudaFree (sm_id_d);

    free(ptr_array);
    free(output);
    free(tStart);
    free(tEnd);
    free(sm_id);
    cudaDeviceSynchronize();
    return EXIT_SUCCESS;
}
