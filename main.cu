#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include "repeat.h"

#define CUDA_CALL(x) do {if((x)!=cudaSuccess) {                         \
            printf("Error at %s:%d ==> %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
            return EXIT_FAILURE;}} while(0)                             \

#define CHECK_PTR(x) do { if(x == NULL){                                \
            printf("Error at %s:%d ===> %s is NULL\n", __FILE__,__LINE__, #x); \
            exit(EXIT_FAILURE);}} while (0)                             \


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

void occMonitor(clock_t *tStart, clock_t *tEnd, int *sm_id, int blocks, int threads,
                int *maxOcc, int *wave, long int* maxLat)
{
    int numSM = mpc();
    int warps = max(1, blocks*threads/32);
    int wpb = max(1, BS/32);
    long int *minCycle, *maxCycle;
    minCycle=(long int*)malloc(sizeof(long int)*numSM);
    maxCycle=(long int*)malloc(sizeof(long int)*numSM);
    int *asgW = (int*)malloc(sizeof(int)*numSM);
    for (int sm = 0; sm < numSM; sm++){
        minCycle[sm] = LONG_MAX; // max of long int
        maxCycle[sm] = LONG_MIN; // min of long int
        asgW[sm] = 0;
    }
    //printf("Station 1, warps: %d\n", warps); fflush(stdout);
    for (int w = 0; w < warps; w ++){
        int sm = sm_id[w/wpb];
        //printf("w=%d, sm=%d\n", w, sm);
        asgW[sm]++;
        if (tStart[w] < minCycle[sm])
            minCycle[sm] = tStart[w];
        if (tEnd[w] > maxCycle[sm])
            maxCycle[sm] = tEnd[w];
    }


    int **occ = (int**)malloc(sizeof(int*)*numSM);
    for (int sm = 0; sm < numSM; sm ++){
        size_t occSize = sizeof(int)*(maxCycle[sm]-minCycle[sm]+1);
        occ[sm] = (int*)malloc(occSize);
        for (int i = 0; i < maxCycle[sm]-minCycle[sm]+1; i ++)
            occ[sm][i] = 0;
    }
    for (int w = 0; w < warps; w ++){
        int sm = sm_id[w/wpb];
        occ[sm][tStart[w]-minCycle[sm]] ++;
        occ[sm][tEnd[w]-minCycle[sm]] --;
    }

    printf("Occupancy Summary:\n");
    printf("smid  max\tave      \tlatency \ttotal\n");
    float sumOverAllSM = 0.0f;
    *maxLat=0;
    float sumWave = 0.0f;
    float sumThr = 0.0f;
    for (int sm = 0; sm < numSM; sm ++){
        int state = 0;
        int max = 0;
        long long int sum = 0;

        long int len = maxCycle[sm]-minCycle[sm]+1;
        for (int i = 0; i < len; i ++){
            state += occ[sm][i];
            occ[sm][i] = state;
            if (state > max) max = state;
            sum += state;
        }
        printf("%2d    %2d\t%f\t%ld \t%d\n", sm, max, (float)sum/(float)len, len-1, asgW[sm]);
        sumOverAllSM += (float)sum/(float)len;
        if ( (len-1) > *maxLat) *maxLat = len-1;
        sumWave += (float)asgW[sm] / (float)max;
        *maxOcc=max;
        sumThr += (float)ITER * asgW[sm] * ALPHA / (float)(len-1);
    }
    *wave = (int) (sumWave/numSM);
    printf("Ave Occupancy: %f\n",sumOverAllSM/numSM);
    printf("Max Occupancy: %d\n", *maxOcc);
    printf("Max Latency: %ld\n", *maxLat);
    printf("Ave Waves: %d\n", *wave);
    printf("Ave throughput: %f\n", sumThr/numSM);

    free (minCycle);
    free (maxCycle);
    free (asgW);
    for (int sm = 0; sm < numSM; sm ++)
        free (occ[sm]);
    free (occ);
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
    svalue[threadIdx.x] = laneid;
}
#endif



int main (int argc, char *argv[])
{
    /*
     * Array size = 512 MB = 64 M elements of 8-byte uintptr_t
     * The extra BS*8 byte is made for the last block's last iteration
     */
    int arrayLen = 64*1024*1024;
    size_t arraySize = arrayLen*8 + BS*8;
    uintptr_t *ptr_array = (uintptr_t*)malloc(arraySize);
    CHECK_PTR(ptr_array);
    uintptr_t *ptr_array_d = 0;
    /*
     * A global memory chunk of (4 GB +arraySize) bytes is allocated so that
     * there is always a chunk of memory inside the allocated chunk, whose
     * starting address has the form: 0xxxxxxxxx00000000, i.e. the lower 32
     * bits of the address is 0.
     *
     * - Why need to do that?
     *   The returned address from cudaMalloc is changing on eldar-11 (It could
     *   be the case that the GPU is shared by other processes on the server so
     *   that every time the OS allocates memory on different part of the device
     *   memory). The intention is make sure that the lower 32 bits of ptr_array_d's
     *   starting address is always 0.
     * - How to get the address for ptr_array_d?
     *   This is done by computing the offset between working_ptr.low32 and 0. Then
     *   ptr_array_d should be working_ptr + offset.
     */
    uint64_t workingSize = (uint64_t)4*1024*1024*1024 + arraySize; // 4 GB + arraySize
    void *working_ptr;
    cudaMalloc((void**)&working_ptr, workingSize);
    uint32_t low32_start = (uint32_t)(uint64_t)working_ptr;
    uint64_t offset;
    if (low32_start == 0){
        offset = 0;
    }
    else{
        offset = (uint64_t)(0xffffffff - low32_start + 1);
    }
    ptr_array_d = (uintptr_t*) ((uint64_t)working_ptr+offset);
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

    CUDA_CALL(cudaMemcpy (ptr_array_d, ptr_array, arraySize, cudaMemcpyHostToDevice));

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

    /*
     * On GTX 1080, get_shared_mem() returns the max shared memory per block (48 KB).
     * Since the total size of shared memory per SM is 96 KB, there are at least
     * 2 blocks per SM.
     */
    int activeBlocks = 96*1024/(SMEM*4);
    printf("blocks: %d, threads: %d, warps: %d, ", blocks, threads, warps);
    printf("#blocks/SM limited by smem: %d, ", activeBlocks);
    printf("Expected maxOcc: %d\n", activeBlocks*warpsPerBlock);

    /*
     * Initialize output array, one element for each warp
     */
    int *output = (int*)malloc(sizeof(int)*warps);
    CHECK_PTR(output);
    int *output_d;
    CUDA_CALL(cudaMalloc((void**)&output_d, sizeof(int)*warps));
    /*
     * Initialize array to record timestamps
     * One warp has one start and one end timestamps
     */
    clock_t *tStart = (clock_t*)malloc(sizeof(clock_t)*warps);
    CHECK_PTR(tStart);
    clock_t *tEnd = (clock_t*)malloc(sizeof(clock_t)*warps);
    CHECK_PTR(tEnd);
    clock_t *tStart_d, *tEnd_d;
    CUDA_CALL(cudaMalloc((void**)&tStart_d, sizeof(clock_t)*warps));
    CUDA_CALL(cudaMalloc((void**)&tEnd_d, sizeof(clock_t)*warps));
    /*
     * Initialize array to record sm_id for each warp
     * Note that warps from the same block share the same sm_id
     */
    int * sm_id = (int*)malloc(sizeof(int)*blocks);
    CHECK_PTR(sm_id);
    int * sm_id_d;
    CUDA_CALL(cudaMalloc((void**)&sm_id_d, sizeof(int)*blocks));


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

    CUDA_CALL(cudaGetLastError());

    CUDA_CALL(cudaMemcpy(output, output_d, sizeof(int)*warps, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(sm_id, sm_id_d, sizeof(int)*blocks, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tStart, tStart_d, sizeof(clock_t)*warps, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(tEnd, tEnd_d, sizeof(clock_t)*warps, cudaMemcpyDeviceToHost));
    /*
     * clock_t is long int, which is signed 8-byte data type
     */
    if (tStart[0] == 0){
        printf(">>>>> Ugly cycle numbers: tStart[0] = %ld \n", tStart[0]);
        exit(EXIT_FAILURE);
    }

    /*
     * Simple occupancy summary
     */
    int maxOcc, wave;
    long int maxLat;
    occMonitor(tStart, tEnd, sm_id, blocks, BS, &maxOcc, &wave, &maxLat);


    int writeToFile = 0;
    if (argc == 2){
        writeToFile = atoi(argv[1]);
    }
    /*
     * Write to file
     * Only write the tStart, tEnd and sm_id of each warp
     */
    if (writeToFile){
        char filename[64];
        sprintf(filename, "Pascal_CC61_result_%d_%d_%d_%ld_%d.txt", ALPHA, maxOcc, wave, maxLat, warps);
        FILE *fptr = fopen(filename, "w");
        CHECK_PTR(fptr);
        fprintf(fptr, "warpid\tsmid\tstart\t\tend\n");
        for (int w = 0; w < warps; w++){
            fprintf(fptr, "%5d\t%2d\t%ld\t%ld\n",
                    w, sm_id[w/warpsPerBlock], tStart[w], tEnd[w]);
        }
        fclose(fptr);
    }

    /* clean up */
    cudaFree (working_ptr);
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
