#include <stdlib.h>
#include <stdint.h>

#define NUM_DIRS 4
typedef enum {
    UP, DOWN, LEFT, RIGHT
} Direction;


/**
    @brief Calculate imposed constraints by a cell's current domain

    Determines the imposed constraints in a specified direction
    from a specified cell. Uses each possible value in `cell` to union
    the rules together from `constraints` in the direction `dir`.

    The output should be intersected by the cell in direction `dir` to
    constrain its domain to be arc consistent.
    
    @param cell A pointer to the imposing cell's domain
    @param constraints A pointer to the array of constraints
    @param dir The direction the cell is imposing constraints
    @return Complete set of allowed values for domain in direction `dir`
 */
extern "C" 
__device__ 
inline
int findConstraints(int32_t *cell, int32_t *constraints, Direction dir) {
    if (!(*cell)) { return; }
    int validDomain, offset, i;
    
    // `i` is the amount to shift `*cell` to have bit0 be a set bit
    // By extension, `i` also represents the tile id we are concerned with
    // Note: __ffs is used to provide hardware level integer intrinsics
    // and brings the count from worst case 30 wasted ops to 0 wasted ops
    i = __ffs(*cell)-1;
    // `validDomain` is the domain the cell in direction `dir` is allowed to be
    validDomain = 0;
    do {

        // Union the possible domains of each tile to determine the possible
        // domain of the cell in direction `dir`
        validDomain |= constraints[i*NUM_DIRS + dir];
        
        // Find the distance to the next set bit
        offset = __ffs((*cell) >> i+1);
        i += offset;
        
        // Repeat if there is a next set bit
    } while (offset != 0);

    return validDomain;
}

extern "C" 
__global__ 
void iterate_ac(int32_t *domains, int32_t *constraints,
        int32_t width, int32_t height, int32_t *changesOccured){

    __shared__ int changed;
    changed = 0;
    __syncthreads();

    int32_t x = (threadIdx.x + blockDim.x * blockIdx.x) % width; 
    int32_t y = (threadIdx.x + blockDim.x * blockIdx.x) / width; 
    if (x >= width || y >= height) { return; }

    int32_t *domain = &domains[x+y*width];
    int32_t origDomain = *domain;
    // Intersect the restricted domain of each neighbour
    if (x > 0) {
        *domain &= findConstraints(
                &domains[x-1 + y*width],
                constraints,
                RIGHT);   // Neighbour is left, relativeDir is right (3)
    }
    if (x < width-1){
        *domain &= findConstraints(
                &domains[x+1 + y*width],
                constraints,
                LEFT);
    }
    if (y > 0) {
        *domain &= findConstraints(
                &domains[x + (y-1)*width],
                constraints,
                DOWN);
    }
    if (y < height-1) {
        *domain &= findConstraints(
                &domains[x + (y+1)*width],
                constraints,
                UP);
    }

    // Set changesOccured return flag to 1 if the domain was altered
    if (origDomain != *domain) { changed = 1; }

    // Minimize memory access to global memory by only having 1 thread per
    // block attempt to set it
    __syncthreads();
    if (threadIdx.x + threadIdx.y == 0 && changed) {
        *changesOccured = 1;
    }
}

/**
  @brief Construct a vector of element id's where `counts[id] == searchVal`

  @param counts The array of values to search
  @param searchVal The value to search for
  @param results The buffer to store the found ids
  @param counter The length of the vector
*/
extern "C"
__global__
void collect_ids(uint32_t *counts, uint32_t searchVal, uint32_t length, 
        uint32_t *results, uint32_t *resCount){

    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= length || counts[i] != searchVal) { return; }
    results[atomicAdd(resCount,1)] = i;
}


/**
  @brief Counts the length of each domain and outputs the result into a new buffer

  @param domains The array containing bitfields representing domain values
  @param results The output buffer to store the # of values in corres. domain
  @param length The length of both the domains and the buffer.
*/
extern "C"
__global__
void count_domains(int32_t *domains, uint32_t *results, uint32_t length){
    uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= length) { return; } // Prevent OOB access
    results[i] = __popc(domains[i]); // __popc provides hardware level bit counting
}

extern "C"
__global__
void bit_count_min(uint32_t *counts, uint32_t *results, uint32_t length, uint32_t *result){
    extern __shared__ uint32_t s_results[];

    uint32_t i = threadIdx.x;
    uint32_t minVal = counts[i + blockIdx.x*blockDim.x];
    
    if (i + blockDim.x < length) {
        minVal = min(counts[i+blockDim.x], minVal);
    }

    s_results[i] = minVal;
    __syncthreads();

    // Continually cut the stride in half, only allowing threads within
    // the stride to continue reduction. Eventually the block converges
    // to a single thread.
    for (uint32_t stride=blockDim.x/2; stride > 0; stride /= 2) {
        if (i < stride) {
            minVal = min(s_results[i + stride], minVal);
            s_results[i] = minVal;
        }
        __syncthreads();
    }

    // Write the result of the block into the global result
    // Each block writes into its corresponding index
    if (i == 0) {
        results[blockIdx.x] = minVal;
        // Also emit a single value, this is only useful on the last 
        // kernel launch
        *result = minVal;
    }
}
