#include <stdlib.h>
#include <stdint.h>

#define NUM_DIRS 4
typedef enum {
    UP, DOWN, LEFT, RIGHT
} Direction;


/**
  @brief Set the value of each element to the position of the first set bit

  Used to convert a set of values to its represented integer value.

  @param bitfields The set of bitfields to convert
  @param length The number of elements
*/
extern "C"
__global__
void bitfield_to_id(uint32_t *bitfields, uint32_t length){
    uint32_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < length) {
        bitfields[id] = __ffs(bitfields[id])-1;
    }
}

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
inline __device__ 
int findConstraints(int32_t cell, int32_t *constraints, Direction dir) {
    if (!cell) { return 0; }

    int validDomain, offset, i;
    
    // `i` is the amount to shift `*cell` to have bit0 be a set bit
    // By extension, `i` also represents the tile id we are concerned with
    // Note: __ffs is used to provide hardware level integer intrinsics
    // and brings the count from worst case 31 asted ops to 0 wasted ops (per call)
    i = __ffs(cell)-1;
    // `validDomain` is the domain the cell in direction `dir` is allowed to be
    validDomain = 0;
    do {

        // Union the possible domains of each tile to determine the possible
        // domain of the cell in direction `dir`
        validDomain |= constraints[i*NUM_DIRS + dir];
        
        // Find the distance to the next set bit
        offset = __ffs(cell >> (i+1));
        i += offset;
        
        // Repeat if there is a next set bit
    } while (offset != 0);

    return validDomain;
}

/**
    @brief Perform a single step of AC3 propagation (in parallel)

    Constrains each cell using its neighbours imposed constraints.
    Note, that another cell might complete before one another resulting
    in a call to a different cell in a different step of propagation.
    This is okay given that cells cannot reach invalid states, as propagation
    will result in eventual consistency in subsequent calls.

    @param domains The array of bitfields to act on.
    @param constraints The set of all constraints.
    @param width The width of the 2D map.
    @param height The height of the 2D map.
    @param changesOccured Set to 1 if _any_ cell is modified.
  */
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

    int32_t domain = domains[x+y*width];
    int32_t origDomain = domain;
    // Intersect the restricted domain of each neighbour
    if (x > 0) {
        domain &= findConstraints(
                domains[x-1 + y*width],
                constraints,
                RIGHT);   // Neighbour is left, relativeDir is right (3)
    }
    if (x < width-1){
        domain &= findConstraints(
                domains[x+1 + y*width],
                constraints,
                LEFT);
    }
    if (y > 0) {
        domain &= findConstraints(
                domains[x + (y-1)*width],
                constraints,
                DOWN);
    }
    if (y < height-1) {
        domain &= findConstraints(
                domains[x + (y+1)*width],
                constraints,
                UP);
    }
    domains[x+y*width] = domain;
    // Set changesOccured return flag to 1 if the domain was altered
    if (origDomain != domain) { changed = 1; }

    // Minimize memory access to global memory by only having 1 thread per
    // block attempt to set it
    __syncthreads();
    if (threadIdx.x + threadIdx.y == 0 && changed) {
        *changesOccured = 1;
    }
}

/**
  @brief Construct a vector of element id's where `counts[id] == searchVal`

  @param values The array of values to search
  @param searchVal The value to search for
  @param length The # of elements in the values buffer
  @param results The buffer to store the found ids
  @param resCount The length of the vector
*/
extern "C"
__global__
void collect_ids(uint32_t *values, uint32_t searchVal, uint32_t length, 
        uint32_t *results, uint32_t *resCount){
    
    uint32_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < length && values[id] == searchVal){
        results[atomicAdd(resCount, 1)] = id;
    }

    // TODO: implement block-wide memory caches and perform clone to global
    // memory once per block
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
    if (i < length) {
        results[i] = __popc(domains[i]); // __popc provides hardware level bit counting
    }
}



/**
  @brief Parallel reduction to find min/max values of the input array)

  This performs a modified parallel reduction from the one explained by 
  Nvidia engineer Mark Harris
    https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  Interestingly, the warp unrolling optimization had no effect, as such it is
  excluded.
    
  The results are stored into a single integer, the least significant 16-bits
  representing the min, the others the max.
  Results can be found by either bitmasking 0xFFFF for min or shifting 16 right
  for max.

  Output must be reduced again until they have been reduced on a single block. That is,
  a call with 2 blocks would output results into index 0 and 1 of `results`, these
  must be reduced as well. This is as a result of no inter-block synchronization,
  using the kernel launch as the block sync barrier.

  Note, min will exclude values of 1 to allow for proper entropy selection in WFC

  @param values The array to find the min and max value of
  @param results The output results buffer
  @param length The number of elements in values
  @param firstReduction A flag to denote the values array are numbers, not bounds
*/
extern "C"
__global__
void reduce_bounds(uint32_t *values, uint32_t *results, uint32_t length, int firstReduction){
    extern __shared__ uint32_t s_results[];

    uint32_t i = threadIdx.x;
    uint32_t globalId = i + blockIdx.x * blockDim.x*2;
    if (globalId >= length) {
        if (i == 0){
            // Handle the case where the entire block is out of bounds
            // by setting domain result to [0xFFFF, 0]
            results[blockIdx.x] = 0xFFFF;
        }
        return;
    }
    
    // Used to determine if we are reducing a raw number or a previous iteration (value vs bound)
    uint32_t mask = firstReduction ? ~0 : 0xFFFF;
    uint32_t shift = firstReduction ? 0 : 16;
    // Init the bounds variables using the thread's globalId
    uint32_t val = values[globalId];
    uint32_t minVal = (val & mask) != 1 ? val & mask : 0xFFFF;
    uint32_t maxVal = val >> shift;
    
    // Perform the first reduction in place to avoid having 50% idle threads
    if (globalId + blockDim.x < length) {
        val = values[globalId+blockDim.x];
        minVal = min((val & mask) != 1 ? val & mask : 0xFFFF, minVal);
        maxVal = max(val >> shift, maxVal);
    }

    // Store the encoded result into shared memory for usage within the block
    s_results[i] = minVal | (maxVal<<16);
    __syncthreads();

    // Continually cut the stride in half, only allowing threads within
    // the stride to continue reduction. Eventually the block converges
    // to a single thread performing a single min/max.
    for (uint32_t stride=blockDim.x/2; stride > 0; stride>>=1) {
        if (i < stride && i + stride < length) {
            minVal = min(s_results[i + stride]&(0xFFFF), minVal);
            maxVal = max(s_results[i + stride]>>16, maxVal);
            s_results[i] = minVal | (maxVal<<16);
        }
        __syncthreads();
    }

    // Write the result of the block into the global result
    // Each block writes into its corresponding index
    if (i == 0) {
        results[blockIdx.x] = minVal | (maxVal<<16);
    }
}
