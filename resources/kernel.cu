#include <stdlib.h>
#include <cooperative_groups.h>

// `flags` is a group of bitfields for each cell in the map
// bit 0    -- Cell is collapsed
extern __shared__ int flags[];

extern "C" typedef enum {
    UP, DOWN, LEFT, RIGHT
} Direction;

// Up down left right

extern "C" __device__ int findConstraints(int *domain, int *constraints, 
        int numConstraints, int relativeDir) {
    
    int allowedDomain = 0;
    int tmpDomain = *domain;
    int i=0;
    // TODO potentially optimize
    while (tmpDomain) {
        // If the currently viewed bit is set then tile 'i' is in the domain
        if (tmpDomain & 1){
            // We need to union the allowed values based on this fact
            allowedDomain |= constraints[i*4 + relativeDir];
        }
        // Advance the cursor in the bitfield
        tmpDomain>>=1;
        ++i;
    }
    return allowedDomain;
}

extern "C" __device__ void propagate(int *buffer0, int *buffer1, int *constraints, 
    int numConstraints, int width, int height, int x, int y, int *activeBuffer) {
    __shared__ int propagating;

    do {
        propagating = 0;
        __syncthreads();

        // Swap buffers if needed
        if (!((*activeBuffer)%2)){
            int *tmp = buffer0;
            buffer0 = buffer1;
            buffer1 = tmp;
        }
        
        int *domain = &buffer0[x+y*width];
        int origDomain = *domain;
        // Intersect the restricted domain of each neighbour
        if (x > 0) {
            *domain &= findConstraints(
                    &buffer0[x-1 + y*width],
                    constraints,
                    numConstraints,
                    RIGHT);   // Neighbour is left, relativeDir is right (3)
        }
        if (x < width-1){
            *domain &= findConstraints(
                    &buffer0[x+1 + y*width],
                    constraints,
                    numConstraints,
                    LEFT);
        }
        if (y > 0) {
            *domain &= findConstraints(
                    &buffer0[x + (y-1)*width],
                    constraints,
                    numConstraints,
                    DOWN);
        }
        if (y < height-1) {
            *domain &= findConstraints(
                    &buffer0[x + (y+1)*width],
                    constraints,
                    numConstraints,
                    UP);
        }

        if (origDomain != *domain) { propagating = 1; }
        
        __syncthreads();
        if (x == 0 && y == 0){
            ++(*activeBuffer);
        }
        __syncthreads();

        // Propagate again until none are queued
    } while (propagating);     
}



extern "C" __global__ void collapse(int *buffer0, int *buffer1, int *constraints, 
        int numConstraints, int width, int height, int *resultBuffer){
    
    // blockCollapsed is a flag to denote every thread is assigned to a collapsed cell (end cond.)
    __shared__ int blockCollapsed;
    // TODO enable use of blocks, this will require block level synchronization
    int x = (threadIdx.x + blockDim.x * blockIdx.x) % width; 
    int y = (threadIdx.x + blockDim.x * blockIdx.x) / width; 
    if (x >= width || y >= height) { return; } // Ignore out of bounds threads
    
    // Init values before starting
    blockCollapsed = 0; // Each thread inits this but that doesn't matter due to sync
    flags[x+y*width] &= ~1;
    __syncthreads();


    while (!blockCollapsed) {
        // Call a propagation step, this is done until all values are stable
        propagate(buffer0, buffer1, 
                constraints, numConstraints, 
                width, height, x, y, 
                resultBuffer);
        return; 
        
        // Perform a test that the block is collapsed
        // TODO perform test as a parallel reduction
        // NOTE: Does not work with blocks, as only thread level sync is performed
        if (x == 0 && y == 0) {
            blockCollapsed = 1;
            for (int i = 0; i < width * height; ++i) {
                // Skip cells with collapsed flag set
                if (flags[i]&1) { continue; }
                
                // Clone the domain so we can bitshift it
                int testVal = buffer0[i];
                // If it is 0 we have issues (unsolvable)
                // TODO error handling of invalid state, backtracking maybe
                if (!testVal) { return; }

                // Iterate over each bit by shifting right each iteration
                while (testVal) {
                    // Check if the 0th-bit and another are set
                   if ((testVal & 1) && (testVal > 1)) {
                       // Block is not collapsed in the event of multiple set bits
                        blockCollapsed = 0;
                        goto doneTest;
                    }

                    testVal>>=1;
                }
                flags[i] |= 1;
            }  
doneTest:
        }
        __syncthreads();
    }
    
}


