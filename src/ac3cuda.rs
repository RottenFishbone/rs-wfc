#![allow(unused_imports)]


use std::{ffi::CString, fmt::Display};

use rand::prelude::*;
use rand::seq::SliceRandom;

use rustacuda::{prelude::*, error::CudaError};
use rustacuda::memory::{DeviceBox, UnifiedBox};
use rustacuda_core::DevicePointer;
use std::collections::HashSet;

use crate::datatype::{Vec2, Map, Tilemap};

pub fn collapse_from_sample(sample: &Map<i32>, output_size: Vec2) -> Option<Tilemap> {
    CudaWavemap::collapse_from_sample(sample, output_size)
}


struct Constraints(Vec<[i32; 4]>);
// Implement pretty printing for Constraints
impl Display for Constraints {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Iterate over each tile and direction, outputting their ruleset
        let mut output = String::new();
        for (i, _) in self.0.iter().enumerate() {
            output.push_str(&format!("Tile: {}\n------\n", i)[..]);
            
            for j in 0..4 {
                let valid_tiles = &self.0[i][j];
                let dir_str:String = crate::datatype::Direction::from(j as u32).into();
                output.push_str(&format!("Dir: {} | ", dir_str.chars().next().unwrap())[..]);
                output.push_str(&format!("{:#034b}", valid_tiles));
                output.push_str("\n");
            }
            if i < self.0.len()-1 {
                output.push_str("\n");
            }
        }
        write!(f, "{}", output)
    }
}
impl From<&Tilemap> for Constraints {
    fn from(sample: &Tilemap) -> Self {
        let max_cell = *(sample.data.iter().max().unwrap());
        let mut constraints = Constraints(Vec::with_capacity(max_cell as usize));
        let constraint_data = &mut constraints.0;

        for _ in 0..max_cell+1 {
            constraint_data.push([0_i32,0,0,0]);
        }

        for (i, cell) in sample.data.iter().enumerate() {
            for (dir, nbr_id) in sample.neighbour_list[i].iter().enumerate() {
                if let Some(nbr_id) = nbr_id {
                    constraint_data[*cell as usize][dir] |= 1<<(sample.data[*nbr_id]);
                }
            }
        }
        constraints
    }
}
struct CudaWavemap(Map<i32>);
impl CudaWavemap {
    fn collapse_from_sample(sample: &Map<i32>, output_size: Vec2) -> Option<Map<i32>> {
        let dims = (output_size.x as u32, output_size.y as u32);
        let size = dims.0 * dims.1;
 
        // Init CUDA/RustaCUDA
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _context = Context::create_and_push(ContextFlags::MAP_HOST, device).unwrap();
        let module_str = CString::new(include_str!("../resources/kernel.ptx")).unwrap();
        let module = Module::load_from_string(&module_str).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        
        // Build constraint information
        let constraints = Constraints::from(sample);
        let constraints: Vec<i32> = constraints.0.into_iter().flatten().collect();
        let mut constraints_dev = 
            unsafe {DeviceBuffer::uninitialized(constraints.len()).unwrap()};
        constraints_dev.copy_from(&constraints[..]).unwrap();
        
        
        // Initial domain (all possible values set)
        // Take full set and shift off the amount of cells we need to be 32(bits) - max_domain
        let initial_val = (!0u32) >> (32u32 - (constraints.len() as u32)/4);
        let mut output_map: Map<i32> = Map::new(output_size, Some(initial_val as i32));
        
        // Allocate memory on GPU
        let (mut cache_buffer, mut cell_buffer, mut counts_buffer, mut min_cells_buffer);
        unsafe {
            // Holds cell domain bitsets
            cell_buffer = DeviceBuffer::uninitialized(size as usize).unwrap();
            // Caches arc-consistent states
            cache_buffer = DeviceBuffer::uninitialized(size as usize).unwrap();
            // Holds counts of remaining bits per cell
            counts_buffer = DeviceBuffer::uninitialized(size as usize).unwrap();
            // Holds list of cells with lowest entropy (held on GPU)
            min_cells_buffer = DeviceBuffer::uninitialized(size as usize).unwrap();
        }
        // Clone uncollapsed states to GPU
        cell_buffer.copy_from(&output_map.data[..]).unwrap();
        cache_buffer.copy_from(&cell_buffer).unwrap();
        
        // changes_occured(_dev) track if any changes occur during propagation step
        let mut changes_occured_dev = DeviceBox::new(&0_i32).unwrap();
        let mut rng = rand::thread_rng();
        
        // Holds the untested list of bits for selected cell
        let mut bit_set = HashSet::<i32>::new();
        let mut selected_cell_id = 0;

        // Iterate until max_cell == 1 (all cells hold a single value) or min_cell == 0
        loop {
            
            let mut changes_occured: i32 = 1;
            while changes_occured != 0 {
                // Default the device state to 'no changes have occured (yet)'
                changes_occured_dev.copy_from(&0_i32).unwrap();
                
                // Run a single iteration of parallel AC3
                unsafe { 
                    parallel_propagate(output_size, &module, &stream,
                        cell_buffer.as_device_ptr(),
                        constraints_dev.as_device_ptr(),
                        changes_occured_dev.as_device_ptr()
                    ).unwrap();
                }
                stream.synchronize().unwrap();
                // Copy `changes_occured_dev` back to host memory
                changes_occured_dev.copy_to(&mut changes_occured).unwrap();
            }

            // Calculate the upper and lower bounds of all domains (excluding 1's for min)
            let (min_cell, max_cell);
            unsafe {
                (min_cell, max_cell) = parallel_bounds(
                    size, &module, &stream, 
                    cell_buffer.as_device_ptr(),
                    counts_buffer.as_device_ptr()
                ).unwrap();
            }
            
            if max_cell == 1 { 
                // We are in an arc-consistent, collapsed state (yay)
                break; 
            }
            else if min_cell == 0 { 
                // We hit an invalid cell state
                
                if bit_set.is_empty() {
                    // Simple backtracking is no longer possible
                    return None;
                }
                
                // Revert the cell_buffer to cached state
                cell_buffer.copy_from(&cache_buffer).unwrap();
            }
            else {
                // We are in an arc-consistent, uncollapsed state.
                
                // Cache the arc-consistent cell state before collapse
                cache_buffer.copy_from(&cell_buffer).unwrap();
                
                // Find all cells matching min_cell
                unsafe {
                    let num_choices = parallel_match_cells(min_cell, size, 
                        &module, &stream,
                        counts_buffer.as_device_ptr(),
                        &mut min_cells_buffer).unwrap();
                    
                    // Create a list of choices (done to allow for implementation of more adv
                    // backtracking
                    let cell_choices: Vec<i32> = (0..num_choices).map(|i| i as i32).collect();
                    // Select one from the list
                    let rand_id = *(cell_choices.iter().choose(&mut rng).unwrap()) as usize;
                    // Grab the selected id from the list of matching cells
                    let mut selected_cell_container = [0];
                    min_cells_buffer[rand_id..(rand_id+1)].copy_to(&mut selected_cell_container[..]).unwrap();
                    selected_cell_id = selected_cell_container[0];
                    // Grab the actual cell from the cell buffer (converted to a hashset of bits)
                    bit_set = get_bit_set(&cell_buffer, selected_cell_id);
                }    
            }
            
            // Pop a possible bit from the set and try collapsing it
            let chosen_bit = bit_set.iter().choose(&mut rng).unwrap().clone();
            bit_set.remove(&chosen_bit);
            // Push the collapsed value to cell buffer
            let target_range = (selected_cell_id as usize)..=(selected_cell_id as usize);
            cell_buffer[target_range].copy_from(&[1<<chosen_bit]).unwrap();
        }

        unsafe {
            // Convert all the bitfields into their index value
            parallel_convert_bitfields(size, &module, &stream, 
                cell_buffer.as_device_ptr()).unwrap();
        }

        // Copy the final output back to the CPU
        cell_buffer.copy_to(&mut output_map.data).unwrap();
        Some(output_map)
    }
}

fn get_bit_set(buffer: &DeviceBuffer<i32>, selected_cell_id: i32) -> HashSet<i32> {
    // Hacky way to copy single value into a container using ranges
    // Copy cell value from device into selected_cell_container at index 0
    let mut selected_cell_container: Vec<i32> = vec![0];
    let cell_range = (selected_cell_id as usize)..(selected_cell_id+1) as usize;
    buffer[cell_range.clone()].copy_to(&mut selected_cell_container).unwrap();
    // Unwrap the container into a single integer val
    let mut selected_cell_val = selected_cell_container[0];

    // Break the integer into a vec of _set_ bit value (indicies)
    let mut bit_set = HashSet::<i32>::new();
    let mut i = 0;
    while selected_cell_val > 0 {
        if (selected_cell_val & 1) == 1 { bit_set.insert(i); }
        selected_cell_val>>=1;
        i+=1;
    }

    bit_set
}

/**
 * Performs a single iteration of AC3 on the GPU.
 */
unsafe fn parallel_propagate(size: Vec2,
    module: &Module, stream: &Stream,
    cell_buffer: DevicePointer<i32>, constraint_buffer: DevicePointer<i32>, changed: DevicePointer<i32> )
    -> Result<(), CudaError> {
 
        let block_size = 512;
        let blocks_needed = ((size.x*size.y)+block_size-1)/block_size;
        launch!(
            module.iterate_ac<<<blocks_needed as u32, block_size as u32, 0, stream>>>(
                cell_buffer,
                constraint_buffer,
                size.x, size.y,
                changed
            )
        )?;

        Ok(())
}

/**
 * Matches all cells within the input buffer equaling `value`
 * The results are stored into the results_buffer and its length is returned.
 */
unsafe fn parallel_match_cells( value: i32, size: u32,
    module: &Module, stream: &Stream,
    input_buffer: DevicePointer<i32>,
    results_buffer: &mut DeviceBuffer<i32>)
    -> Result<u32, CudaError> {
    
        let block_size = 512;
        let blocks_needed = (size+block_size-1)/block_size;

        let mut result_count = UnifiedBox::new(0_u32)?; // Output vector size

        launch!(module.collect_ids<<<blocks_needed, block_size, 4*(block_size+1), stream>>>
            (input_buffer, value, size, results_buffer.as_device_ptr(), 
             result_count.as_unified_ptr()))?;
        stream.synchronize()?;
        
        Ok(*result_count)
}

/**
 * Peforms a parallel reduction on the GPU to calculate the lower and upper bound
 * of the domain_buffer.
 *
 * The domain buffer is counted, the result being placed into counts_buffer and
 * using that data a reduction can be performed to calculate bounds. The reduction
 * buffers are discarded as only the final value is of use.
 */
unsafe fn parallel_bounds(size: u32,
    module: &Module, stream: &Stream,
    domain_buffer: DevicePointer<i32>,
    counts_buffer: DevicePointer<i32>)
    -> Result<(i32, i32), CudaError> {

        let block_size = 512;
        // Block count is calculated as the number of block_size blocks to contain `size`
        let mut block_count = (size+block_size-1)/block_size;
        // Reduce blocks is half that number, as reduction's first iteration is 2*blocksize
        let mut reduce_blocks = (block_count + 1) / 2;

        // Count the bits set in each domain and store the result in counts_buffer
        launch!(module.count_domains<<<block_count, block_size, 0, stream>>>
            (domain_buffer,counts_buffer,size))?;
        stream.synchronize()?;
        
        // Allocate unified memory to place the result from each block into
        let mut reduce_buffer0 = 
            UnifiedBuffer::<i32>::uninitialized(reduce_blocks as usize)?;
        // Note: buffer1 requires less space as it is used after the initial
        // reduction (thus its size is reduced once again)
        let mut reduce_buffer1 = 
            UnifiedBuffer::<i32>::uninitialized(
                (((reduce_blocks+block_size-1)/block_size+1)/2) as usize)?;

        // Perform a parallel reduction, each block will output its result into
        // reduce_buffer[blockIdx]
        launch!(module.reduce_bounds<<<reduce_blocks, block_size, 4*block_size, stream>>>
            (counts_buffer, reduce_buffer0.as_unified_ptr(), size, 1))?;
        stream.synchronize()?;


        // Continue reduction as many times as needed until a launch with a single
        // block is performed
        let mut swap_buffers = false;
        let mut last_block_launch = reduce_blocks;
        while last_block_launch > 1 {
            block_count = (reduce_blocks+block_size-1)/block_size;
            reduce_blocks = (block_count+1) / 2;

            // Handle swapping reduction between the two buffers
            // Buffers are not resized as its better to just reuse them
            let buf0 = 
                if swap_buffers { reduce_buffer1.as_unified_ptr() } 
                else { reduce_buffer0.as_unified_ptr() };
            let buf1 = 
                if swap_buffers { reduce_buffer0.as_unified_ptr() } 
                else { reduce_buffer1.as_unified_ptr() };
            swap_buffers = !swap_buffers;
            
            launch!(module.reduce_bounds<<<reduce_blocks, block_size, 4*block_size, stream>>>
                (buf0, buf1, last_block_launch, 0))?;
            stream.synchronize()?;
            
            last_block_launch = reduce_blocks;
        }
        
        let final_result = if swap_buffers {
            reduce_buffer1.first().unwrap().clone()
        } else {
            reduce_buffer0.first().unwrap().clone()
        };
    
        Ok((final_result & 0xFFFF, final_result >> 16))
}

/**
 * Uses the GPU to convert all the collapsed bitfields into the actual
 * index they represesnt.
 *
 * This can be used to convert from the CUDA-compatible internal representation
 * of domains into our regular internal representation, which is a Map of integers
 * each representing the content-index of the cell.
 */
unsafe fn parallel_convert_bitfields(size: u32, 
    module: &Module, stream: &Stream,
    buffer: DevicePointer<i32>)
    -> Result<(), CudaError> {
    
    let block_size = 512;
    let blocks_needed = (size+block_size-1)/block_size;

    launch!(module.bitfield_to_id<<<blocks_needed, block_size, 0, stream>>>
        (buffer, size))?;

    Ok(())
}


