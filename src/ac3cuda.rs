#![allow(unused_imports)]


use std::{ffi::CString, fmt::Display};

use rustacuda::{prelude::*, error::CudaError};
use rustacuda::memory::DeviceBox;
use rustacuda_core::DevicePointer;
use std::collections::HashSet;

use crate::datatype::{Vec2, Map, Tilemap};
pub fn collapse_from_sample(sample: &Map<i32>, output_size: Vec2) -> Tilemap {
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
                output.push_str(&format!("Dir: {} | ", j)[..]);
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
    fn collapse_from_sample(sample: &Map<i32>, output_size: Vec2) -> Tilemap {
    
        let block_size = 512;
        let dims = (output_size.x as u32, output_size.y as u32);
        let size = dims.0 * dims.1;
        let blocks_needed = (size+block_size-1)/block_size;
 
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
        

        let mut output_map: Map<u32> = Map::new(output_size, Some(999));
        
        //output_map.data[0] = 1<<2;
        //output_map.data[1] = 1<<2;
        //output_map.data[2] = 1<<0;
        //output_map.data[(output_size.x*2) as usize] = 1<<0;
        output_map.data[(output_size.x*output_size.y) as usize - 1] = 1<<1;
        //output_map.data[(output_size.x*(output_size.y-3)) as usize - 1] = 1<<2;
        //output_map.data[((output_size.x)*(output_size.y-1)) as usize - 2] = 1<<0;
        
        // Allocate some memory for GPU-side buffers
        let mut dev_buffers: Vec<DeviceBuffer<u32>> = Vec::new();
        for _ in 0..3 {
            unsafe{ 
                dev_buffers.push(DeviceBuffer::uninitialized(size as usize).unwrap());
            }
        }
        dev_buffers[0].copy_from(&output_map.data[..]).unwrap();
        

        // changes_occured(_dev) track if any changes occur during propagation step
        let mut changes_occured_dev = DeviceBox::new(&0_i32).unwrap();
        let mut changes_occured: i32 = 1;

        let mut call_count: u32 = 0;
        while changes_occured != 0 {
            call_count+=1;   
            
            // Default the device state to 'no changes have occured (yet)'
            changes_occured_dev.copy_from(&0_i32).unwrap();

            // Run a single iteration of parallel AC3
            unsafe { 
                launch!(
                    module.iterate_ac<<<blocks_needed, block_size, 0, stream>>>(
                        dev_buffers[0].as_device_ptr(),
                        constraints_dev.as_device_ptr(),
                        output_map.size.x, output_map.size.y,
                        changes_occured_dev.as_device_ptr()
                    )
                ).unwrap();
            }
            stream.synchronize().unwrap();
            // Copy `changes_occured_dev` back to host memory
            changes_occured_dev.copy_to(&mut changes_occured).unwrap();
        }

        // Calculate the upper and lower bounds of all domains
        let (min_cell, max_cell);
        unsafe {
            (min_cell, max_cell) = CudaWavemap::parallel_bounds(
                size, &module, &stream, 
                dev_buffers[0].as_device_ptr(),
                dev_buffers[1].as_device_ptr()
            ).unwrap();
        }

        dev_buffers[1].copy_to(&mut output_map.data).unwrap();
        println!("{}", output_map);
        println!("blocks: {}x{} threads.\nTotal threads: {}", 
            blocks_needed, block_size, block_size*blocks_needed);
        println!("Called: {} times", call_count);

        println!("Domain Range: [{}, {}]", min_cell, max_cell);
        Tilemap::new(output_size, None)
    }

    unsafe fn parallel_collect_ids( value: u32, size: u32,
        module: &Module, stream: &Stream,
        input_buffer: DevicePointer<u32>,
        results_buffer: &mut DeviceBuffer<u32>)
        -> Result<Vec<u32>, CudaError> {

        let block_size = 512;
        let blocks_needed = (size+block_size-1)/block_size;
        let mut result_count_dev = DeviceBox::new(&0_u32)?;
        launch!(module.collect_ids<<<blocks_needed, block_size, 0, stream>>>
            (input_buffer, value, size, results_buffer.as_device_ptr(), 
             result_count_dev.as_device_ptr()))?;
        stream.synchronize()?;

        let mut result_count = 0_u32;
        result_count_dev.copy_to(&mut result_count)?;
        let (result_slice_dev, _) = results_buffer.split_at(result_count as usize);
        let mut result_vec = Vec::new();
        result_slice_dev.copy_to(&mut result_vec)?;
        Ok(result_vec)
    }

    unsafe fn parallel_bounds(size: u32,
        module: &Module, stream: &Stream,
        domain_buffer: DevicePointer<u32>,
        counts_buffer: DevicePointer<u32>)
        -> Result<(u32, u32), CudaError> {

        let block_size = 512;
        let blocks_count = (size+block_size-1)/block_size;
        let reduce_blocks = (blocks_count + 1) / 2;

        let mut final_result = u32::MAX;

        let mut final_val_dev = DeviceBox::new(&u32::MAX)?;
        let mut reduce_buffer = 
            DeviceBuffer::<u32>::uninitialized(reduce_blocks as usize)?;
        
        // Count the bits set in each domain and store the result in counts_buffer
        launch!(module.count_domains<<<blocks_count, block_size, 0, stream>>>
            (domain_buffer,counts_buffer,size))?;
        stream.synchronize()?;
        
        // Perform a parallel reduction, each block will output its result into
        // reduce_buffer[blockIdx]
        launch!(module.reduce_bounds<<<reduce_blocks, block_size, 4*block_size, stream>>>
            (counts_buffer, reduce_buffer.as_device_ptr(), size, 
             final_val_dev.as_device_ptr(), 1))?;
        stream.synchronize()?;
        
        let mut out_vec: Vec<u32> = Vec::with_capacity(reduce_blocks as usize);
        out_vec.resize(reduce_blocks as usize, u32::MAX);
        reduce_buffer.copy_to(&mut out_vec)?;
        println!("{:?}", out_vec.iter().map(|x| (x&0xFFFF, x>>16)).collect::<Vec<(u32,u32)>>());
        println!("reduce called with {} blocks of {} on size {}", reduce_blocks, block_size, size);
        
        // If needed, reduce blocks once more using a single block
        if blocks_count > 1 {
            let mut final_buffer = 
                DeviceBuffer::<u32>::uninitialized((blocks_count/2) as usize)?;

            launch!(module.reduce_bounds<<<1, block_size, 4*block_size, stream>>>
                (reduce_buffer.as_device_ptr(), final_buffer.as_device_ptr(), 
                 (blocks_count+1)/2, final_val_dev.as_device_ptr(), 0))?;
            stream.synchronize()?;
        }
        
        final_val_dev.copy_to(&mut final_result)?;

        Ok((final_result & 0xFFFF, final_result >> 16))
    }
}


