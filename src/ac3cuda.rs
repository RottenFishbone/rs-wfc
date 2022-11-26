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

        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _context = Context::create_and_push(ContextFlags::MAP_HOST, device).unwrap();
        
        let module_str = CString::new(
            include_str!("../resources/kernel.ptx")
        ).unwrap();
        let module = Module::load_from_string(&module_str).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        
        let mut output_map: Map<u32> = Map::new(output_size, Some(999));
        
        //output_map.data[0] = 1<<2;
        //output_map.data[1] = 1<<2;
        //output_map.data[2] = 1<<0;
        //output_map.data[(output_size.x*2) as usize] = 1<<0;
        //output_map.data[(output_size.x*output_size.y) as usize - 1] = 1<<1;
        //output_map.data[((output_size.x)*(output_size.y-1)) as usize - 2] = 1<<0;
        
        let mut buffer_dev = unsafe {
            DeviceBuffer::uninitialized(output_map.data.len()).unwrap()
        };
        buffer_dev.copy_from(&output_map.data[..]).unwrap();
        
        let constraints = Constraints::from(sample);
        let constraints: Vec<i32> = constraints.0.into_iter().flatten().collect();
        let mut constraints_dev = unsafe {
            DeviceBuffer::uninitialized(constraints.len()).unwrap()
        };
        constraints_dev.copy_from(&constraints[..]).unwrap();
        
        let mut buffer0_dev = unsafe {
            DeviceBuffer::uninitialized(output_map.data.len()).unwrap()
        };
        buffer0_dev.copy_from(&output_map.data[..]).unwrap();
        
        let mut buffer1_dev = unsafe {
            DeviceBuffer::uninitialized(output_map.data.len()).unwrap()
        };
        buffer1_dev.copy_from(&output_map.data[..]).unwrap();
        

        let mut changes_occured_dev = DeviceBox::new(&0_i32).unwrap();
        
        let dims = (output_size.x as u32, output_size.y as u32);
        let size = dims.0 * dims.1;
        let blocks_needed = (size+block_size-1)/block_size;

        let mut changes_occured: i32 = 1;
        let mut call_count: u32 = 0;
        let mut result = 0;
        while changes_occured == 1 {
            // Default device state to 'no changes'
            changes_occured_dev.copy_from(&0_i32).unwrap();
            // Run a single iteration of parallel AC3
            unsafe { 
                launch!(
                    module.iterate_ac<<<blocks_needed, block_size, 0, stream>>>(
                        buffer_dev.as_device_ptr(),
                        constraints_dev.as_device_ptr(),
                        output_map.size.x, output_map.size.y,
                        changes_occured_dev.as_device_ptr()
                    )
                ).unwrap();
            }
            stream.synchronize().unwrap(); // Sync grid

            // Transfer shared memory to host to test if changes occured
            changes_occured_dev.copy_to(&mut changes_occured).unwrap();
            call_count+=1;   
            unsafe {
                result = CudaWavemap::parallel_min_domain(
                    size, &module, &stream, 
                    buffer_dev.as_device_ptr(),
                    buffer0_dev.as_device_ptr(),
                    buffer1_dev.as_device_ptr()).unwrap();
            }
        }


        buffer1_dev.copy_to(&mut output_map.data).unwrap();
        //println!("Domain sizes: {}", output_map);
        buffer_dev.copy_to(&mut output_map.data).unwrap();
        //println!("State (bitfields): {}", output_map);
        println!("blocks: {}x{} threads.\nTotal threads: {}", 
            blocks_needed, block_size, block_size*blocks_needed);
        println!("Called: {} times", call_count);

        println!("Smallest domain: {}", result);
        Tilemap::new(output_size, None)
    }

    //TODO select random cell using collect_ids kernel in unified memory

    /**
     * Performs parallel reduction in CUDA kernels to find the minimum and maximum
     * domain size at the same time.
     *
     * `count_buffer*` should be the same size as the `domain_buffer`.
     * One or both `count_buffer*`s will be modified.
     */
    unsafe fn parallel_min_domain(size: u32,
        module: &Module, stream: &Stream,
        domain_buffer: DevicePointer<u32>,
        count_buffer0: DevicePointer<u32>,
        count_buffer1: DevicePointer<u32>) 
        -> Result<u32, CudaError> {
        
        let block_size = 512;
        let mut blocks_count = (size+block_size-1)/block_size;

        let mut final_val_dev = DeviceBox::new(&u32::MAX)?;

        launch!(module.count_domains<<<blocks_count, block_size, 0, stream>>>
            (domain_buffer,count_buffer0,size))?;
        stream.synchronize()?;

        launch!(module.bit_count_min<<<blocks_count, block_size, 4*block_size, stream>>>
            (count_buffer0, count_buffer1, size, 
             final_val_dev.as_device_ptr()))?;
        stream.synchronize()?;
        
        // When true, denotes that buffer1 contains the data values
        let mut swap_buffers = true;
        
        // Continue reducing the values in place until reduction is performed within
        // a single block.
        while blocks_count > 1 {
            let prev_count = blocks_count;
            blocks_count += block_size-1;
            blocks_count /= block_size;

            let buf0 = if swap_buffers { count_buffer1 } else { count_buffer0 };
            let buf1 = if swap_buffers { count_buffer0 } else { count_buffer1 };
            launch!(module.bit_count_min<<<blocks_count, block_size, 4*block_size, stream>>>
                (buf0, buf1, prev_count, final_val_dev.as_device_ptr()))?;
            stream.synchronize()?;
            swap_buffers = !swap_buffers;
        }
        
        let mut final_result = u32::MAX;
        final_val_dev.copy_to(&mut final_result)?;

        Ok(final_result)
    }
}


