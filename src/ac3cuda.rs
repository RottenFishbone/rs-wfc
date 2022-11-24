#![allow(unused_imports)]


use std::{ffi::CString, fmt::Display};

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use std::collections::HashSet;

use crate::datatype::{Vec2, Map, Tilemap};
pub fn collapse_from_sample(sample: &Map<i32>, output_size: Vec2) -> Tilemap {
    println!("{}", Constraints::from(sample));
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
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let _context = Context::create_and_push(ContextFlags::MAP_HOST, device).unwrap();
        
        let module_str = CString::new(
            include_str!("../resources/kernel.ptx")
        ).unwrap();
        let module = Module::load_from_string(&module_str).unwrap();

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();
        
        let mut output_map: Map<i32> = Map::new(output_size, Some(7));
        output_map.data[0] = 1<<2;
        output_map.data[2] = 1<<0;
        output_map.data[(output_size.x*2) as usize] = 1<<0;
        let mut buffer1_dev = unsafe {
            DeviceBuffer::uninitialized(output_map.data.len()).unwrap()
        };
        let mut buffer2_dev = unsafe {
            DeviceBuffer::uninitialized(output_map.data.len()).unwrap()
        };
        buffer1_dev.copy_from(&output_map.data[..]).unwrap();
        buffer2_dev.copy_from(&output_map.data[..]).unwrap();
        
        let constraints = Constraints::from(sample);
        let constraints: Vec<i32> = constraints.0.into_iter().flatten().collect();
        let mut constraints_dev = unsafe {
            DeviceBuffer::uninitialized(constraints.len()).unwrap()
        };
        constraints_dev.copy_from(&constraints[..]).unwrap();
        let mut res_buffer_dev = DeviceBox::new(&0_u32).unwrap();

        unsafe {
            let dims = (output_size.x as u32, output_size.y as u32);
            launch!(module.collapse<<<10, 512, (dims.0*dims.1*4), stream>>>(
                buffer1_dev.as_device_ptr(),
                buffer2_dev.as_device_ptr(),
                constraints_dev.as_device_ptr(),
                constraints.len()/4,
                output_map.size.x,
                output_map.size.y,
                res_buffer_dev.as_device_ptr()
            )).unwrap();
        }

        stream.synchronize().unwrap();

        let mut res_in_buffer_host = 0;
        res_buffer_dev.copy_to(&mut res_in_buffer_host).unwrap();
        if res_in_buffer_host == 0 {
            buffer1_dev.copy_to(&mut output_map.data).unwrap();
        } else {
            buffer2_dev.copy_to(&mut output_map.data).unwrap();
        }
        println!("{}", output_map);
        println!("{:?}", res_in_buffer_host);
        
        Tilemap::new(output_size, None)
    }
}


