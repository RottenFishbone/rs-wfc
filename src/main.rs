#[macro_use]
extern crate rustacuda;

mod ac3;
mod ac3cuda;

mod datatype;

use std::fmt::Display;

use datatype::{Vec2, Map, Tilemap};

use clap::{Parser, ValueEnum};

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd, Ord, ValueEnum)]
enum Mode {
    /// Sequential AC3
    Ac3,
    /// CUDA-propagated AC3
    Ac3Cuda,
}
impl Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mode::Ac3 => f.write_str("ac3"),
            Mode::Ac3Cuda => f.write_str("ac3-cuda"),
        }
    }
}

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    sample: String,
    
    /// The width of the output
    #[clap(short='W', long, default_value_t = 32)]
    width: u32,

    /// The height of the output
    #[clap(short='H', long, default_value_t = 32)]
    height: u32,

    #[clap(short='m', default_value_t = Mode::Ac3)]
    mode: Mode,
}

fn main() {
    let args = Args::parse();
    let output_size = Vec2::new(args.width as i32, args.height as i32);
    
    // TODO create a process to make a sample data structure from character representation
    let sample = match std::fs::read_to_string(args.sample) {
        Ok(sample_str) => {
            let width = match sample_str.lines().next() {
                Some(line) => line.len(),
                None => {
                    eprintln!("Empty sample file. Aborting.");
                    return;
                }
            };
            let height = sample_str.lines().count();
            
            let sample_chars: Vec<i32> = sample_str.chars()
                .filter(|c| *c != '\n' && *c != '\r')
                .map(|c| c.to_digit(10).unwrap() as i32)
                .collect();
            
            let mut sample = Map::new(Vec2::new(width as i32, height as i32), None);
            sample.data = sample_chars;
            sample
        },
        Err(err) => {
            eprintln!("Error while opening sample file: {:?}", err);
            return;
        }
    };
    

    let output: Tilemap = match args.mode {
        Mode::Ac3 => ac3::collapse_from_sample(&sample, output_size),
        Mode::Ac3Cuda => ac3cuda::collapse_from_sample(&sample, output_size),
    };
    
    // TODO use sample datastructure to rebuild character representation
    // Map output to character set for a graphical printing
    /*const TILESET: [char; 5] = [' ', 'i', '|','#','.'];
    for (i, cell) in output.data.iter().enumerate() {
        if i % output.size.x as usize == 0 {
            println!("");
        }
        
        let id = *cell as usize;//((*cell) as f32).log2() as usize;
        print!("{} ", TILESET[id]);
    }*/
}
