#[macro_use] extern crate rustacuda;

mod ac3;
mod ac3cuda;

mod converter;
mod datatype;

use std::fmt::Display;

use converter::{UnicodeConverter, Converter};
use datatype::{Vec2,Tilemap};

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
    /// Path to input sample
    sample: String,
    
    /// The width of the output
    #[clap(short='W', long, default_value_t = 32)]
    width: u32,

    /// The height of the output
    #[clap(short='H', long, default_value_t = 32)]
    height: u32,

    /// Algorithm to perform WFC with
    #[clap(short='m', default_value_t = Mode::Ac3)]
    mode: Mode,

    /// Filename to output final result into.
    #[clap(short='o', default_value_t = String::from("wfc_out"))]
    output: String,

    /// Abort the collapse when an invalid state is reached.
    #[clap(short='A', default_value_t = false)]
    abort_on_fail: bool
}

fn main() {
    let args = Args::parse();
    let output_size = Vec2::new(args.width as i32, args.height as i32);
    let sample_path = std::path::PathBuf::from(args.sample);
    if !sample_path.exists() {
        panic!("Provided path does not exist");
    }

    let mut converter = UnicodeConverter::new();
    let sample = converter.build_sample(&sample_path).unwrap();
    
    let mut output: Option<Tilemap>;
    loop {
        output = match args.mode {
            Mode::Ac3 => ac3::collapse_from_sample(&sample, output_size),
            Mode::Ac3Cuda => ac3cuda::collapse_from_sample(&sample, output_size),
        };
        
        if output.is_some() || args.abort_on_fail { break; }
        println!("Retrying...");
    }
    
    let output = match output {
        Some(val) => val,
        None => { eprintln!("Aborted on invalid state."); panic!();},
    };
    
    converter.output_map(&output, &std::path::PathBuf::from(args.output)).unwrap();
    println!("Done.");
}
