mod ac3;
mod datatype;

use datatype::Vec2;
use ac3::{Map, Wavemap};

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Path to sample text to use 
    #[clap(short, long = "sample")]
    sample: String,
    
    /// The width of the output
    #[clap(short='W', long, default_value_t = 16)]
    width: u32,

    /// The height of the output
    #[clap(short='H', long, default_value_t = 16)]
    height: u32,
}

fn main() {
    let args = Args::parse();
    let output_size = Vec2::new(args.width as i32, args.height as i32);
    
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

    let map = Wavemap::collapse_from_sample(&sample, output_size);
    
    // Map output to character set for a graphical printing
    const TILESET: [char; 3] = [' ', '~', '#'];
    for (i, domain) in map.data.iter().enumerate() {
        if i % map.size.x as usize == 0 {
            println!("");
        }
        
        print!("{} ", TILESET[*domain.iter().next().unwrap() as usize]);
    }
}
