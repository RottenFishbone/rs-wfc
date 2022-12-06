use crate::datatype::{Tilemap, Vec2};
use std::{path::Path, collections::HashSet};

use unicode_segmentation::UnicodeSegmentation;

/**
 * Abstraction of the intermediate format `Tilemap` and the user's input format.
 * Implementing this trait allows a user to convert from an arbitrary type into a `Tilemap` for 
 * use in WFC.
 */
pub trait Converter {
    /**
     * Outputs data into a file using the converter to place it into the specified filetype
     */
    fn output_map(&self, collapsed_map: &Tilemap, path: &Path) -> Result<(), std::io::Error>;
    
    /**
     * Load data from a source file and store relevant conversion data into the Converter
     */
    fn build_sample(&mut self, path: &Path) -> Result<Tilemap, std::io::Error>;
}

/**
 * The Unicode-Tilemap converter. This handles conversion between
 * Unicode samples and the native format (`Tilemap`) of the WFC modes.
 */
#[derive(Default)]
pub struct UnicodeConverter {
    char_map: Vec<String>,
}
impl UnicodeConverter {
    pub fn new() -> Self {
        Self { char_map: Vec::new() }
    }
}
impl Converter for UnicodeConverter {
    fn output_map(&self, collapsed_map: &Tilemap, path: &Path) -> Result<(), std::io::Error> {
        if path.parent().is_none() {
            panic!("Attempted to write to path that does not exist.");
        }

        let mut output = String::new();
        for (i, cell) in collapsed_map.data.iter().enumerate() {
            let cell = *cell as usize;
            if cell > self.char_map.len() {
                panic!("Attempted to convert to Unicode glyph using a cell with an index larger than possible.");
            }
            // Push newlines at the end of a row
            if i % collapsed_map.size.x as usize == 0 && i > 0 {
                output.push('\n');
            }
            // Push the value as the correct character
            output.push_str(&self.char_map[cell]);

        }
        
        std::fs::write(path, output)
    }

    fn build_sample(&mut self, path: &Path) -> Result<Tilemap, std::io::Error> {
        let input = std::fs::read_to_string(path)?;

        // Find the dimensions of the sample
        let width = input.lines().next()
            .expect("Empty sample file provided.")
            .graphemes(true)
            .count() as i32;
        let height = input.lines().count() as i32;
       
        let mut added = HashSet::<String>::new(); // Used for deduplication in char_map
        let mut sample = Tilemap::new(Vec2::new(width, height), Some(0)); // The output sample
        
        let mut i = 0; // counter for number of chars added to sample.data
        
        // Iterate over every line of input (strips newlines)
        for line in input.lines() {
            // Chunk the line into graphemes ('characters' in unicode, sorta)
            let graphemes = line.graphemes(true).collect::<Vec<&str>>();

            for glyph in graphemes {
                // Clone the glyph from a slice into a string
                let glyph = String::from(glyph);
                // Insert into HashSet if first encounter, this prevents duplication
                // Also insert into character map, assigning a new ID to each glyph (the position)
                if !added.contains(&glyph) {
                    added.insert(glyph.clone());
                    self.char_map.push(glyph.clone());
                }

                // Find the character's id in the charmap
                let index = self.char_map.iter().position(|c| *c == glyph).unwrap();
                // Push as the id to the sample map output
                sample.data[i] = index as i32;
                i+=1;
            }
        }


        Ok(sample)
    }
}
