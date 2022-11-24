use crate::datatype::{Vec2, Map, Tilemap};

use std::{
    collections::{
        hash_set::HashSet,
        VecDeque,
    },
    fmt::{Debug, Display}
};
use rand::seq::IteratorRandom;

pub fn collapse_from_sample(sample: &Map<i32>, output_size: Vec2) -> Tilemap {
    Ac3Wavemap::collapse_from_sample(sample, output_size)
}

/// The defined rules for every possible tile in a `Wavemap`.
///
/// Constraints can be derived from a sample `Tilemap` using `into()/from()`.
/// Constraints for a tile can be accessed using `get_constraints()` 
#[derive(Debug)]
struct Constraints(Vec<[HashSet<i32>; 4]>);         
impl Constraints {
    /// Return the set of constraints in a `direction` for a specified `tile`.
    fn get_constraints(&self, tile: i32, direction: usize) -> &HashSet<i32> {
        &self.0[tile as usize][direction as usize]
    }
}
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
                for tile in valid_tiles {
                    output.push_str(&format!("{} ", *tile)[..]);
                }
                output.push_str("\n");
            }
            if i < self.0.len()-1 {
                output.push_str("\n");
            }
        }
        write!(f, "{}", output)
    }
}
// Implement From<Tilemap> to allow to building Constraints from a sample map
impl From<&Tilemap> for Constraints {
    fn from(sample: &Tilemap) -> Self {
        let max_cell = *(sample.data.iter().max().unwrap());
        let mut constraints = Constraints(Vec::with_capacity(max_cell as usize));
        let constraint_data = &mut constraints.0;
        
        // Foreach tile up to `max_cell`, push an array of empty domains
        for _ in 0..max_cell+1 {
            constraint_data.push([HashSet::new(), HashSet::new(), HashSet::new(), HashSet::new()]);
        }
        
        // Iterate over each cell in the sample
        for (i, cell) in sample.data.iter().enumerate() {
            // Iterate over all neighbour Options
            for (dir, nbr_id) in sample.neighbour_list[i].iter().enumerate() {
                // Filter None values
                if let Some(nbr_id) = nbr_id {
                    // Insert the neighbours value into the relevant constraint set
                    constraint_data[*cell as usize][dir].insert(sample.data[*nbr_id]);
                }
            }
        }
    
        // Return the populated constraints
        constraints
    }
}

/// Wavemap is a Map containing domains used during the WFC algorithm
struct Ac3Wavemap(Map<HashSet<i32>>);
impl From<Ac3Wavemap> for Map<i32> {
    fn from(wavemap: Ac3Wavemap) -> Self {
        let mut map = Self::new(wavemap.0.size, None);
        for cell in wavemap.0.data.iter() {
            let cell_val = *cell.iter().next().unwrap();
            map.data.push(cell_val);
        }
        
        map
    }
}
// Implement helpers for wave function collapse maps
impl Ac3Wavemap {
    pub fn collapse_from_sample(sample: &Map<i32>, output_size: Vec2) -> Tilemap {
        // Create a set of constraints using the sample
        let constraints = Constraints::from(sample);
        
        // Build a set of all possible values using the sample
        let domain: HashSet<i32> = HashSet::from_iter(sample.data.clone());

        // Create a wavemap with each variable holding the full domain
        let mut domains = Ac3Wavemap(Map::new(output_size, Some(domain)));

        while !domains.is_collapsed() {
            let collapsed = domains.collapse_lowest();
            domains.propagate(collapsed, &constraints);
        }

        domains.into()
    }

    /// Finds the lowest entropy cell and collapses domain into a single choice
    pub fn collapse_lowest(&mut self) -> usize{
        let cell_choice = self.least_entropy();
        let map = &mut self.0;
        // Extract all options in domain
        let domain = map.data[cell_choice].drain();
        
        // Grab a random one and add it back
        let mut rng = rand::thread_rng();
        let final_choice = domain.choose(&mut rng).unwrap();
        map.data[cell_choice].insert(final_choice);

        // Return with cell was chosen
        return cell_choice;
    }

    /// Scans the Wavemap for the lowest entropy cells and chooses a random one from the list
    /// Panics on empty Wavemap.
    pub fn least_entropy(&self) -> usize {
        let map = &self.0;
        // Find the lowest length, non-1, domain by folding the domains into the minimum value
        let lowest_entropy = map.data.iter()
            .fold(usize::MAX, |acc, cell| {
                if cell.len() == 0 { panic!("Illegal state. Aborting collapse."); }
                else if cell.len() != 1 { cell.len().min(acc) }
                else { acc }
            });
        // Note: if lowest_entropy is usize::MAX then the Wavemap was collapsed before calling
        // TODO handle this case, which *if* used correctly would never arise.
        
        // Get every cell('s id) of the lowest entropy
        let lowest_cells: Vec<usize> = map.data.iter().enumerate()
            .filter(|(_, cell)| cell.len() == lowest_entropy)   // Filter by lowest entropy
            .map(|(i, _)| i)                                    // Remap output to be the cell's id
            .collect();                                         // Collect into a Vec
        
        // Grab a random choice from the lowest cells
        let mut rng = rand::thread_rng();
        return *lowest_cells.iter().choose(&mut rng).unwrap();
    }

    /// Propagates changes in constraints to the rest of the Wavemap
    pub fn propagate(&mut self, root_id: usize, constraints: &Constraints) {
        let map = &mut self.0;
        // Store all pending constraint checks into a queue
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(root_id);
        
        while queue.len() > 0 {
            let cell_id = queue.pop_front().unwrap(); // Cell will always exist, as queue.len > 0
            let cell_domain = map.data[cell_id].clone();

            // Check each valid cell neighbour for changed constraints
            for (dir, nbr_id) in map.neighbour_list[cell_id].iter().enumerate() {
                if let Some(nbr_id) = nbr_id {
                    let mut constrained: HashSet<i32> = HashSet::new();
                    
                    for tile in cell_domain.iter() {
                        let rules = constraints.get_constraints(*tile, dir);
                        for rule in rules {
                            constrained.insert(*rule);
                        }
                    }

                    let intersection: Vec<i32> = constrained
                        .intersection(&map.data[*nbr_id])
                        .map(|x| *x)
                        .collect();

                    if intersection.len() < map.data[*nbr_id].len() {
                        queue.push_back(*nbr_id);
                        map.data[*nbr_id] = HashSet::from_iter(intersection.into_iter());
                    }
                }
            }
        }
    }

    /// Scans wavemap for an uncollapsed cell. Returns true if at least one cell isn't collapsed
    pub fn is_collapsed(&self) -> bool {
        for cell in self.0.data.iter() {
            if cell.len() > 1 {
                return false;
            }
        }

        true
    }
}


